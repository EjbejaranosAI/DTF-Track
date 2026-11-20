"""Dynamic tracking ROS 2 node — LiDAR-based Kalman tracking with optional YOLO metadata."""

import json
import math
import os
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

StampKey = Tuple[int, int]

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as BuiltinTime
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
import message_filters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray
from vision_msgs.msg import Detection2DArray

try:
    from pose_estimation_interfaces.msg import DotYoloCropArray
except ImportError:
    DotYoloCropArray = None  # type: ignore

from .camera_geometry import (
    all_tracks_laser_polar,
    bearing_laser_from_optical_ray,
    bearing_laser_from_track,
    project_tracks_to_image,
    scaled_intrinsics,
)
from .identity import IdentityManager, ReIdConfig
from .diagnostics import LidarDiagnostics, PersonTrackLogger, PipelineTiming, PoseDebugLogger, SpencerExtractDiagnostics, StageTimer
from .map_filters import (
    AreaFilter, CostmapFilter, StaticMapFilter, load_zones_file, parse_zone,
)
from .marker_remap import remap_marker_array
from .leg_pair import merge_leg_pairs
from .lidar_detection import LidarExtractDebugStats
from .lidar_detection.pipeline import build_lidar_pipeline
from .lidar_detection.detectors.blob import BlobDetectorParams
from .lidar_detection.pipeline import extract_lidar_clusters, extract_lidar_dual
from .scan_clustering import (
    LegHeuristicParams, repair_interleaved_returns, scan_window_best_range,
)
from .tf_utils import MarkerRemapTF, TFHelper
from .track_pipeline import (
    apply_yolo_auxiliary_person_confirmation,
    apply_yolo_auxiliary_person_rejection,
    tracks_to_detection_array,
    update_tracks,
    update_yolo_auxiliary_camera_fov_state,
)
from .track_visualization import (
    build_area_zone_marker_array,
    build_lidar_cluster_marker_array,
    build_track_marker_array,
    build_yolo_laser_polar_debug_marker_array,
    track_label_compact_line,
)
from .types import LidarMeasurement, PendingLidarSpawn, Track, YoloMeasurement
from .yolo_backends import (
    OnnxYoloBackend,
    RawYoloDet,
    UltralyticsYoloBackend,
    YoloInferenceBackend,
    resolve_yolo_onnx_path,
)


# ─── Helpers ────────────────────────────────────────────────────────


def resolve_zones_file(name: str) -> str:
    """Absolute/relative path as given, else a name inside share/dynamic_tracking/config/zones/."""
    path = os.path.expanduser(name)
    if os.path.isfile(path):
        return path
    zones_dir = os.path.join(get_package_share_directory("dynamic_tracking"), "config", "zones")
    for cand in (name, name + ".yaml"):
        cand_path = os.path.join(zones_dir, cand)
        if os.path.isfile(cand_path):
            return cand_path
    raise FileNotFoundError(f"area_filter_zones_file '{name}' not found (nor in {zones_dir})")


def _resolve_yolo_device(device_raw: str) -> Tuple[str, str]:
    raw = str(device_raw).strip().lower()
    if raw not in ("auto", ""):
        return str(device_raw).strip(), "fixed by param"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0", "CUDA GPU (auto)"
        mps_b = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_b is not None and mps_b.is_available():
            return "mps", "Apple MPS (auto)"
    except Exception as exc:
        return "cpu", f"CPU (error checking GPU: {exc})"
    return "cpu", "CPU (no CUDA/MPS, auto)"


_RawYoloDet = RawYoloDet


class _YoloBackgroundWorker:
    """Runs YOLO inference on a background thread; results indexed by image stamp."""

    _MAX_PENDING = 2
    _MAX_COMPLETED = 20

    def __init__(
        self,
        backend: YoloInferenceBackend,
        is_target_fn,
        logger,
        keep_latest_only: bool = True,
        on_annotated=None,
        stage_timer=None,
        reid_embedder=None,
        reid_cache=None,
        reid_period_sec: float = 0.5,
        reid_skip_overlap_iou: float = 0.3,
    ) -> None:
        self._backend = backend
        self._is_target = is_target_fn
        self._logger = logger
        self._keep_latest_only = keep_latest_only
        # Si se da, se llama desde este hilo al terminar cada inferencia
        # (publica la imagen anotada sin esperar al siguiente frame de camara).
        self._on_annotated = on_annotated
        self._stage_timer = stage_timer
        # Re-ID: embeddings OSNet de las cajas person en un hilo propio (reid_bg): YOLO no
        # espera a la ReID (con la GPU limitada, p.ej. con bateria, un embedding tardaba
        # ~175 ms y frenaba las detecciones). El pipeline los recoge de `reid_cache` por bbox.
        self._reid = reid_embedder
        self._reid_cache = reid_cache
        self._reid_period = max(0.0, float(reid_period_sec))
        self._reid_overlap_iou = float(reid_skip_overlap_iou)
        self._reid_last_wall = 0.0
        self._reid_job = None
        self._reid_event = threading.Event()
        self._reid_job_lock = threading.Lock()
        if self._reid is not None:
            threading.Thread(target=self._reid_loop, daemon=True, name="reid_bg").start()

        self._result_lock = threading.Lock()
        self._latest_boxes: List[_RawYoloDet] = []
        self._latest_stamp_sec: float = 0.0
        self._latest_yolo_ms: float = 0.0

        self._pending_lock = threading.Lock()
        self._pending_queue: deque = deque()

        self._done_lock = threading.Lock()
        self._completed: "OrderedDict[StampKey, Tuple[List[_RawYoloDet], float]]" = OrderedDict()
        self._annotated_lock = threading.Lock()
        self._annotated_queue: deque = deque()
        self._logged_first_infer = False

        self._thread = threading.Thread(target=self._loop, daemon=True, name="yolo_bg")
        self._thread.start()

    def submit(self, cv_image: np.ndarray, stamp_key: StampKey, frame_id: str = "") -> None:
        with self._pending_lock:
            if self._keep_latest_only:
                self._pending_queue.clear()
            self._pending_queue.append((cv_image.copy(), stamp_key, str(frame_id or "")))
            while len(self._pending_queue) > self._MAX_PENDING:
                self._pending_queue.popleft()

    def get_latest(self) -> Tuple[List[_RawYoloDet], float, float]:
        with self._result_lock:
            return list(self._latest_boxes), self._latest_stamp_sec, self._latest_yolo_ms

    def drain_annotated(
        self,
    ) -> List[Tuple[np.ndarray, StampKey, List[_RawYoloDet], str]]:
        with self._annotated_lock:
            items = list(self._annotated_queue)
            self._annotated_queue.clear()
        return items

    def take_boxes_for_stamp(self, stamp_key: StampKey, max_wait: float) -> Tuple[List[_RawYoloDet], float]:
        deadline = time.perf_counter() + max(0.0, max_wait)
        while True:
            with self._done_lock:
                if stamp_key in self._completed:
                    boxes, ms = self._completed.pop(stamp_key)
                    return list(boxes), ms
            if time.perf_counter() >= deadline:
                return [], 0.0
            time.sleep(0.001)

    def _store(self, key: StampKey, boxes: List[_RawYoloDet], ms: float) -> None:
        with self._done_lock:
            self._completed[key] = (boxes, ms)
            while len(self._completed) > self._MAX_COMPLETED:
                self._completed.popitem(last=False)

    def set_reid_rate(self, hz: float) -> None:
        self._reid_period = (1.0 / hz) if hz > 0 else 0.0

    def _submit_reid(self, cv_image: np.ndarray, boxes: List[_RawYoloDet]) -> None:
        """Desde el hilo YOLO: deja el frame al hilo ReID (solo el ultimo) y sigue."""
        now = time.perf_counter()
        if now - self._reid_last_wall < self._reid_period:
            return
        if not any(str(b[4]).lower() == "person" for b in boxes):
            return
        self._reid_last_wall = now
        with self._reid_job_lock:
            self._reid_job = (cv_image, list(boxes))
        self._reid_event.set()

    def _reid_loop(self) -> None:
        while True:
            self._reid_event.wait()
            with self._reid_job_lock:
                job, self._reid_job = self._reid_job, None
                self._reid_event.clear()
            if job is not None:
                self._embed_persons(*job)

    def _embed_persons(self, cv_image: np.ndarray, boxes: List[_RawYoloDet]) -> None:
        person = [tuple(b[:4]) for b in boxes if str(b[4]).lower() == "person"]
        if not person:
            return
        # Cajas solapadas mezclan dos personas en el crop: no generan embedding.
        clean = [
            bb for i, bb in enumerate(person)
            if all(
                j == i or _bbox_iou(bb, other) < self._reid_overlap_iou
                for j, other in enumerate(person)
            )
        ]
        if not clean:
            return
        try:
            t0 = time.perf_counter()
            embs = self._reid.embed(cv_image, clean)
            if self._stage_timer is not None:
                self._stage_timer.record("reid_embed", (time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            self._logger.error(f"ReID embed: {exc}")
            return
        for i, emb in embs.items():
            self._reid_cache.put(clean[i], emb)

    def _loop(self) -> None:
        while True:
            with self._pending_lock:
                job = self._pending_queue.popleft() if self._pending_queue else None
            if job is None:
                time.sleep(0.003)
                continue
            cv_image, stamp_key, frame_id = job
            try:
                t0 = time.perf_counter()
                boxes = self._backend.infer(cv_image)
                ms = (time.perf_counter() - t0) * 1000.0
                if self._stage_timer is not None:
                    self._stage_timer.record("yolo_infer", ms)
                if self._reid is not None:
                    self._submit_reid(cv_image, boxes)
                self._store(stamp_key, boxes, ms)
                sk_sec = float(stamp_key[0]) + float(stamp_key[1]) * 1e-9
                with self._result_lock:
                    self._latest_boxes = boxes
                    self._latest_stamp_sec = sk_sec
                    self._latest_yolo_ms = ms
                if self._on_annotated is not None:
                    self._on_annotated(cv_image, stamp_key, list(boxes), frame_id)
                else:
                    with self._annotated_lock:
                        self._annotated_queue.append(
                            (cv_image, stamp_key, list(boxes), frame_id)
                        )
                        while len(self._annotated_queue) > 3:
                            self._annotated_queue.popleft()
                if not self._logged_first_infer:
                    self._logged_first_infer = True
                    self._logger.info(
                        f"YOLO primera inferencia: {len(boxes)} det ({ms:.0f}ms)"
                    )
            except Exception as exc:
                self._logger.error(f"YOLO bg worker: {exc}")


def _bbox_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


# ─── Parameter declaration helper ───────────────────────────────────

def _p(node, name: str, default, *, cast=None):
    """Declare + get a ROS parameter in one call."""
    node.declare_parameter(name, default)
    val = node.get_parameter(name).value
    if cast is not None:
        return cast(val)
    return val


# ─── Main node ──────────────────────────────────────────────────────


class DynamicTrackingNode(Node):
    def __init__(self) -> None:
        super().__init__("dynamic_tracking_node")
        p = lambda name, default, cast=None: _p(self, name, default, cast=cast)

        # ── Topics ──
        image_topic       = p("image_topic", "/image_raw")
        detections_topic  = p("detections_topic", "/detections")
        annotated_topic   = p("annotated_image_topic", "/image_annotated")
        model_name        = p("model_name", "yolo26n.pt")
        self.yolo_backend  = str(p("yolo_backend", "ultralytics")).strip().lower()
        self.yolo_onnx_model = str(p("yolo_onnx_model", "yolo26n_256.onnx"))
        onnx_providers_raw = p("yolo_onnx_providers", ["CPUExecutionProvider"])
        self.yolo_onnx_providers = (
            list(onnx_providers_raw) if isinstance(onnx_providers_raw, (list, tuple))
            else [str(onnx_providers_raw)]
        )
        self.yolo_onnx_intra_threads = max(0, p("yolo_onnx_intra_op_threads", 2, int))
        self.yolo_onnx_inter_threads = max(0, p("yolo_onnx_inter_op_threads", 1, int))
        lidar_topic_str   = p("lidar_topic", "/scan")

        # ── YOLO / device ──
        self.enable_yolo    = p("enable_yolo_detections", False, bool)
        self.device, self._yolo_device_reason = _resolve_yolo_device(p("device", "auto"))
        self.conf_threshold = p("confidence_threshold", 0.35, float)
        target_cls_raw      = p("target_classes", ["person", "bicycle", "car", "motorcycle", "bus", "truck"])
        self.target_classes  = {str(c).lower() for c in target_cls_raw}
        self.yolo_imgsz      = p("yolo_imgsz", 384, int)
        yolo_half_param      = p("yolo_half", True, bool)
        self.yolo_max_det    = p("yolo_max_det", 20, int)
        self._yolo_stride    = max(1, p("yolo_frame_stride", 1, int))
        self._yolo_frame_ctr = 0
        self._yolo_keep_latest = p("yolo_keep_latest_frame_only", True, bool)
        dev_lower = str(self.device).lower()
        self._yolo_half = yolo_half_param and ("cuda" in dev_lower or "mps" in dev_lower)

        # ── Re-ID por apariencia (OSNet): recupera el ID original al volver a ver a la persona ──
        self.enable_reid       = p("enable_reid", False, bool) and self.enable_yolo
        self.reid_model        = str(p("reid_model", "osnet_x0_25"))
        self.reid_weights      = str(p("reid_weights", "osnet_x0_25_msmt17.pth"))
        self.reid_rate_hz      = p("reid_rate_hz", 2.0, float)
        # mientras algun track decide si es alguien de la galeria: mas rapido (veredicto antes)
        self.reid_rate_pending_hz = p("reid_rate_pending_hz", 4.0, float)
        self.reid_half         = p("reid_half", False, bool)  # fp16 en CUDA (independiente de yolo_half)
        self.reid_skip_overlap = p("reid_skip_overlap_iou", 0.3, float)
        self.reid_gallery_file = os.path.expanduser(str(p("reid_gallery_file", "")))
        reid_topic_ns          = str(p("reid_topic_namespace", "/dynamic_tracking/reid")).rstrip("/")
        self._reid_cfg = ReIdConfig(
            match_threshold=p("reid_match_threshold", 0.75, float),
            match_margin=p("reid_match_margin", 0.05, float),
            duplicate_threshold=p("reid_duplicate_threshold", 0.8, float),
            confirm_hits=max(1, p("reid_confirm_hits", 2, int)),
            strong_threshold=p("reid_strong_threshold", 0.85, float),
            max_samples_to_match=max(1, p("reid_max_samples_to_match", 20, int)),
            lost_ttl_sec=p("reid_lost_ttl_sec", 300.0, float),
        )

        # ── Sensor QoS ──
        img_qos_str  = p("image_qos_reliability", "best_effort")
        ann_qos_str  = p("annotated_image_qos_reliability", "reliable")
        lidar_qos_str = p("lidar_qos_reliability", "best_effort")
        qos_depth     = p("sensor_qos_depth", 10, int)
        lidar_qos_depth = p("lidar_qos_depth", 1, int)

        # ── Tracking core ──
        self.tracking_frame  = p("tracking_frame", "map")
        self.min_dist_m      = p("min_track_distance_m", 0.25, float)
        self.max_dist_m      = p("max_track_distance_m", 8.0, float)
        self.enable_lidar    = p("enable_lidar_fusion", False, bool)
        self.sync_img_lidar  = p("sync_image_with_lidar", True, bool)
        self._lidar_track_on_scan = p(
            "lidar_track_on_scan",
            bool(self.enable_lidar and not self.sync_img_lidar),
            bool,
        )
        self.sync_slop       = p("sensor_sync_slop_sec", 0.12, float)
        self.sync_queue      = p("sensor_sync_queue_size", 16, int)

        # ── LiDAR ──
        self.fov_rad          = math.radians(p("camera_horizontal_fov_deg", 69.4, float))
        self.lidar_window_rad = math.radians(p("lidar_angular_window_deg", 20.0, float))
        self.angle_offset_rad = math.radians(p("lidar_fusion_angle_offset_deg", 0.0, float))
        self.use_scan_stamp_tf = p("lidar_fusion_use_scan_stamp_for_tf", True, bool)
        self.use_tf_lidar_ray  = p("use_tf_lidar_ray", True, bool)
        self.use_stamped_tf_for_cluster = p("lidar_cluster_use_stamped_tf", False, bool)
        # Si la TF del scan aun no ha llegado (odometria ~65 ms tarde), extrapolar la ultima
        # con la velocidad de la propia TF hasta este maximo (s). 0 = usar la ultima sin corregir.
        self.scan_tf_extrap_max = p("scan_tf_extrapolate_max_sec", 0.0, float)
        # Une las dos piernas de una persona en una medicion (punto medio) si estan a <= este
        # valor (m). Quita la alternancia izq/der al caminar. 0 = cada pierna es una medicion.
        self.leg_pair_merge_m = p("leg_pair_merge_max_sep_m", 0.0, float)
        self.ray_bbox_bottom   = p("lidar_ray_use_bbox_bottom", True, bool)
        self.cluster_jump_m    = p("lidar_cluster_jump_m", 0.30, float)
        self.cluster_pose_mode = p("lidar_cluster_pose_mode", "median")
        self.min_pts_seg       = p("lidar_min_points_per_segment", 3, int)
        self.max_pts_seg       = p("lidar_max_points_per_segment", 50, int)
        self.interleave_half_win = p("lidar_interleave_repair_half_window", 0, int)
        self.interleave_max_r  = p("lidar_interleave_repair_max_range_m", 1.5, float)
        self.min_seg_width_m   = p("lidar_min_segment_width_m", 0.0, float)
        self.blind_zone_r      = p("lidar_blind_zone_radius_m", 0.0, float)

        # ── Leg heuristic ──
        self.leg_params = LegHeuristicParams(
            enabled=p("leg_candidate_enable", True, bool),
            min_span_deg=p("leg_min_span_deg", 2.5, float),
            max_span_deg=p("leg_max_span_deg", 26.0, float),
            max_range_spread_ratio=p("leg_max_range_spread_ratio", 0.16, float),
            min_chord_m=p("leg_min_chord_m", 0.09, float),
            max_chord_m=p("leg_max_chord_m", 0.55, float),
        )
        self.leg_hyp_min_score = p("leg_hypothesis_min_score", 0.62, float)
        _leg_det = p("leg_heuristic_detector_min_score", 0.0, float)
        self.leg_heuristic_det_min_score = (
            _leg_det if _leg_det > 0.0 else self.leg_hyp_min_score
        )
        _spawn_score = p("lidar_spawn_person_min_leg_score", 0.0, float)
        self.spawn_min_leg_score = _spawn_score if _spawn_score > 0 else self.leg_hyp_min_score

        # ── Marker visualization ──
        # Suavizado solo visual de la caja (EMA con constante de tiempo; 0 = sin suavizar)
        self.marker_smooth_tau = p("marker_position_smoothing_tau_sec", 0.0, float)
        self.marker_smooth_snap_m = p("marker_position_smoothing_snap_m", 1.0, float)
        self._last_display_sec = 0.0
        marker_topic         = p("marker_topic", "/dynamic_tracks_markers")
        self.marker_topic     = str(marker_topic).strip() or "/dynamic_tracks_markers"
        self.viz_frame       = p("marker_visualization_frame", "").strip()
        self.remap_latest    = p("marker_remap_prefer_latest_tf", False, bool)
        self.remap_img_fb    = p("marker_remap_try_image_stamp_fallback", True, bool)
        self.remap_latest_fb = p("marker_remap_allow_latest_fallback", False, bool)
        self.remap_timeout   = p("marker_remap_tf_timeout_sec", 0.0, float)
        self.remap_stale_sec = p("marker_remap_stale_tf_cache_sec", 0.75, float)
        self.marker_lifetime = p("marker_lifetime_sec", 0.0, float)

        # Marker appearance
        self.pub_unknown       = p("publish_unknown_markers", True, bool)
        self.unk_min_speed     = p("unknown_min_speed_mps", 0.03, float)
        self.lidar_only_motion = p("lidar_only_markers_require_motion", True, bool)
        self.draw_lidar_only_img = p("draw_lidar_only_on_image", False, bool)
        self.bbox_alpha        = p("marker_bbox_alpha", 0.82, float)
        self.bbox_wire         = p("marker_bbox_wireframe", False, bool)
        self.bbox_line_w       = p("marker_bbox_line_width_m", 0.022, float)
        self.bbox_vel_heading  = p("marker_bbox_use_velocity_heading", True, bool)
        self.bbox_heading_min  = p("marker_bbox_heading_min_speed_mps", 0.06, float)
        self.text_scale_z      = p("marker_text_scale_z", 0.28, float)
        self.compact_labels    = p("marker_compact_map_labels", True, bool)
        self.text_height_m     = p("marker_text_height_m", 1.55, float)
        self.text_track_color  = p("marker_text_use_track_color", False, bool)
        self.text_r            = p("marker_text_color_r", 0.06, float)
        self.text_g            = p("marker_text_color_g", 0.06, float)
        self.text_b            = p("marker_text_color_b", 0.10, float)
        self.label_ring_r      = p("marker_label_ring_radius_m", 0.35, float)
        self.label_z_stagger   = p("marker_label_z_stagger_m", 0.06, float)
        self.uncertain_static  = p("marker_uncertain_static_lidar_person", True, bool)
        self.hide_uncertain    = p("marker_hide_uncertain_static_lidar_person", False, bool)
        self.hide_missed_person = p("marker_hide_person_if_missed_frames", 0, int)
        self.stale_rgb_sec     = p("person_lidar_uncertain_stale_rgb_sec", 4.0, float)

        # Marker: person boxes
        self.person_box_xy     = p("person_box_size_xy_m", 0.6, float)
        _merge_box = p("person_track_map_merge_box_m", 0.0, float)
        self.merge_box_m       = _merge_box if _merge_box > 1e-6 else self.person_box_xy
        self.person_box_h      = p("person_box_height_m", 1.7, float)
        self.unk_box_xy        = p("unknown_box_size_xy_m", 0.4, float)
        self.unk_box_h         = p("unknown_box_height_m", 1.2, float)

        # Marker: trajectory / history
        self.pub_traj          = p("publish_track_trajectory_markers", True, bool)
        self.traj_z            = p("track_trajectory_z_m", 0.06, float)
        self.traj_line_w       = p("track_trajectory_line_width_m", 0.035, float)
        self.traj_alpha        = p("track_trajectory_alpha", 0.65, float)
        self.pub_cones         = p("publish_track_history_cones", False, bool)
        self.hist_max          = p("track_history_max_markers", 28, int)
        self.hist_h_min        = p("track_history_cylinder_height_min_m", 0.035, float)
        self.hist_h_max        = p("track_history_cylinder_height_max_m", 0.14, float)
        self.hist_d_min        = p("track_history_cylinder_diameter_min_m", 0.05, float)
        self.hist_d_max        = p("track_history_cylinder_diameter_max_m", 0.12, float)
        self.hist_a_min        = p("track_history_alpha_min", 0.12, float)
        self.hist_a_max        = p("track_history_alpha_max", 0.52, float)
        self.hist_z            = p("track_history_z_m", 0.02, float)

        # ── Tracking params ──
        self.assoc_cost_cam    = p("tracking_assoc_cost_threshold", 0.7, float)
        self.assoc_cost_lidar  = p("tracking_lidar_assoc_cost_threshold", 1.6, float)
        self.assoc_person_lid  = p("tracking_person_lidar_assoc_cost_threshold", 0.0, float)
        self.max_missed        = p("tracking_max_missed_frames", 15, int)
        self.person_max_missed = p("tracking_person_max_missed_frames", 45, int)
        self.min_confirmed     = p("tracking_min_confirmed_hits", 2, int)
        self.spawn_unknown     = p("spawn_unknown_from_lidar", False, bool)
        self.leg_as_person     = p("leg_hypothesis_as_person", False, bool)
        self.img_only_fallback = p("enable_image_only_assoc_fallback", True, bool)
        self.img_only_iou_min  = p("image_only_assoc_min_iou", 0.22, float)
        self.kf_dt_cap         = p("kalman_predict_dt_cap_sec", 0.15, float)
        self.innov_gate_m      = p("person_lidar_innovation_gate_m", 0.0, float)
        self.innov_gate_rgb_s  = p("person_lidar_innovation_gate_stale_rgb_sec", 3.5, float)
        self.lidar_max_jump    = p("person_lidar_max_correction_jump_m", 0.55, float)
        self.static_zero_v     = p("person_static_zero_velocity_speed_m", 0.08, float)
        self.hold_skip_sec     = p("person_hold_skip_lidar_association_sec", 0.0, float)
        self.verify_range_tol  = p("scan_verify_range_tolerance_m", 0.65, float)
        self.verify_window_deg = p("scan_verify_window_deg", 22.0, float)
        self.missed_debounce   = p("person_missed_debounce_frames", 2, int)
        self.motion_speed_thr  = p("motion_speed_threshold_mps", 0.05, float)
        self.motion_accel_thr  = p("motion_accel_threshold_mps2", 0.03, float)
        self.tf_timeout        = p("tf_lookup_timeout_sec", 0.05, float)

        # LiDAR dual-tier (strict spawn + lax maintenance)
        self.lidar_dual_tier   = p("lidar_dual_tier_enable", False, bool)
        self.spawn_fusion_pol  = p("lidar_spawn_fusion_policy", "require_two")
        if p("lidar_spawn_require_all_detectors", False, bool):
            self.spawn_fusion_pol = "unanimous"
        self.maint_fusion_pol  = p("lidar_maintenance_fusion_policy", "max_score")
        self.maint_min_leg     = p("lidar_maintenance_min_leg_score", 0.0, float)
        self.spawn_init_step   = p("spawn_initiator_max_step_displacement_m", 0.85, float)
        self.spawn_init_vel    = p("spawn_initiator_max_velocity_mps", 2.2, float)
        self.track_dual_audit  = p("person_track_dual_audit_enable", True, bool)
        self.track_audit_iv    = p("person_track_dual_audit_interval_frames", 8, int)
        self.track_audit_r     = p("person_track_dual_audit_radius_m", 0.42, float)
        self.leg_snap_on       = p("person_track_leg_snap_enable", False, bool)
        self.leg_snap_max      = p("person_track_leg_snap_max_dist_m", 1.45, float)
        self.leg_snap_min      = p("person_track_leg_snap_min_leg_score", 0.65, float)
        self.lidar_micro_stable = p("person_lidar_micro_stable_m", 0.0, float)
        self.snap_vel_jump_min = p("person_track_leg_snap_min_jump_for_velocity_m", 0.45, float)
        self.snap_vel_cap      = p("person_track_leg_snap_max_velocity_mps", 1.0, float)
        self.snap_spawn_jump   = p("person_track_leg_snap_require_spawn_tier_jump_m", 0.55, float)
        self.verify_need_cluster = p("scan_verify_require_nearby_cluster", True, bool)
        self.verify_near_r     = p("scan_verify_nearby_cluster_m", 0.55, float)
        self.verify_near_leg   = p("scan_verify_nearby_min_leg_score", 0.62, float)

        # LiDAR spawning
        self.spawn_dedup_m     = p("lidar_spawn_same_frame_dedup_m", 0.0, float)
        self.spawn_confirm_n   = p("lidar_spawn_confirm_frames", 0, int)
        self.spawn_confirm_r   = p("lidar_spawn_confirm_radius_m", 0.42, float)
        self.spawn_suppress_r  = p("person_lidar_spawn_suppress_radius_m", 0.0, float)
        self.spawn_suppress_s  = p("person_lidar_spawn_suppress_rgb_sec", 0.0, float)

        # Track merging / duplicates
        self.merge_iou         = p("person_track_map_merge_min_iou", 0.42, float)
        self.merge_lidar_only  = p("person_track_map_merge_lidar_only", False, bool)
        self.dup_enable        = p("duplicate_track_temporal_enable", True, bool)
        self.dup_samples       = p("duplicate_track_compare_samples", 10, int)
        self.dup_must_match    = p("duplicate_track_must_match", 6, int)
        self.dup_max_dist      = p("duplicate_track_max_dist_m", 0.18, float)
        self.dup_lidar_only    = p("duplicate_track_lidar_only", False, bool)

        # Person dynamics
        self.dyn_window        = p("person_dynamic_window_sec", 2.0, float)
        self.dyn_min_disp      = p("person_dynamic_min_displacement_m", 0.32, float)
        self.dyn_hist_sec      = p("person_dynamic_pose_history_sec", 4.0, float)
        _pdc = p("person_dynamic_classes", ["person"])
        self.dyn_classes = tuple(str(x).lower() for x in _pdc) if _pdc else ("person",)
        self.dyn_suffix        = p("dynamic_person_suffix_in_class_id", False, bool)

        # Spencer-style
        self.lidar_only_missed = p("person_lidar_only_max_missed_frames", 88, int)
        self.lidar_static_sec  = p("lidar_initiation_max_age_static_sec", 0.0, float)
        self.lidar_static_min_travel = p("lidar_initiation_static_min_travel_m", 0.0, float)
        self.max_occ           = p("max_occlusions_before_deletion", 120, int)
        self.max_occ_mature    = p("max_occlusions_mature_track", 0, int)
        self.use_mahalanobis   = p("use_mahalanobis_lidar", True, bool)
        self.require_vis_conf  = p("require_visual_confirmation", True, bool)

        # ── YOLO auxiliary confirmation ──
        self.rgb_coast_sec     = p("person_rgb_coast_cl_sec", 5.0, float)
        self.yolo_iou_min      = p("person_yolo_assoc_iou_min", 0.12, float)
        self.yolo_green_sec    = p("person_yolo_seen_green_sec", 0.45, float)
        self.aux_max_dist      = p("yolo_auxiliary_confirm_max_dist_m", 1.05, float)
        self.aux_min_hits      = p("yolo_auxiliary_min_track_hits", 0, int)
        self.aux_dist_lidar    = p("yolo_auxiliary_distance_match_lidar_ray_only", True, bool)
        self.aux_proj_tracks   = p("yolo_auxiliary_project_tracks_to_image", True, bool)
        self.aux_proj_h        = p("yolo_auxiliary_projection_height_m", 1.0, float)
        self.aux_proj_margin   = p("yolo_auxiliary_projection_bbox_margin_px", 20.0, float)
        self.aux_tf_extra      = p("yolo_auxiliary_projection_tf_extra_timeout_sec", 0.0, float)
        self.aux_bearing_on    = p("yolo_auxiliary_bearing_match_enable", True, bool)
        self.aux_bearing_max   = p("yolo_auxiliary_bearing_match_max_rad", 0.35, float)
        self.aux_polar_on      = p("yolo_auxiliary_polar_match_enable", True, bool)
        self.aux_polar_range   = p("yolo_auxiliary_polar_match_max_range_m", 1.05, float)
        self.aux_polar_bear    = p("yolo_auxiliary_polar_match_max_bearing_rad", 0.45, float)
        self.yolo_visual_par   = p("yolo_visual_lidar_parity", True, bool)
        self.yolo_reject_fov   = p("yolo_auxiliary_reject_in_fov_enable", False, bool)
        self.yolo_reject_frames = p("yolo_auxiliary_reject_in_fov_frames", 8, int)
        self.yolo_reject_prune = p("yolo_auxiliary_reject_prune_missed_frames", 9999, int)
        self.marker_coast_bgr = self._parse_bgr_param(
            p("marker_color_coast_bgr", [180, 200, 40]), (180, 200, 40)
        )

        # YOLO sync
        self.yolo_wait_sec     = p("yolo_sync_wait_max_sec", 0.0, float)
        self.yolo_skew_sec     = p("yolo_assoc_max_stamp_skew_sec", 0.02, float)
        self.yolo_fb_latest    = p("yolo_assoc_fallback_latest", False, bool)
        self.yolo_age_sec      = p("yolo_latest_max_age_sec", 0.15, float)
        self.yolo_annotated_pair = p("yolo_annotated_pair_with_infer", True, bool)
        self.overlay_age_sec   = p("image_track_overlay_max_age_sec", 0.0, float)

        # YOLO debug
        self.dbg_proj          = p("yolo_debug_draw_track_projections", False, bool)
        self.dbg_polar         = p("yolo_debug_laser_polar_markers", False, bool)
        _polar_topic           = p("yolo_debug_laser_polar_topic", "/dynamic_tracking/yolo_laser_polar_markers")
        self.dbg_polar_topic   = str(_polar_topic).strip() or "/dynamic_tracking/yolo_laser_polar_markers"
        self.dbg_polar_line_w  = p("yolo_debug_laser_polar_line_width_m", 0.022, float)

        # ── Image publishing ──
        self.pub_annotated     = p("publish_annotated_image", True, bool)
        self.pub_compressed    = p("publish_compressed_image", False, bool)
        self.jpeg_quality      = p("jpeg_quality", 70, int)
        self.pub_clusters      = p("publish_lidar_cluster_markers", False, bool)
        self.cluster_diam      = p("lidar_cluster_marker_diameter_m", 0.14, float)
        self.pub_dot_yolo_crops = p("publish_dot_yolo_crops", False, bool)
        # Legacy param name (same effect)
        if not self.pub_dot_yolo_crops:
            self.pub_dot_yolo_crops = p("publish_person_crops", False, bool)
        _crop_topic = str(p("dot_yolo_crops_topic", "")).strip()
        if not _crop_topic:
            _crop_topic = str(p("person_crops_topic", "/dot_yolo_crops")).strip()
        self.dot_yolo_crops_topic = _crop_topic or "/dot_yolo_crops"
        self.dot_yolo_crop_margin = float(p("dot_yolo_crop_margin_px", 8.0, float))

        # ── Camera info ──
        self.cam_info_topic    = p("camera_info_topic", "").strip()

        # ── Costmap / static-map filters ──
        map_filter_on          = p("map_filter_enable", False, bool)
        self.costmap_topic     = p("costmap_topic", "/local_costmap/costmap").strip()
        self._costmap_filter: Optional[CostmapFilter] = None
        if map_filter_on:
            self._costmap_filter = CostmapFilter(
                occupancy_threshold=p("occupancy_threshold", 253, int),
                inflation_threshold=p("map_filter_inflation_threshold", 0, int),
                unknown_as_free=p("map_filter_unknown_as_free", True, bool),
                neighbor_radius_cells=p("map_filter_neighbor_radius_cells", 0, int),
            )
        else:
            p("occupancy_threshold", 253, int)
            p("map_filter_inflation_threshold", 0, int)
            p("map_filter_unknown_as_free", True, bool)
            p("map_filter_neighbor_radius_cells", 0, int)

        static_map_on          = p("static_map_filter_enable", True, bool)
        self.static_map_topic  = p("static_map_topic", "/map").strip()
        self._static_filter: Optional[StaticMapFilter] = None
        if static_map_on:
            self._static_filter = StaticMapFilter(
                occupied_threshold=p("static_map_occupied_threshold", 65, int),
                neighbor_radius_cells=p("static_map_neighbor_radius_cells", 1, int),
                keep_if_tf_fails=p("static_map_keep_cluster_if_tf_fails", False, bool),
                corner_probe_enable=p("static_map_corner_probe_enable", True, bool),
                probe_radius_m=p("static_map_probe_radius_m", 0.11, float),
                corner_probe_radius_m=p("static_map_corner_probe_radius_m", 0.0, float),
                disk_filter_enable=p("static_map_disk_filter_enable", True, bool),
                disk_radius_m=p("static_map_disk_radius_m", 0.15, float),
                disk_min_free_ratio=p("static_map_disk_min_free_ratio", 0.88, float),
                disk_permit_outside_map=p("static_map_disk_permit_outside_map", True, bool),
            )
        else:
            for _sn, _sd, _sc in [
                ("static_map_occupied_threshold", 65, int), ("static_map_neighbor_radius_cells", 1, int),
                ("static_map_keep_cluster_if_tf_fails", False, bool), ("static_map_corner_probe_enable", True, bool),
                ("static_map_probe_radius_m", 0.11, float), ("static_map_corner_probe_radius_m", 0.0, float),
                ("static_map_disk_filter_enable", True, bool), ("static_map_disk_radius_m", 0.15, float),
                ("static_map_disk_min_free_ratio", 0.88, float), ("static_map_disk_permit_outside_map", True, bool),
            ]:
                p(_sn, _sd, _sc)

        self.reject_fusion_wall = p("reject_lidar_rgb_fusion_on_static_map", True, bool)
        self._log_verbose_startup = p("log_verbose_startup", False, bool)

        # Area filter
        area_on = p("area_filter_enable", False, bool)
        area_frame = p("area_filter_frame", "map")
        _zones_file = str(p("area_filter_zones_file", "")).strip()
        _zones_json = p("area_filter_zones_json", "[]")
        self.area_filter_pub_markers = p("area_filter_publish_markers", True, bool)
        self.area_marker_topic = str(
            p("detection_area_marker_topic", "/detection_area_markers")
        ).strip() or "/detection_area_markers"
        self._area_filter: Optional[AreaFilter] = None
        self._area_filter_zones: List[dict] = []
        self._area_filter_frame = "map"
        if area_on:
            import json
            zone_list: List[dict] = []
            try:
                if _zones_file:
                    zones_path = resolve_zones_file(_zones_file)
                    file_frame, zone_list = load_zones_file(zones_path)
                    if file_frame:
                        area_frame = file_frame
                    self.get_logger().info(
                        f"Area filter: {len(zone_list)} zone(s) from {zones_path}"
                    )
                json_str = str(_zones_json).strip()
                if not zone_list and json_str and json_str not in ("[]", '""'):
                    zone_list = [parse_zone(z) for z in json.loads(json_str)]
                if not zone_list:
                    xmin = p("area_filter_x_min", 2.0, float)
                    xmax = p("area_filter_x_max", 9.0, float)
                    ymin = p("area_filter_y_min", -5.0, float)
                    ymax = p("area_filter_y_max", 2.0, float)
                    if xmax > xmin and ymax > ymin:
                        zone_list = [{
                            "x_min": xmin, "x_max": xmax,
                            "y_min": ymin, "y_max": ymax,
                            "add": True,
                        }]
                        if self._log_verbose_startup:
                            self.get_logger().info(
                                f"Area filter: using rectangle params "
                                f"x=[{xmin},{xmax}] y=[{ymin},{ymax}]"
                            )
                if not zone_list:
                    self.get_logger().error(
                        "area_filter_enable=true but no zones "
                        "(set area_filter_zones_file, area_filter_zones_json "
                        "or area_filter_x_min/x_max/y_min/y_max)"
                    )
                else:
                    self._area_filter_zones = zone_list
                    self._area_filter_frame = str(area_frame).strip()
                    self._area_filter = AreaFilter(
                        zone_list, self._area_filter_frame,
                        track_margin_m=p("area_filter_track_margin_m", 0.0, float),
                    )
                    if self._log_verbose_startup:
                        for i, z in enumerate(zone_list):
                            shape = (
                                f"polygon {len(z['polygon'])} pts, bbox "
                                if "polygon" in z else ""
                            )
                            self.get_logger().info(
                                f"Area filter zone #{i}: frame={area_frame} {shape}"
                                f"x=[{z['x_min']},{z['x_max']}] y=[{z['y_min']},{z['y_max']}] "
                                f"add={z['add']}"
                            )
                        if self.area_filter_pub_markers:
                            self.get_logger().info(
                                f"Detection ROI RViz: topic {self.area_marker_topic} "
                                f"(MarkerArray, frame={self._area_filter_frame})"
                            )
            except (Exception) as e:
                self.get_logger().error(f"area_filter config error: {e}")

        # Depth
        self.depth_topic = p("depth_image_topic", "").strip()
        self.use_depth_fb = p("use_depth_for_camera_fallback", True, bool)
        self.depth_tol = p("depth_time_tolerance_sec", 0.35, float)

        # Backward-compat: declared so old YAML files don't cause "undeclared param" errors
        p("enable_camera_map_fallback", True)
        p("camera_only_depth_m", 3.0)
        p("yolo_auxiliary_use_lidar_ray_match", True)
        p("yolo_republish_markers_after_aux", False)

        # Diagnostics
        self._pipe_timing = PipelineTiming(p("log_pipeline_timing_sec", 0.0, float))
        self._stage_timer = StageTimer(self._pipe_timing.interval)
        self._lidar_diag = LidarDiagnostics(p("log_lidar_diag_sec", 0.0, float))
        self._person_log = PersonTrackLogger(p("log_person_tracks_sec", 2.0, float))
        self._extract_diag = SpencerExtractDiagnostics(
            p("log_spencer_extract_diag_sec", 0.0, float)
        )
        self._debug_tf_projection = p("debug_tf_projection", False, bool)
        self._debug_tf_log_interval = p("debug_tf_log_interval_sec", 1.0, float)
        self._debug_tf_last_log = 0.0
        _pose_log_iv = 0.0
        if p("debug_pose_log_enable", False, bool):
            _pose_log_iv = p("debug_pose_log_interval_sec", 0.1, float)
            if _pose_log_iv <= 1e-6:
                _pose_log_iv = 0.1
        self._pose_debug = PoseDebugLogger(
            _pose_log_iv,
            robot_frame=p("debug_pose_robot_frame", "base_link"),
        )
        self._scan_lag_sec = 0.0
        # Multi-detector LiDAR pipeline (Spencer-style, single node)
        _rf_path_raw = p("spencer_rf_model_path", "").strip()
        self._rf_threshold = p("spencer_rf_decision_threshold", 0.5, float)
        self._rf_only_pos = p("spencer_rf_only_positive_classifications", True, bool)
        self._rf_post_min = p("spencer_rf_postfilter_min_confidence", 0.0, float)
        _ada_path = p("spencer_adaboost_model_path", "").strip()
        self._ada_threshold = p("spencer_adaboost_decision_threshold", -10.0, float)
        _det_enabled = p(
            "lidar_detectors_enabled",
            ["spencer_rf", "leg_heuristic"],
        )
        if isinstance(_det_enabled, str):
            _det_enabled = [s.strip() for s in _det_enabled.split(",") if s.strip()]
        self._lidar_pipeline = build_lidar_pipeline(
            enabled=list(_det_enabled),
            fusion_policy=p("lidar_fusion_policy", "max_score"),
            fusion_any_high_threshold=p("lidar_fusion_any_high_threshold", 0.70, float),
            only_positive_rf=bool(self._rf_only_pos),
            dual_tier_enable=self.lidar_dual_tier,
            spawn_fusion_policy=self.spawn_fusion_pol,
            maintenance_fusion_policy=self.maint_fusion_pol,
            maintenance_min_leg_score=self.maint_min_leg,
            cluster_jump_m=self.cluster_jump_m,
            min_track_distance_m=self.min_dist_m,
            max_track_distance_m=self.max_dist_m,
            blind_zone_m=self.blind_zone_r,
            pose_mode=self.cluster_pose_mode,
            min_points=self.min_pts_seg,
            max_points=self.max_pts_seg,
            min_segment_width_m=self.min_seg_width_m,
            leg_params=self.leg_params,
            leg_heuristic_score_threshold=self.leg_heuristic_det_min_score,
            rf_model_path=_rf_path_raw,
            rf_threshold=self._rf_threshold,
            adaboost_model_path=_ada_path,
            adaboost_threshold=self._ada_threshold,
            blob_params=BlobDetectorParams(
                min_points=p("blob_min_points", 4.0, float),
                max_points=p("blob_max_points", 25.0, float),
                min_width_m=p("blob_min_width_m", 0.05, float),
                max_width_m=p("blob_max_width_m", 0.55, float),
                max_range_m=p("blob_max_range_m", 20.0, float),
            ),
            cascade_min_leg_score=p("lidar_cascade_min_leg_score", 0.0, float),
        )

        # ─── Runtime state ───
        self.bridge = CvBridge()
        self.latest_scan: Optional[LaserScan] = None
        self.latest_lidar_clusters: List[LidarMeasurement] = []
        self.latest_lidar_spawn_clusters: List[LidarMeasurement] = []
        self.latest_scan_frame: str = "base_link"
        self.tracks: List[Track] = []
        self._pending_spawns: List[PendingLidarSpawn] = []
        self.next_track_id = 1
        self._last_kf_sec: Optional[float] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self._last_optical_frame: str = ""
        self._costmap: Optional[OccupancyGrid] = None
        self._static_map: Optional[OccupancyGrid] = None
        self.latest_depth_msg: Optional[Image] = None
        self._last_frame_wall: Optional[float] = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self._tf = TFHelper(self.tf_buffer, self.tf_timeout)
        self._remap_tf = MarkerRemapTF()
        self._warned_remap = [False]
        
        self._scan_tf_captured: Optional[Tuple[float, float, float]] = None
        self._scan_tf_mode: str = "none"
        self._lidar_lock = threading.Lock()
        self._tracks_lock = threading.Lock()
        self._scan_cb_group = MutuallyExclusiveCallbackGroup()

        # YOLO
        self.model = None
        self.class_map: Dict[int, str] = {}
        self._yolo_backend_label = "n/a"
        self._yolo: Optional[_YoloBackgroundWorker] = None
        self._reid_embedder = None
        self._reid_cache = None
        self._identity: Optional[IdentityManager] = None
        if self.enable_reid:
            self._init_reid()
        if self.enable_yolo:
            self._init_yolo(model_name)

        # ─── Subscriptions & publishers ───
        img_qos = self._sensor_qos(img_qos_str, qos_depth)
        ann_qos = self._sensor_qos(ann_qos_str, qos_depth)
        lidar_qos = self._sensor_qos(lidar_qos_str, lidar_qos_depth)
        lidar_sync_qos = self._sensor_qos(lidar_qos_str, max(lidar_qos_depth, self.sync_queue))

        if self.enable_lidar and self.sync_img_lidar:
            self.create_subscription(
                LaserScan, lidar_topic_str, self._scan_cb, lidar_qos,
                callback_group=self._scan_cb_group,
            )
            img_sub = message_filters.Subscriber(self, Image, image_topic, qos_profile=img_qos)
            scan_sub = message_filters.Subscriber(self, LaserScan, lidar_topic_str, qos_profile=lidar_sync_qos)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [img_sub, scan_sub], queue_size=max(2, self.sync_queue), slop=self.sync_slop,
            )
            self._sync.registerCallback(self._sync_cb)
        else:
            self.create_subscription(Image, image_topic, self._image_cb, img_qos)
            if self.enable_lidar:
                self.create_subscription(
                    LaserScan, lidar_topic_str, self._scan_cb, lidar_qos,
                    callback_group=self._scan_cb_group,
                )

        if self.cam_info_topic:
            self.create_subscription(CameraInfo, self.cam_info_topic, self._caminfo_cb, img_qos)

        self.det_pub = self.create_publisher(Detection2DArray, detections_topic, 10)
        self.track_pub = self.create_publisher(Detection2DArray, detections_topic + "/tracked", 10)
        # Subset de /tracked con bbox de imagen real y fresca (dentro del FOV
        # de la camara) -- para consumidores como vlm_hri que solo pueden
        # razonar sobre lo que hay en imagen, sin los tracks solo-LiDAR.
        self.track_fov_pub = self.create_publisher(Detection2DArray, detections_topic + "/tracked_fov", 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        if self._identity is not None:
            self.reid_pub = self.create_publisher(String, reid_topic_ns + "/identities", 10)
            self.create_subscription(String, reid_topic_ns + "/enroll", self._reid_enroll_cb, 10)
            self.create_subscription(String, reid_topic_ns + "/forget", self._reid_forget_cb, 10)
            if self.reid_gallery_file:
                self._reid_save_timer = self.create_timer(30.0, self._reid_save)
        from rclpy.qos import QoSDurabilityPolicy
        area_marker_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.area_marker_pub = self.create_publisher(
            MarkerArray, self.area_marker_topic, area_marker_qos,
        )
        self.polar_pub = self.create_publisher(MarkerArray, self.dbg_polar_topic, 10)
        self.ann_pub = self.create_publisher(Image, annotated_topic, ann_qos)
        self.cmp_pub = self.create_publisher(CompressedImage, annotated_topic + "/compressed", ann_qos)
        self.crop_pub = None
        if self.pub_dot_yolo_crops:
            if DotYoloCropArray is None:
                self.get_logger().error(
                    "publish_dot_yolo_crops=true but pose_estimation_interfaces is not built. "
                    "Colcon-build pose-estimation interfaces first, or set publish_dot_yolo_crops: false."
                )
                self.pub_dot_yolo_crops = False
            else:
                self.crop_pub = self.create_publisher(
                    DotYoloCropArray, self.dot_yolo_crops_topic, img_qos
                )
                self.get_logger().info(f"Publishing DOT YOLO crops on {self.dot_yolo_crops_topic}")

        if self._costmap_filter and self.costmap_topic:
            self.create_subscription(OccupancyGrid, self.costmap_topic, self._costmap_cb, 10)

        if self._static_filter and self.static_map_topic:
            from rclpy.qos import QoSDurabilityPolicy
            map_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(OccupancyGrid, self.static_map_topic, self._staticmap_cb, map_qos)

        if self.depth_topic and self.use_depth_fb:
            self.create_subscription(Image, self.depth_topic, self._depth_cb, img_qos)

        self._log_startup()
        if self._area_filter_zones and self.area_filter_pub_markers:
            self._publish_area_zone_markers_heartbeat()
            self._area_zone_timer = self.create_timer(1.0, self._publish_area_zone_markers_heartbeat)
        if self.enable_yolo and self.sync_img_lidar:
            self._sync_watchdog = self.create_timer(10.0, self._sync_watchdog_cb)
        else:
            self._sync_watchdog = None

    def _sync_watchdog_cb(self) -> None:
        if not self.enable_yolo or not self.sync_img_lidar:
            return
        now = time.perf_counter()
        if self._last_frame_wall is None or (now - self._last_frame_wall) > 10.0:
            self.get_logger().warn(
                "YOLO inactivo: no hay frames sincronizados imagen+láser en >10 s. "
                "Prueba sync_image_with_lidar:=false y/o image_qos_reliability:=best_effort "
                "(TIAGO) o :=reliable según ros2 topic info -v (ver README)."
            )

    # ─── Init helpers ────────────────────────────────────────────────

    def _init_yolo(self, model_name: str) -> None:
        pkg_dir = get_package_share_directory("dynamic_tracking")
        self.model = None
        self.class_map = {}
        self._yolo_backend_label = "n/a"

        if self.yolo_backend == "onnx":
            onnx_path = resolve_yolo_onnx_path(self.yolo_onnx_model, pkg_dir)
            backend: YoloInferenceBackend = OnnxYoloBackend(
                onnx_path,
                class_map=None,
                conf=self.conf_threshold,
                imgsz=self.yolo_imgsz,
                max_det=self.yolo_max_det,
                providers=self.yolo_onnx_providers,
                intra_op_threads=self.yolo_onnx_intra_threads,
                is_target=self._is_target,
                inter_op_threads=self.yolo_onnx_inter_threads,
            )
            self.class_map = dict(backend.class_map)
            loaded_src = onnx_path
        else:
            from ultralytics import YOLO

            pkg_model = os.path.join(pkg_dir, "models", model_name)
            src = pkg_model if os.path.exists(pkg_model) else model_name
            self.get_logger().debug(f"Loading YOLO: {src}")
            self.model = YOLO(src)
            self.class_map = self._resolve_model_names()
            nthr = _p(self, "yolo_cpu_num_threads", 0, cast=int)
            if nthr > 0 and "cuda" not in str(self.device).lower():
                try:
                    import torch
                    torch.set_num_threads(nthr)
                except Exception:
                    pass
            if self._yolo_half:
                try:
                    self.model.half()
                except Exception as exc:
                    self.get_logger().warn(f"model.half() failed, using FP32: {exc}")
                    self._yolo_half = False
            backend = UltralyticsYoloBackend(
                self.model,
                self.class_map,
                self.conf_threshold,
                self.yolo_imgsz,
                self.yolo_max_det,
                self.device,
                self._is_target,
            )
            loaded_src = src

        self._yolo_backend_label = backend.backend_label
        self._yolo = _YoloBackgroundWorker(
            backend,
            self._is_target,
            self.get_logger(),
            keep_latest_only=self._yolo_keep_latest,
            on_annotated=(
                self._publish_annotated_from_worker
                if self.pub_annotated and self.yolo_annotated_pair else None
            ),
            stage_timer=self._stage_timer,
            reid_embedder=self._reid_embedder,
            reid_cache=self._reid_cache,
            reid_period_sec=(1.0 / self.reid_rate_hz) if self.reid_rate_hz > 0 else 0.0,
            reid_skip_overlap_iou=self.reid_skip_overlap,
        )
        try:
            backend.warmup()
        except Exception as exc:
            self.get_logger().warn(f"YOLO warmup: {exc}")
        self.get_logger().info(f"YOLO listo ({backend.backend_label}): {loaded_src}")

    def _init_reid(self) -> None:
        from .reid import EmbeddingCache, ReIdEmbedder

        pkg_dir = get_package_share_directory("dynamic_tracking")
        w = self.reid_weights
        if not os.path.isabs(w):
            w = os.path.join(pkg_dir, "models", "reid", w)
        try:
            self._reid_embedder = ReIdEmbedder(
                w, variant=self.reid_model, device=str(self.device), half=self.reid_half,
            )
            self._reid_embedder.warmup()
        except Exception as exc:
            self.get_logger().error(f"ReID desactivado: no se pudo cargar {w}: {exc}")
            self.enable_reid = False
            self._reid_embedder = None
            return
        self._reid_cache = EmbeddingCache()
        self._identity = IdentityManager(self._reid_cfg, log=self.get_logger().info)
        if self.reid_gallery_file:
            try:
                n = self._identity.load_named(self.reid_gallery_file, time.time())
                if n:
                    self.get_logger().info(f"ReID: {n} identidades cargadas de {self.reid_gallery_file}")
            except Exception as exc:
                self.get_logger().warn(f"ReID: no se pudo leer {self.reid_gallery_file}: {exc}")
        self.get_logger().info(
            f"ReID listo ({self.reid_model}, {self._reid_embedder.device}, "
            f"{'fp16' if self._reid_embedder.half else 'fp32'}, {self.reid_rate_hz:.1f} Hz, "
            f"umbral {self._reid_cfg.match_threshold:.2f})"
        )

    def _reid_enroll_cb(self, msg: String) -> None:
        """'<track_id>:<nombre>' o JSON {"track_id": 3, "name": "Edison"}."""
        raw = str(msg.data).strip()
        try:
            if raw.startswith("{"):
                d = json.loads(raw)
                tid, name = int(d["track_id"]), str(d["name"])
            else:
                tid_s, name = raw.split(":", 1)
                tid = int(tid_s)
        except Exception:
            self.get_logger().warn(f"ReID enroll: formato invalido '{raw}' (usa '3:Edison')")
            return
        with self._tracks_lock:
            ok = self._identity.enroll(self.tracks, tid, name)
        if not ok:
            self.get_logger().warn(f"ReID enroll: no existe el track {tid}")
            return
        self._reid_save()

    def _reid_forget_cb(self, msg: String) -> None:
        name = str(msg.data).strip()
        with self._tracks_lock:
            ok = self._identity.forget(name)
        self.get_logger().info(f"ReID: '{name}' {'olvidado' if ok else 'no encontrado'}")
        self._reid_save()

    def _reid_save(self) -> None:
        if not (self._identity and self.reid_gallery_file):
            return
        try:
            with self._tracks_lock:
                self._identity.save_named(self.reid_gallery_file)
        except Exception as exc:
            self.get_logger().warn(f"ReID: no se pudo guardar {self.reid_gallery_file}: {exc}")

    def _reid_publish_identities(self, frame_sec: float) -> None:
        payload = {
            "stamp": float(frame_sec),
            "identities": self._identity.identities(self.tracks),
            "gallery": [
                {"track_id": e.track_id, "name": e.name, "lost_sec": round(e.lost_sec, 2)}
                for e in self._identity.gallery.values()
            ],
        }
        self.reid_pub.publish(String(data=json.dumps(payload)))

    def _resolve_model_names(self) -> Dict[int, str]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(n) for i, n in enumerate(names)}

    @staticmethod
    def _sensor_qos(reliability: str, depth: int) -> QoSProfile:
        rel = QoSReliabilityPolicy.RELIABLE if reliability.strip().lower() == "reliable" else QoSReliabilityPolicy.BEST_EFFORT
        return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=max(1, depth), reliability=rel)

    def _is_target(self, cn: str) -> bool:
        return not self.target_classes or cn.lower() in self.target_classes

    def _log_startup(self) -> None:
        log = self.get_logger().info
        if self._log_verbose_startup:
            log(f"Tracking frame: {self.tracking_frame}")
            if self.enable_yolo:
                yolo_mode = (
                    "sync img+scan (YOLO en _sync_cb)"
                    if self.sync_img_lidar
                    else "async imagen (YOLO en cada frame)"
                )
                stride_s = (
                    f", 1/{self._yolo_stride} frames"
                    if self._yolo_stride > 1
                    else ""
                )
                log(
                    f"YOLO on ({self._yolo_backend_label}, imgsz={self.yolo_imgsz}, "
                    f"{'fp16' if self._yolo_half else 'fp32'}{stride_s}) — {yolo_mode}"
                )
            else:
                log("Pipeline: LiDAR-only (enable_yolo_detections=false)")
            if self.enable_lidar:
                log(f"LiDAR fusion: window={math.degrees(self.lidar_window_rad):.1f}°, "
                    f"lidar=immediate, yolo={'sync' if self.sync_img_lidar else 'async'}")
                if self._lidar_pipeline.dual_tier_enabled:
                    log(
                        f"LiDAR dual-tier: spawn={self.spawn_fusion_pol}, "
                        f"maintenance={self.maint_fusion_pol}, "
                        f"maint_min_leg={self.maint_min_leg:.2f}, "
                        f"confirm_frames={self.spawn_confirm_n}, "
                        f"initiator step={self.spawn_init_step:.2f}m vel={self.spawn_init_vel:.2f}m/s"
                    )
                else:
                    log(
                        f"LiDAR detectors: {self._lidar_pipeline.enabled_detector_ids}, "
                        f"policy={self._lidar_pipeline.fusion_policy}, "
                        f"only_positive_rf={self._rf_only_pos}"
                    )
            if self.viz_frame:
                log(f"Marker remap: {self.tracking_frame} → {self.viz_frame}")
            if self._pose_debug.enabled:
                log(
                    f"Pose debug: every {self._pose_debug.interval * 1000:.0f} ms, "
                    f"robot_frame={self._pose_debug.robot_frame}"
                )
        else:
            yolo_s = f"YOLO {self._yolo_backend_label}" if self.enable_yolo else "LiDAR-only"
            tick = "scan-rate" if self._lidar_track_on_scan else "image-rate"
            log(f"dynamic_tracking listo — frame={self.tracking_frame}, {yolo_s}, "
                f"tracking={tick} (debug:=true para logs [POSE])")

    # ─── Callbacks ───────────────────────────────────────────────────

    def _caminfo_cb(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        self._costmap = msg

    def _staticmap_cb(self, msg: OccupancyGrid) -> None:
        self._static_map = msg
        self.get_logger().debug(f"Static map received: {msg.info.width}x{msg.info.height}")

    def _depth_cb(self, msg: Image) -> None:
        self.latest_depth_msg = msg

    def _scan_cb(self, msg: LaserScan) -> None:
        t0 = time.perf_counter()
        self._apply_scan(msg)
        self._stage_timer.record("scan_total", (time.perf_counter() - t0) * 1000.0)
        self._stage_timer.log_if_due(self.get_logger())

    def _apply_scan(self, msg: LaserScan) -> None:
        msg = repair_interleaved_returns(
            msg, self.interleave_half_win, self.interleave_max_r, self.cluster_jump_m
        )
        scan_frame = msg.header.frame_id or self.latest_scan_frame
        scan_stamp = msg.header.stamp

        tf_at_stamp = self._tf.lookup_planar_stamped_only(
            self.tracking_frame, scan_frame, scan_stamp
        )
        if tf_at_stamp is not None:
            scan_tf_captured = tf_at_stamp
            scan_tf_mode = "stamped"
        else:
            scan_tf_captured = None
            if self.scan_tf_extrap_max > 0.0:
                scan_tf_captured = self._tf.lookup_planar_extrapolated(
                    self.tracking_frame, scan_frame, scan_stamp, self.scan_tf_extrap_max
                )
                scan_tf_mode = "extrapolated"
            if scan_tf_captured is None:
                scan_tf_captured = self._tf.lookup_planar(
                    self.tracking_frame, scan_frame, Time()
                )
                scan_tf_mode = "latest"

        try:
            scan_sec = float(scan_stamp.sec) + float(scan_stamp.nanosec) * 1e-9
            scan_lag_sec = self.get_clock().now().nanoseconds / 1e9 - scan_sec
        except Exception:
            scan_lag_sec = 0.0

        self._scan_tf_captured = scan_tf_captured
        self._scan_tf_mode = scan_tf_mode

        extract_stats = (
            LidarExtractDebugStats() if self._extract_diag.enabled else None
        )
        if self.lidar_dual_tier and self._lidar_pipeline.dual_tier_enabled:
            dual = extract_lidar_dual(
                msg,
                self._lidar_pipeline,
                scan_frame,
                self._transform_xy,
                debug_stats=extract_stats,
            )
            clusters = dual.maintenance
            spawn_clusters = dual.spawn
        else:
            clusters = extract_lidar_clusters(
                msg,
                self._lidar_pipeline,
                scan_frame,
                self._transform_xy,
                debug_stats=extract_stats,
            )
            spawn_clusters = []

        with self._lidar_lock:
            self.latest_scan = msg
            self.latest_scan_frame = scan_frame
            self._scan_tf_captured = scan_tf_captured
            self._scan_tf_mode = scan_tf_mode
            self._scan_lag_sec = scan_lag_sec
            self.latest_lidar_clusters = clusters
            self.latest_lidar_spawn_clusters = spawn_clusters

        if extract_stats is not None:
            self._extract_diag.accumulate(extract_stats, float(self.spawn_min_leg_score))
            self._extract_diag.log_if_due(
                time.perf_counter(),
                self.get_logger(),
                spawn_min_leg_score=float(self.spawn_min_leg_score),
                rf_threshold=float(self._rf_threshold),
                only_positive=bool(self._rf_only_pos),
                scan_frame=scan_frame,
            )

        if self._lidar_track_on_scan:
            wall = time.perf_counter()
            frame_sec = self._stamp_sec(scan_stamp)
            self._lidar_tracking_tick(scan_stamp, frame_sec, wall)
            self._stage_timer.record("scan_track", (time.perf_counter() - wall) * 1000.0)

    def _filtered_lidar_clusters(
        self,
    ) -> Tuple[List[LidarMeasurement], Optional[List[LidarMeasurement]]]:
        with self._lidar_lock:
            raw_clusters = list(self.latest_lidar_clusters)
            spawn_raw = list(self.latest_lidar_spawn_clusters)
        # Maintenance clusters may lie in the area-filter margin (tracks keep their ID
        # along the border); spawn clusters must be strictly inside the zone.
        area_margin = self._area_filter.track_margin_m if self._area_filter else 0.0
        clusters = merge_leg_pairs(
            self._filter_lidar_measurements(raw_clusters, area_margin), self.leg_pair_merge_m
        )
        spawn_clusters: Optional[List[LidarMeasurement]] = None
        if self.lidar_dual_tier and spawn_raw:
            spawn_clusters = merge_leg_pairs(
                self._filter_lidar_measurements(spawn_raw), self.leg_pair_merge_m
            )
        return clusters, spawn_clusters

    def _kalman_dt(self, fusion_sec: float) -> float:
        dt = (1.0 / 30.0) if self._last_kf_sec is None else max(fusion_sec - self._last_kf_sec, 1e-3)
        self._last_kf_sec = fusion_sec
        return min(dt, self.kf_dt_cap) if self.kf_dt_cap > 1e-6 else dt

    def _call_update_tracks(
        self,
        clusters: List[LidarMeasurement],
        spawn_clusters: Optional[List[LidarMeasurement]],
        dt_kf: float,
        frame_sec: float,
    ) -> None:
        self.next_track_id = update_tracks(
            self.tracks, self.next_track_id, [], clusters, dt_kf, frame_sec,
            self.enable_lidar, self.assoc_cost_cam, self.max_missed, self.leg_hyp_min_score,
            self.assoc_cost_lidar, self.assoc_person_lid, self.person_max_missed,
            self.spawn_unknown, self.leg_as_person, self.img_only_fallback, self.img_only_iou_min,
            self.dyn_window, self.dyn_min_disp, self.dyn_hist_sec, self.dyn_classes,
            self.innov_gate_m, self.innov_gate_rgb_s,
            person_lidar_max_correction_jump_m=self.lidar_max_jump,
            person_static_zero_velocity_speed_m=self.static_zero_v,
            scan_verify_data=self._scan_verify_data(),
            scan_verify_range_tolerance_m=self.verify_range_tol,
            scan_verify_window_deg=self.verify_window_deg,
            person_missed_debounce_frames=self.missed_debounce,
            person_lidar_only_max_missed_frames=self.lidar_only_missed,
            lidar_initiation_max_age_static_sec=self.lidar_static_sec,
            lidar_initiation_static_min_travel_m=self.lidar_static_min_travel,
            max_occlusions_before_deletion=self.max_occ,
            max_occlusions_mature_track=self.max_occ_mature,
            use_mahalanobis_lidar=self.use_mahalanobis,
            person_hold_skip_lidar_correct_sec=self.hold_skip_sec,
            person_lidar_spawn_suppress_radius_m=self.spawn_suppress_r,
            person_lidar_spawn_suppress_rgb_sec=self.spawn_suppress_s,
            lidar_spawn_person_min_leg_score=self.spawn_min_leg_score,
            lidar_spawn_same_frame_dedup_m=self.spawn_dedup_m,
            pending_lidar_spawns=self._pending_spawns,
            lidar_spawn_meas=spawn_clusters,
            lidar_spawn_confirm_frames=self.spawn_confirm_n,
            lidar_spawn_confirm_radius_m=self.spawn_confirm_r,
            spawn_initiator_max_step_displacement_m=self.spawn_init_step,
            spawn_initiator_max_velocity_mps=self.spawn_init_vel,
            person_track_dual_audit_enable=self.track_dual_audit,
            person_track_dual_audit_interval_frames=self.track_audit_iv,
            person_track_dual_audit_radius_m=self.track_audit_r,
            person_yolo_seen_green_sec=self.yolo_green_sec,
            person_track_leg_snap_enable=self.leg_snap_on,
            person_track_leg_snap_max_dist_m=self.leg_snap_max,
            person_track_leg_snap_min_leg_score=self.leg_snap_min,
            person_lidar_micro_stable_m=self.lidar_micro_stable,
            person_track_leg_snap_min_jump_for_velocity_m=self.snap_vel_jump_min,
            person_track_leg_snap_max_velocity_mps=self.snap_vel_cap,
            person_track_leg_snap_require_spawn_tier_jump_m=self.snap_spawn_jump,
            scan_verify_require_nearby_cluster=self.verify_need_cluster,
            scan_verify_nearby_cluster_m=self.verify_near_r,
            scan_verify_nearby_min_leg_score=self.verify_near_leg,
            person_track_map_merge_min_iou=self.merge_iou,
            person_track_map_merge_box_m=self.merge_box_m,
            person_track_map_merge_lidar_only=self.merge_lidar_only,
            duplicate_track_temporal_enable=self.dup_enable,
            duplicate_track_compare_samples=self.dup_samples,
            duplicate_track_must_match=self.dup_must_match,
            duplicate_track_max_dist_m=self.dup_max_dist,
            duplicate_track_lidar_only=self.dup_lidar_only,
        )

    def _publish_tracking_outputs(
        self,
        pub_header: Header,
        frame_sec: float,
        clusters: List[LidarMeasurement],
        yolo_parity: bool,
        remap_fbs=None,
    ) -> None:
        self.det_pub.publish(Detection2DArray(header=pub_header))
        self._update_display_positions(frame_sec)
        tracked = tracks_to_detection_array(
            self.tracks, self.min_confirmed, pub_header, self.dyn_suffix,
            require_visual_confirmation=self.require_vis_conf,
        )
        self.track_pub.publish(tracked)
        tracked_fov = tracks_to_detection_array(
            self.tracks, self.min_confirmed, pub_header, self.dyn_suffix,
            require_visual_confirmation=self.require_vis_conf,
            fov_only=True,
            overlay_max_age_sec=self.overlay_age_sec,
            frame_sec=frame_sec,
        )
        self.track_fov_pub.publish(tracked_fov)
        self._publish_markers(pub_header, frame_sec, clusters, yolo_parity, remap_fbs)

    def _update_display_positions(self, frame_sec: float) -> None:
        """Low-pass the drawn box position (RViz only; tracking keeps the raw Kalman state)."""
        tau = self.marker_smooth_tau
        dt = frame_sec - self._last_display_sec if self._last_display_sec > 0.0 else 0.0
        self._last_display_sec = frame_sec
        alpha = 1.0 if tau <= 0.0 or dt <= 0.0 else 1.0 - math.exp(-min(dt, 1.0) / tau)
        for tr in self.tracks:
            tr.update_display(alpha, self.marker_smooth_snap_m)

    def _lidar_tracking_tick(self, fusion_stamp, frame_sec: float, wall_start: float) -> None:
        """Kalman + publicación a frecuencia de /scan (robot async, sin sync imagen)."""
        st = self._stage_timer
        t = time.perf_counter()
        with self._tracks_lock:
            t = st.lap("trk_lock_wait", t)
            self._tf.reset_frame_caches()
            fusion_sec = self._stamp_sec(fusion_stamp)
            dt_kf = self._kalman_dt(fusion_sec)
            clusters, spawn_clusters = self._filtered_lidar_clusters()
            t = st.lap("trk_filter", t)
            self._call_update_tracks(clusters, spawn_clusters, dt_kf, frame_sec)
            t = st.lap("trk_update", t)
            if self._area_filter:
                self._area_filter.filter_tracks(self.tracks, self.tracking_frame, self._tf)
            if self.enable_yolo and self.aux_proj_tracks:
                self._update_camera_fov_state(fusion_stamp, frame_sec)
            t = st.lap("trk_fov", t)
        pub_header = Header(stamp=fusion_stamp, frame_id=self.tracking_frame)
        yolo_parity = self.enable_yolo and self.yolo_visual_par
        with self._tracks_lock:
            self._person_log.log_if_due(
                wall_start,
                self.get_logger(),
                self.tracks,
                min_confirmed_hits=self.min_confirmed,
                n_clusters=len(clusters),
                yolo_detections=-1,
            )
            self._log_pose_debug_if_due(wall_start, clusters)
            self._publish_tracking_outputs(pub_header, frame_sec, clusters, yolo_parity)
        st.lap("trk_publish", t)

    def _transform_xy(self, x, y, frame, stamp) -> Optional[Tuple[float, float]]:
        """Transform laser (x,y) to tracking_frame using TF captured in _apply_scan."""
        if self._scan_tf_captured is not None:
            tx, ty, yaw = self._scan_tf_captured
            c, s = math.cos(yaw), math.sin(yaw)
            return (tx + c * x - s * y, ty + s * x + c * y)
        return self._tf.transform_xy_latest(x, y, frame, self.tracking_frame)

    def _log_pose_debug_if_due(self, wall_sec: float, clusters) -> None:
        if not self._pose_debug.enabled:
            return
        robot_frame = self._pose_debug.robot_frame
        robot_map = self._tf.lookup_planar(self.tracking_frame, robot_frame, Time())
        robot_odom = self._tf.lookup_planar("odom", robot_frame, Time())
        self._pose_debug.log_if_due(
            wall_sec,
            self.get_logger(),
            tracking_frame=self.tracking_frame,
            robot_map_tf=robot_map,
            robot_odom_tf=robot_odom,
            scan_tf_mode=self._scan_tf_mode,
            scan_lag_sec=self._scan_lag_sec,
            scan_frame=self.latest_scan_frame,
            tracks=self.tracks,
            clusters=clusters,
            min_confirmed_hits=self.min_confirmed,
        )

    def _log_tf_detailed(self, x_local, y_local, frame, stamp_sec, result, tf_mode) -> None:
        """Log detailed TF info for diagnosing projection drift."""
        now = time.perf_counter()
        if now - self._debug_tf_last_log < self._debug_tf_log_interval:
            return
        self._debug_tf_last_log = now

        # Get robot pose from TF
        tf_latest = self._tf.lookup_planar(self.tracking_frame, frame, Time())
        
        log = self.get_logger().info
        angle_local = math.degrees(math.atan2(y_local, x_local))
        dist_local = math.hypot(x_local, y_local)
        
        log(f"[TF-DIAG] ========== Projection Debug ==========")
        log(f"[TF-DIAG] Mode: {tf_mode} | Scan stamp: {stamp_sec:.3f}s")
        log(f"[TF-DIAG] LOCAL (laser frame): x={x_local:.3f}m, y={y_local:.3f}m | angle={angle_local:.1f}°, dist={dist_local:.2f}m")
        log(f"[TF-DIAG] MAP (result):         x={result[0]:.3f}m, y={result[1]:.3f}m")
        
        if tf_latest is not None:
            tx, ty, yaw = tf_latest
            log(f"[TF-DIAG] ROBOT pose (TF@latest): x={tx:.3f}m, y={ty:.3f}m, yaw={math.degrees(yaw):.1f}°")
            
            # Compute what result WOULD be with latest TF
            c, s = math.cos(yaw), math.sin(yaw)
            result_latest = (tx + c * x_local - s * y_local, ty + s * x_local + c * y_local)
            diff = math.hypot(result_latest[0] - result[0], result_latest[1] - result[1])
            
            if diff > 0.01:
                log(f"[TF-DIAG] *** DIFF stamped vs latest: {diff:.3f}m ***")
                log(f"[TF-DIAG]   If used latest: x={result_latest[0]:.3f}m, y={result_latest[1]:.3f}m")
            else:
                log(f"[TF-DIAG] Stamped == Latest (no robot motion detected)")
        else:
            log(f"[TF-DIAG] WARNING: TF@latest not available!")
        
        log(f"[TF-DIAG] =========================================")

    def _log_tf_debug(self, x_local, y_local, frame, stamp, result) -> None:
        """Log TF projection debug info to diagnose rotation-induced drift."""
        now = time.perf_counter()
        if now - self._debug_tf_last_log < self._debug_tf_log_interval:
            return
        self._debug_tf_last_log = now

        angle_local = math.degrees(math.atan2(y_local, x_local))
        dist_local = math.hypot(x_local, y_local)

        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9 if hasattr(stamp, 'sec') else 0.0

        tf_stamped_only = self._tf.lookup_planar_stamped_only(self.tracking_frame, frame, stamp)
        tf_latest = self._tf.lookup_planar(self.tracking_frame, frame, Time())

        log = self.get_logger().info
        log(f"[TF DEBUG] Scan stamp: {stamp_sec:.3f}s")
        log(f"[TF DEBUG] Local (laser): x={x_local:.2f}, y={y_local:.2f}, "
            f"angle={angle_local:.1f}°, dist={dist_local:.2f}m")
        log(f"[TF DEBUG] Projected (map): x={result[0]:.2f}, y={result[1]:.2f}")

        if tf_stamped_only is not None:
            tx_s, ty_s, yaw_s = tf_stamped_only
            log(f"[TF DEBUG] TF@stamp: tx={tx_s:.2f}, ty={ty_s:.2f}, yaw={math.degrees(yaw_s):.1f}°")
        else:
            log(f"[TF DEBUG] TF@stamp: NOT AVAILABLE (stamp={stamp_sec:.3f}s)")

        if tf_latest is not None:
            tx_l, ty_l, yaw_l = tf_latest
            log(f"[TF DEBUG] TF@latest: tx={tx_l:.2f}, ty={ty_l:.2f}, yaw={math.degrees(yaw_l):.1f}°")

            if tf_stamped_only is not None:
                yaw_diff = math.degrees(yaw_l - yaw_s)
                pos_diff = math.hypot(tx_l - tx_s, ty_l - ty_s)
                if abs(yaw_diff) > 0.5 or pos_diff > 0.01:
                    log(f"[TF DEBUG] *** DRIFT: yaw_diff={yaw_diff:.1f}°, pos_diff={pos_diff:.3f}m ***")
            else:
                log(f"[TF DEBUG] *** WARNING: Using latest TF, projection may drift ***")

    def _filter_lidar_measurements(
        self, clusters: List[LidarMeasurement], area_margin_m: float = 0.0,
    ) -> List[LidarMeasurement]:
        out = clusters
        if self._costmap_filter:
            out = self._costmap_filter.filter(out, self._costmap, self.tracking_frame, self._tf)
        if self._static_filter:
            out = self._static_filter.filter(out, self._static_map, self.tracking_frame, self._tf)
        if self._area_filter:
            out = self._area_filter.filter_measurements(
                out, self.tracking_frame, self._tf, area_margin_m
            )
        if self._rf_post_min > 1e-9 and "spencer_rf" in self._lidar_pipeline.enabled_detector_ids:
            out = [c for c in out if float(c.leg_candidate_score) >= float(self._rf_post_min)]
        return out

    def _sync_cb(self, img_msg: Image, scan_msg: LaserScan) -> None:
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            self._process_frame(cv_img, img_msg.header)
        except Exception as exc:
            self.get_logger().error(f"Synced frame error: {exc}")

    def _image_cb(self, msg: Image) -> None:
        try:
            t0 = time.perf_counter()
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._process_frame(cv_img, msg.header)
            self._stage_timer.record("image_cb", (time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            self.get_logger().error(f"Image error: {exc}")

    # ─── Scan verify ─────────────────────────────────────────────────

    def _scan_verify_data(self):
        scan = self.latest_scan
        if scan is None or not self.latest_scan_frame:
            return None
        t = self._tf._raw_lookup(self.tracking_frame, self.latest_scan_frame, Time())
        if t is None:
            return None
        tx, ty, yaw = self._tf._extract_planar(t)
        return (scan, tx, ty, yaw)

    # ─── LiDAR→YOLO measurement pipeline ────────────────────────────

    def _yolo_boxes_to_measurements(
        self, raw: List[_RawYoloDet], w: int, h: int, stamp, optical_frame: str,
    ) -> List[YoloMeasurement]:
        out: List[YoloMeasurement] = []
        laser_frame = (self.latest_scan_frame or "").strip()
        for bx1, by1, bx2, by2, cn, sc in raw:
            u = 0.5 * (bx1 + bx2)
            v = float(by2) if self.ray_bbox_bottom else 0.5 * (by1 + by2)
            est, reason = self._estimate_lidar(u, v, w, h, stamp, optical_frame)
            map_xy: Optional[Tuple[float, float]] = None
            mode = "none"
            lr_m = lb_rad = None
            if est is not None:
                lr_m, lb_rad = float(est[0]), float(est[1])
                xl = lr_m * math.cos(lb_rad)
                yl = lr_m * math.sin(lb_rad)
                mxy = self._tf.transform_xy_latest(xl, yl, laser_frame, self.tracking_frame) if laser_frame else None
                if mxy is not None:
                    map_xy, mode = mxy, "lidar_rgb"
            if self._lidar_diag.enabled and self.enable_lidar and self._is_target(cn):
                self._lidar_diag.dets += 1
                if est is not None:
                    self._lidar_diag.ray_ok += 1
                    if map_xy:
                        self._lidar_diag.map_lidar += 1
                    elif laser_frame:
                        self._lidar_diag.tf_map_fail += 1
                    else:
                        self._lidar_diag.no_map += 1
                else:
                    self._lidar_diag.ray_miss[str(reason)] += 1
            out.append(YoloMeasurement(
                class_name=cn, score=sc, bbox=(bx1, by1, bx2, by2),
                map_xy=map_xy, fusion_mode=mode, laser_range_m=lr_m, laser_bearing_rad=lb_rad,
                embedding=(
                    self._reid_cache.get((bx1, by1, bx2, by2))
                    if self._reid_cache is not None and cn.lower() == "person" else None
                ),
            ))
        return out

    def _estimate_lidar(self, u, v, w, h, stamp, optical_frame):
        if self.latest_scan is None or w <= 0:
            return None, "no_scan"
        scan = self.latest_scan
        if scan.angle_increment == 0.0:
            return None, "zero_increment"
        angle: Optional[float] = None
        used_tf = False
        if self.use_tf_lidar_ray and optical_frame:
            angle = bearing_laser_from_optical_ray(
                u, v, w, h, stamp, optical_frame, self.latest_scan_frame,
                self.latest_camera_info, self.fov_rad, self.angle_offset_rad,
                self._tf, extra_timeout=self.aux_tf_extra,
            )
            used_tf = angle is not None
        if angle is None:
            ci = self.latest_camera_info
            fx, _, cx, _ = scaled_intrinsics(ci, w, h, self.fov_rad)
            angle = math.atan2(u - cx, max(fx, 1e-6)) + self.angle_offset_rad
        br = scan_window_best_range(scan, self.lidar_window_rad, angle)
        if br is None:
            return None, "no_hit" + ("_tf" if used_tf else "")
        return (br[0], br[1]), "ok"

    # ─── Annotated image helpers ─────────────────────────────────────

    @staticmethod
    def _parse_bgr_param(raw, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            return (int(raw[0]), int(raw[1]), int(raw[2]))
        return default

    def _yolo_box_color(self, class_name: str) -> Tuple[int, int, int]:
        return (0, 220, 0) if str(class_name).lower() == "person" else (200, 200, 0)

    @staticmethod
    def _put_text(img, text, xy, scale=0.55, fg=(255, 255, 255)):
        x, y = int(xy[0]), int(xy[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, fg, 1, cv2.LINE_AA)

    @staticmethod
    def _stamp_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    # ─── Publish markers ─────────────────────────────────────────────

    def _area_zone_marker_stamp(self, header):
        """Use map frame stamp; fall back to node clock if header is missing."""
        if header is not None and hasattr(header, "stamp"):
            return header.stamp
        return self.get_clock().now().to_msg()

    def _append_area_zone_markers(self, arr, stamp) -> None:
        if not (self._area_filter and self.area_filter_pub_markers and self._area_filter_zones):
            return
        zone_arr = build_area_zone_marker_array(
            self._area_filter_zones, self._area_filter_frame, stamp,
        )
        arr.markers.extend(zone_arr.markers)

    def _publish_area_zone_markers_heartbeat(self) -> None:
        if not (self.area_filter_pub_markers and self._area_filter_zones):
            return
        stamp = self.get_clock().now().to_msg()
        arr = build_area_zone_marker_array(
            self._area_filter_zones, self._area_filter_frame, stamp,
        )
        self.area_marker_pub.publish(arr)

    def _publish_markers(self, header, frame_sec, lidar_clusters, yolo_parity, remap_fbs=None):
        arr = build_track_marker_array(
            self.tracks, header, self.tracking_frame,
            self.pub_unknown, self.unk_min_speed, self.lidar_only_motion,
            self.motion_speed_thr, self.motion_accel_thr,
            self.person_box_xy, self.person_box_h, self.unk_box_xy, self.unk_box_h,
            self.marker_lifetime, self.text_scale_z, self.compact_labels,
            self.text_height_m, self.text_track_color,
            self.text_r, self.text_g, self.text_b,
            self.label_ring_r, self.label_z_stagger,
            self.uncertain_static, self.stale_rgb_sec, self.rgb_coast_sec,
            self.hide_uncertain,
            marker_hide_person_if_missed_frames=self.hide_missed_person,
            require_visual_confirmation=self.require_vis_conf,
            min_confirmed_hits=self.min_confirmed,
            yolo_visual_lidar_parity=yolo_parity,
            person_yolo_seen_green_sec=self.yolo_green_sec,
            marker_color_coast_bgr=self.marker_coast_bgr,
            marker_bbox_alpha=self.bbox_alpha,
            marker_bbox_wireframe=self.bbox_wire,
            marker_bbox_line_width_m=self.bbox_line_w,
            marker_bbox_use_velocity_heading=self.bbox_vel_heading,
            marker_bbox_heading_min_speed_mps=self.bbox_heading_min,
            publish_track_trajectory=self.pub_traj,
            track_trajectory_z_m=self.traj_z,
            track_trajectory_line_width_m=self.traj_line_w,
            track_trajectory_alpha=self.traj_alpha,
            publish_track_history_cones=self.pub_cones,
            track_history_max_markers=self.hist_max,
            track_history_cylinder_height_min_m=self.hist_h_min,
            track_history_cylinder_height_max_m=self.hist_h_max,
            track_history_cylinder_diameter_min_m=self.hist_d_min,
            track_history_cylinder_diameter_max_m=self.hist_d_max,
            track_history_alpha_min=self.hist_a_min,
            track_history_alpha_max=self.hist_a_max,
            track_history_z_m=self.hist_z,
            label_stamp_sec=frame_sec,
        )
        if self.pub_clusters and self.enable_lidar and lidar_clusters:
            cm = build_lidar_cluster_marker_array(
                lidar_clusters, header, self.tracking_frame, self.marker_lifetime,
                sphere_diameter_m=self.cluster_diam,
            )
            arr.markers.extend(cm.markers)

        if self.viz_frame and self.viz_frame != self.tracking_frame:
            rtimeout = self.remap_timeout if self.remap_timeout > 0 else max(0.2, self.tf_timeout)
            arr = remap_marker_array(
                arr, self.tracking_frame, self.viz_frame, header.stamp,
                self._tf, self._remap_tf,
                prefer_latest=self.remap_latest,
                allow_latest_fallback=self.remap_latest_fb,
                stamp_fallbacks=remap_fbs,
                timeout_sec=rtimeout,
                stale_cache_sec=self.remap_stale_sec,
                logger=self.get_logger(),
                _warned_ref=self._warned_remap,
            )
        self._append_area_zone_markers(arr, self._area_zone_marker_stamp(header))
        self.marker_pub.publish(arr)

    # ─── Main frame processing ───────────────────────────────────────

    def _process_frame(self, cv_image: np.ndarray, header) -> None:
        wall_start = time.perf_counter()
        self._last_frame_wall = wall_start
        if header.frame_id:
            self._last_optical_frame = str(header.frame_id).strip()

        if self.pub_annotated and self._yolo is not None and self.yolo_annotated_pair:
            self._publish_pending_annotated()

        if self._lidar_track_on_scan:
            self._process_image_yolo_light(cv_image, header, wall_start)
            return

        self._tf.reset_frame_caches()

        h_img, w_img = cv_image.shape[:2]
        frame_sec = self._stamp_sec(header.stamp)

        fusion_stamp = header.stamp
        if self.use_scan_stamp_tf and self.latest_scan is not None:
            fusion_stamp = self.latest_scan.header.stamp

        pub_header = Header(stamp=fusion_stamp, frame_id=self.tracking_frame)

        # ── Submit to YOLO (optional frame stride saves CPU on TIAGO) ──
        stamp_key: StampKey = (int(header.stamp.sec), int(header.stamp.nanosec))
        if self._yolo is not None:
            self._yolo_frame_ctr += 1
            if self._yolo_frame_ctr % self._yolo_stride == 0:
                self._yolo.submit(cv_image, stamp_key, header.frame_id)

        clusters, spawn_clusters = self._filtered_lidar_clusters()

        if not self._lidar_track_on_scan:
            fusion_sec = self._stamp_sec(fusion_stamp)
            with self._tracks_lock:
                dt_kf = self._kalman_dt(fusion_sec)
                self._call_update_tracks(clusters, spawn_clusters, dt_kf, frame_sec)
                if self._area_filter:
                    self._area_filter.filter_tracks(self.tracks, self.tracking_frame, self._tf)

        if self._lidar_diag.enabled:
            with self._lidar_lock:
                raw_n = len(self.latest_lidar_clusters)
                latest_scan = self.latest_scan
            self._lidar_diag.frames += 1
            self._lidar_diag.clusters_raw += raw_n
            self._lidar_diag.clusters_kept += len(clusters)
            thr = float(self.spawn_min_leg_score)
            if raw_n:
                with self._lidar_lock:
                    raw_clusters = list(self.latest_lidar_clusters)
                self._lidar_diag.max_leg_raw_sum += max(
                    float(c.leg_candidate_score) for c in raw_clusters
                )
                self._lidar_diag.spawn_eligible_raw_sum += sum(
                    1 for c in raw_clusters if float(c.leg_candidate_score) >= thr
                )
            if clusters:
                self._lidar_diag.max_leg_score_sum += max(
                    float(c.leg_candidate_score) for c in clusters
                )
                self._lidar_diag.spawn_eligible_sum += sum(
                    1 for c in clusters if float(c.leg_candidate_score) >= thr
                )
            if self.enable_lidar and latest_scan is not None:
                skew = abs(self._stamp_sec(header.stamp) - self._stamp_sec(latest_scan.header.stamp))
                self._lidar_diag.stamp_skew_sum += skew
                self._lidar_diag.stamp_skew_n += 1
            if not self._lidar_track_on_scan:
                self._lidar_diag.track_sum += len(self.tracks)
                self._lidar_diag.track_confirmed_sum += sum(
                    1 for t in self.tracks if t.hits >= self.min_confirmed
                )

        # ── YOLO auxiliary pipeline ──
        yolo_ms = 0.0
        boxes_assoc: List[_RawYoloDet] = []
        boxes_overlay: List[_RawYoloDet] = []
        if self._yolo is not None:
            boxes_assoc, boxes_overlay, yolo_ms, _ = self._fetch_yolo_boxes(stamp_key, frame_sec)

        if not self._lidar_track_on_scan:
            self._person_log.log_if_due(
                wall_start,
                self.get_logger(),
                self.tracks,
                min_confirmed_hits=self.min_confirmed,
                n_clusters=len(clusters),
                yolo_detections=(
                    len(boxes_assoc) if self.yolo_annotated_pair else len(boxes_overlay)
                ) if self.enable_yolo else -1,
                yolo_ms=yolo_ms,
            )
            self._log_pose_debug_if_due(wall_start, clusters)

        if self._lidar_diag.enabled and boxes_assoc:
            for *_, cn, _ in boxes_assoc:
                if self._is_target(cn):
                    self._lidar_diag.yolo_target_sum += 1

        dbg_xy: Optional[Dict[int, Tuple[float, float]]] = None
        if boxes_assoc and self.enable_yolo:
            with self._tracks_lock:
                dbg_xy = self._yolo_auxiliary(
                    boxes_assoc, w_img, h_img, fusion_stamp, header.frame_id, frame_sec,
                )
        elif self.tracks and self.enable_yolo and self.dbg_polar and self.latest_scan_frame:
            self._publish_polar_debug([], fusion_stamp)

        yolo_parity = self.enable_yolo and self.yolo_visual_par
        remap_fbs = None
        if self.remap_img_fb:
            fs, hs = fusion_stamp, header.stamp
            if (int(fs.sec), int(fs.nanosec)) != (int(hs.sec), int(hs.nanosec)):
                if not (self.dbg_proj or self.dbg_polar):
                    remap_fbs = [header.stamp]

        with self._tracks_lock:
            self._publish_tracking_outputs(
                pub_header, frame_sec, clusters, yolo_parity, remap_fbs,
            )

        if self.crop_pub is not None:
            from .dot_yolo_crops import build_dot_yolo_crop_array

            crop_arr = build_dot_yolo_crop_array(
                self.tracks,
                cv_image,
                header,
                self.bridge,
                min_confirmed_hits=self.min_confirmed,
                margin_px=self.dot_yolo_crop_margin,
                require_visual_confirmation=self.require_vis_conf,
                overlay_max_age_sec=self.overlay_age_sec,
                frame_sec=frame_sec,
                lidar_clusters=clusters,
                min_leg_score=float(self.leg_hyp_min_score),
            )
            self.crop_pub.publish(crop_arr)

        # ── Annotated image ──
        if self.pub_annotated and not self.yolo_annotated_pair:
            if self.dbg_proj and dbg_xy:
                for _, (uu, vv) in dbg_xy.items():
                    cv2.circle(cv_image, (int(round(uu)), int(round(vv))), 6, (255, 0, 255), 2)
            if boxes_overlay:
                for bx1, by1, bx2, by2, cn, sc in boxes_overlay:
                    color = self._yolo_box_color(cn)
                    cv2.rectangle(cv_image, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
                    self._put_text(cv_image, f"{cn} {sc:.2f}", (int(bx1), max(int(by1) - 6, 0)),
                                   scale=0.45, fg=color)
            ann_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            ann_msg.header = header
            self.ann_pub.publish(ann_msg)
            if self.pub_compressed:
                cmsg = CompressedImage(header=header, format="jpeg")
                enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(self.jpeg_quality, 1, 95))]
                cmsg.data = np.array(cv2.imencode(".jpg", cv_image, enc)[1]).tobytes()
                self.cmp_pub.publish(cmsg)

        with self._tracks_lock:
            for tr in self.tracks:
                vx, vy = tr.velocity()
                tr.prev_speed_mps = math.hypot(vx, vy)

        wall_end = time.perf_counter()
        self._pipe_timing.update(wall_start, wall_end, yolo_ms, self.get_logger())
        self._lidar_diag.log_if_due(
            wall_end, self.enable_lidar, self._costmap_filter is not None,
            self._costmap_filter.neighbor_radius if self._costmap_filter else 0,
            self.get_logger(),
            enable_yolo=self.enable_yolo,
            spencer_rf_ready="spencer_rf" in self._lidar_pipeline.enabled_detector_ids,
            spencer_only_positive=self._rf_only_pos,
            pub_lidar_clusters=self.pub_clusters,
            scan_ok=self.latest_scan is not None,
            marker_topic=self.marker_topic,
            tracking_frame=self.tracking_frame,
            viz_frame=self.viz_frame,
            spawn_min_leg_score=float(self.spawn_min_leg_score),
        )

    # ─── YOLO sub-steps ──────────────────────────────────────────────

    def _process_image_yolo_light(
        self, cv_image: np.ndarray, header, wall_start: float,
    ) -> None:
        """Robot async: YOLO + imagen anotada; LiDAR/Kalman/markers en /scan."""
        h_img, w_img = cv_image.shape[:2]
        frame_sec = self._stamp_sec(header.stamp)
        stamp_key: StampKey = (int(header.stamp.sec), int(header.stamp.nanosec))
        fusion_stamp = header.stamp
        if self.use_scan_stamp_tf and self.latest_scan is not None:
            fusion_stamp = self.latest_scan.header.stamp

        if self._yolo is not None:
            self._yolo_frame_ctr += 1
            if self._yolo_frame_ctr % self._yolo_stride == 0:
                self._yolo.submit(cv_image, stamp_key, header.frame_id)

        boxes_assoc: List[_RawYoloDet] = []
        boxes_stamp: List[_RawYoloDet] = []
        boxes_overlay: List[_RawYoloDet] = []
        yolo_ms = 0.0
        if self._yolo is not None:
            boxes_assoc, boxes_overlay, yolo_ms, boxes_stamp = self._fetch_yolo_boxes(
                stamp_key, frame_sec,
            )

        aux_boxes = boxes_stamp
        if not aux_boxes and self.yolo_age_sec > 0 and self._yolo is not None:
            lb, ls, lm = self._yolo.get_latest()
            if (
                lm > 0.0
                and ls > 0.0
                and abs(frame_sec - ls) <= self.yolo_age_sec
            ):
                aux_boxes = list(lb)
        if aux_boxes and self.enable_yolo:
            with self._tracks_lock:
                self._yolo_auxiliary(
                    aux_boxes, w_img, h_img, fusion_stamp, header.frame_id, frame_sec,
                )

        if self.pub_annotated and not self.yolo_annotated_pair:
            if boxes_overlay:
                for bx1, by1, bx2, by2, cn, sc in boxes_overlay:
                    color = self._yolo_box_color(cn)
                    cv2.rectangle(
                        cv_image, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2,
                    )
                    self._put_text(
                        cv_image, f"{cn} {sc:.2f}",
                        (int(bx1), max(int(by1) - 6, 0)), scale=0.45, fg=color,
                    )
            ann_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            ann_msg.header = header
            self.ann_pub.publish(ann_msg)
            if self.pub_compressed:
                cmsg = CompressedImage(header=header, format="jpeg")
                enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(self.jpeg_quality, 1, 95))]
                cmsg.data = np.array(cv2.imencode(".jpg", cv_image, enc)[1]).tobytes()
                self.cmp_pub.publish(cmsg)

        with self._tracks_lock:
            for tr in self.tracks:
                vx, vy = tr.velocity()
                tr.prev_speed_mps = math.hypot(vx, vy)

        self._pipe_timing.update(wall_start, time.perf_counter(), yolo_ms, self.get_logger())

    def _fetch_yolo_boxes(self, stamp_key, frame_sec):
        t0 = time.perf_counter()
        matched, ms = self._yolo.take_boxes_for_stamp(stamp_key, self.yolo_wait_sec)
        stamp_boxes = list(matched)
        assoc = list(stamp_boxes)
        if not assoc and self.yolo_fb_latest:
            lb, ls, lm = self._yolo.get_latest()
            if (
                lm > 0.0
                and ls > 0.0
                and abs(frame_sec - ls) <= self.yolo_age_sec
            ):
                assoc, ms = list(lb), max(ms, lm)
        # Overlay en imagen en vivo: solo cajas del mismo stamp (evita bbox desfasada).
        overlay = list(matched) if not self.yolo_annotated_pair else []
        yolo_ms = max(ms, (time.perf_counter() - t0) * 1000.0)
        return assoc, overlay, yolo_ms, stamp_boxes

    def _publish_pending_annotated(self) -> None:
        if self._yolo is None:
            return
        for img, sk, boxes, frame_id in self._yolo.drain_annotated():
            hdr = Header(
                stamp=BuiltinTime(sec=int(sk[0]), nanosec=int(sk[1])),
                frame_id=frame_id or "camera_optical_frame",
            )
            self._publish_annotated_image(img, hdr, boxes)

    def _publish_annotated_from_worker(
        self, img: np.ndarray, sk: StampKey, boxes: List[_RawYoloDet], frame_id: str,
    ) -> None:
        hdr = Header(
            stamp=BuiltinTime(sec=int(sk[0]), nanosec=int(sk[1])),
            frame_id=frame_id or "camera_optical_frame",
        )
        t0 = time.perf_counter()
        self._publish_annotated_image(img, hdr, boxes)
        self._stage_timer.record("ann_publish", (time.perf_counter() - t0) * 1000.0)

    def _publish_annotated_image(
        self, cv_image: np.ndarray, header: Header, boxes: List[_RawYoloDet],
    ) -> None:
        out = cv_image
        for bx1, by1, bx2, by2, cn, sc in boxes:
            color = self._yolo_box_color(cn)
            cv2.rectangle(out, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
            self._put_text(out, f"{cn} {sc:.2f}", (int(bx1), max(int(by1) - 6, 0)),
                           scale=0.45, fg=color)
        ann_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        ann_msg.header = header
        self.ann_pub.publish(ann_msg)
        if self.pub_compressed:
            cmsg = CompressedImage(header=header, format="jpeg")
            enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(self.jpeg_quality, 1, 95))]
            cmsg.data = np.array(cv2.imencode(".jpg", out, enc)[1]).tobytes()
            self.cmp_pub.publish(cmsg)

    def _update_camera_fov_state(self, stamp, frame_sec) -> Optional[Dict[int, Tuple[float, float]]]:
        """Proyecta tracks a imagen y marca dentro/fuera del FOV camara."""
        if not (self.aux_proj_tracks and self.tracks):
            return None
        ci = self.latest_camera_info
        optical = self._last_optical_frame.strip()
        if ci is None or not optical:
            return None
        w, h = int(ci.width), int(ci.height)
        if w < 1 or h < 1:
            return None
        track_xy = project_tracks_to_image(
            self.tracks, w, h, optical, stamp,
            ci, self.aux_proj_h, self.tracking_frame,
            self._tf, extra_timeout=self.aux_tf_extra,
        )
        update_yolo_auxiliary_camera_fov_state(
            self.tracks, frame_sec, track_xy, min_track_hits=max(1, self.aux_min_hits),
        )
        return track_xy

    def _yolo_auxiliary(self, boxes, w, h, fusion_stamp, optical_frame, frame_sec):
        tf_now = BuiltinTime()
        if optical_frame:
            self._last_optical_frame = str(optical_frame).strip()
        measurements = self._yolo_boxes_to_measurements(boxes, w, h, tf_now, optical_frame)
        track_xy: Optional[Dict[int, Tuple[float, float]]] = None
        if self.aux_proj_tracks:
            track_xy = self._update_camera_fov_state(tf_now, frame_sec)
            if track_xy is None and boxes:
                track_xy = project_tracks_to_image(
                    self.tracks, w, h, optical_frame, tf_now,
                    self.latest_camera_info, self.aux_proj_h, self.tracking_frame,
                    self._tf, extra_timeout=self.aux_tf_extra,
                )

        track_br = yolo_br = None
        bmax = self.aux_bearing_max
        if measurements and self.aux_bearing_on and bmax > 1e-9:
            track_br = {}
            for ti, tr in enumerate(self.tracks):
                tx, ty = tr.state()
                b = bearing_laser_from_track(
                    tx, ty, tf_now, self.latest_scan_frame, self.tracking_frame,
                    self.angle_offset_rad, self._tf,
                )
                if b is not None:
                    track_br[ti] = b
            yolo_br = {}
            for i, ym in enumerate(measurements):
                if ym.class_name.lower() != "person":
                    continue
                bx1, by1, bx2, by2 = ym.bbox
                u = 0.5 * (bx1 + bx2)
                v = float(by2) if self.ray_bbox_bottom else 0.5 * (by1 + by2)
                br = bearing_laser_from_optical_ray(
                    u, v, w, h, tf_now, optical_frame, self.latest_scan_frame,
                    self.latest_camera_info, self.fov_rad, self.angle_offset_rad,
                    self._tf, extra_timeout=self.aux_tf_extra,
                )
                if br is not None:
                    yolo_br[i] = br

        track_polar = None
        if self.aux_polar_on or self.dbg_polar:
            track_polar = all_tracks_laser_polar(
                self.tracks, tf_now, self.latest_scan_frame, self.tracking_frame, self._tf,
            )

        person_matched: Set[int] = set()
        if measurements:
            person_matched = apply_yolo_auxiliary_person_confirmation(
                self.tracks, measurements, frame_sec,
                self.aux_max_dist, self.yolo_iou_min,
                min_track_hits=max(1, self.aux_min_hits),
                distance_match_lidar_ray_only=self.aux_dist_lidar,
                track_image_xy=track_xy,
                image_point_bbox_margin_px=self.aux_proj_margin,
                track_laser_bearing_rad=track_br,
                yolo_meas_laser_bearing_rad=yolo_br,
                bearing_match_max_rad=bmax if self.aux_bearing_on else 0.0,
                polar_match_enable=self.aux_polar_on,
                polar_match_max_range_m=self.aux_polar_range,
                polar_match_max_bearing_rad=self.aux_polar_bear,
                track_laser_polar=track_polar,
                reid_cfg=self._reid_cfg if self._identity is not None else None,
            )
            if self.yolo_reject_fov:
                apply_yolo_auxiliary_person_rejection(
                    self.tracks, frame_sec, track_xy, person_matched,
                    reject_streak_frames=self.yolo_reject_frames,
                    min_track_hits=max(1, self.aux_min_hits),
                    prune_missed_frames=self.yolo_reject_prune,
                )

        if self.dbg_polar and self.latest_scan_frame:
            yolo_pl = [
                (float(ym.laser_range_m), float(ym.laser_bearing_rad))
                for ym in measurements
                if ym.class_name.lower() == "person" and ym.laser_range_m is not None and ym.laser_bearing_rad is not None
            ]
            self._publish_polar_debug(yolo_pl, fusion_stamp, track_polar)

        if self._identity is not None:
            self._identity.step(self.tracks, frame_sec)
            self._reid_publish_identities(frame_sec)
            hz = (self.reid_rate_pending_hz if self._identity.any_pending(self.tracks)
                  else self.reid_rate_hz)
            if self._yolo is not None:
                self._yolo.set_reid_rate(hz)

        return track_xy

    def _publish_polar_debug(self, yolo_pl=None, fusion_stamp=None, track_polar=None):
        ph = Header(stamp=BuiltinTime(), frame_id=self.latest_scan_frame.strip())
        if track_polar is None:
            track_polar = all_tracks_laser_polar(
                self.tracks, BuiltinTime(), self.latest_scan_frame, self.tracking_frame, self._tf,
            )
        dbg = build_yolo_laser_polar_debug_marker_array(
            ph, yolo_pl or [], track_polar,
            marker_lifetime_sec=0.0, line_width_m=max(0.004, self.dbg_polar_line_w),
        )
        self.polar_pub.publish(dbg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DynamicTrackingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
