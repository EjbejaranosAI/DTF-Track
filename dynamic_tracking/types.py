"""Types and Kalman state shared by the tracking node."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from .ekf2d import ConstantVelocityEKF


@dataclass
class YoloMeasurement:
    class_name: str
    score: float
    bbox: Tuple[float, float, float, float]
    map_xy: Optional[Tuple[float, float]]
    fusion_mode: str = "none"  # lidar_rgb | rgb_only | none
    # /scan hit inside the bbox window (same convention as scan.angle_min + i*increment);
    # allows matching without a TF to the map (polar phase in apply_yolo_auxiliary).
    laser_range_m: Optional[float] = None
    laser_bearing_rad: Optional[float] = None
    # Re-ID embedding (OSNet, L2-normalized) if it was computed for this box.
    embedding: Optional[np.ndarray] = None


@dataclass
class LidarMeasurement:
    map_x: float
    map_y: float
    # 0..1 geometric leg-like score (cluster span, range flatness, chord width)
    leg_candidate_score: float = 0.0
    # True if the segment would pass spawn fusion (e.g. require_two); periodic track audit.
    spawn_tier_pass: bool = False


@dataclass
class LidarDualExtract:
    """Strict clusters for spawn + lax clusters for track maintenance (one /scan pass)."""

    spawn: List[LidarMeasurement]
    maintenance: List[LidarMeasurement]
    n_segments: int = 0


@dataclass
class PendingLidarSpawn:
    """Accumulates laser-only person detections at ~the same (map) spot before creating a Track."""

    map_x: float
    map_y: float
    streak: int
    last_stamp_sec: float
    leg_candidate_score: float
    class_name: str
    score: float
    leg_hypothesis_active: bool
    touched_this_cycle: bool = False
    created_stamp_sec: float = 0.0


class Track:
    def __init__(
        self,
        track_id: int,
        class_name: str,
        map_x: float,
        map_y: float,
        score: float,
        stamp_sec: float,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.score = score
        self.last_bbox = bbox
        self.last_update_source = "unknown"
        self.last_seen_sec = stamp_sec
        self.created_sec = stamp_sec
        self.missed_frames = 0
        # Consecutive cycles without cluster or scan_verify (FOV only); debounce before missed++.
        self.consecutive_no_lidar_presence = 0
        self.hits = 1
        self.prev_speed_mps = 0.0
        # LiDAR-only person hypothesis from leg geometry (cleared when camera updates)
        self.leg_hypothesis_active = False
        self.leg_hypothesis_score = 0.0
        # Last time this track received a YOLO person class (0 = never; RViz highlight).
        self.last_rgb_person_sec = 0.0
        # Camera FOV (track->image projection).
        self.last_camera_fov_sec = 0.0
        self.last_outside_camera_fov_sec = 0.0
        # Consecutive cycles in the FOV without a YOLO person match (negative rejection).
        self.yolo_in_fov_reject_streak = 0
        # Last correct() from a LiDAR cluster (0 = never).
        self.last_lidar_correct_sec = 0.0
        self.last_lidar_map_x = float(map_x)
        self.last_lidar_map_y = float(map_y)
        # Image stamp at which last_bbox was updated with a real detection (avoids a ghost bbox on /image_annotated).
        self.last_image_bbox_stamp_sec = float(stamp_sec) if bbox is not None else 0.0
        # Samples (stamp, map_x, map_y) after a map correct() — real displacement vs static FP.
        self.motion_pose_history: Deque[Tuple[float, float, float]] = deque(maxlen=48)
        # Ring of (x,y) positions after each cycle: temporal duplicates (Spencer) + RViz trail.
        self.pose_trail_ring: Deque[Tuple[float, float]] = deque(maxlen=64)
        # Largest distance (m) of the Kalman state from the birth point: a person who already
        # walked is not furniture even if they are standing still now.
        self.spawn_xy = (float(map_x), float(map_y))
        self.max_travel_from_spawn_m = 0.0
        self.motion_dynamic_confirmed = False
        # Spencer occlusion: separate counter for out-of-FOV cycles.
        self.consecutive_occlusions = 0
        # Smoothed position for visualization only (RViz); tracking uses state().
        self.display_xy: Optional[Tuple[float, float]] = None
        # Appearance Re-ID (identity.py): embeddings of the associated YOLO person crops.
        self.reid_appearance: Optional[np.ndarray] = None
        self.reid_bank: Optional[Deque[np.ndarray]] = None
        self.reid_last_emb: Optional[np.ndarray] = None
        self.reid_last_raw: Optional[np.ndarray] = None  # cache object (to detect repeats)
        self.reid_last_emb_sec = 0.0
        self.reid_last_bank_sec = 0.0
        self.reid_last_eval_sec = 0.0
        self.reid_first_rgb_sec = 0.0
        self.reid_samples = 0
        self.reid_matched = False
        self.reid_restored_from: Optional[int] = None
        self.reid_name: Optional[str] = None

        # 2D CV EKF (Spencer srl_nearest_neighbor_tracker): [x,y,vx,vy], measurement [x,y].
        self.kf = ConstantVelocityEKF(
            map_x, map_y, process_noise_diag=(1e-2, 1e-2, 0.55, 0.55), measurement_noise=5e-2
        )

    def predict(self, dt_sec: float) -> Tuple[float, float]:
        pred = self.kf.predict(dt_sec)
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, map_x: float, map_y: float, measurement_noise_scale: float = 1.0) -> None:
        """Correct with a measurement. scale>1 = trust it less (laser-only with recent RGB)."""
        if measurement_noise_scale != 1.0:
            self.kf.measurementNoiseCov = self.kf._R_base * float(measurement_noise_scale)
        else:
            self.kf.measurementNoiseCov = self.kf._R_base.copy()
        self.kf.correct(map_x, map_y)

    def state(self) -> Tuple[float, float]:
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])

    def update_display(self, alpha: float, snap_m: float) -> None:
        """EMA of state() for drawing only. alpha=1 -> raw state; a jump > snap_m resets
        (re-association), so the box does not slide across the room."""
        x, y = self.state()
        if self.display_xy is None or alpha >= 1.0:
            self.display_xy = (x, y)
            return
        dx, dy = x - self.display_xy[0], y - self.display_xy[1]
        if snap_m > 0.0 and (dx * dx + dy * dy) > snap_m * snap_m:
            self.display_xy = (x, y)
            return
        self.display_xy = (self.display_xy[0] + alpha * dx, self.display_xy[1] + alpha * dy)

    def display_position(self) -> Tuple[float, float]:
        return self.display_xy if self.display_xy is not None else self.state()

    def association_position(self) -> Tuple[float, float]:
        """Position before correct(): statePre after predict() (Spencer / EKF)."""
        return float(self.kf.statePre[0, 0]), float(self.kf.statePre[1, 0])

    def mahalanobis_sq(self, meas_x: float, meas_y: float) -> float:
        """Spencer: squared Mahalanobis distance with the innovation covariance."""
        h = self.kf.measurementMatrix
        p = self.kf.errorCovPre
        r = self.kf.measurementNoiseCov
        s = h @ p @ h.T + r
        pred_x = float(self.kf.statePre[0, 0])
        pred_y = float(self.kf.statePre[1, 0])
        v = np.array([[meas_x - pred_x], [meas_y - pred_y]], dtype=np.float32)
        try:
            s_inv = np.linalg.inv(s)
            d = float(v.T @ s_inv @ v)
            return max(d, 0.0)
        except np.linalg.LinAlgError:
            return float("inf")

    def velocity(self) -> Tuple[float, float]:
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    def zero_velocity(self) -> None:
        self.kf.statePost[2, 0] = 0.0
        self.kf.statePost[3, 0] = 0.0
        self.kf.statePre[2, 0] = 0.0
        self.kf.statePre[3, 0] = 0.0

    def set_velocity(self, vx: float, vy: float) -> None:
        self.kf.statePost[2, 0] = float(vx)
        self.kf.statePost[3, 0] = float(vy)
        self.kf.statePre[2, 0] = float(vx)
        self.kf.statePre[3, 0] = float(vy)

    def occluded_update(self) -> None:
        self.kf.occluded_update()

    def append_pose_trail_sample(self) -> None:
        """One sample per cycle (Kalman state in map/tracking_frame)."""
        mx, my = self.state()
        self.pose_trail_ring.append((float(mx), float(my)))
        self.max_travel_from_spawn_m = max(
            self.max_travel_from_spawn_m,
            math.hypot(mx - self.spawn_xy[0], my - self.spawn_xy[1]),
        )

    def record_map_pose_after_correct(self, stamp_sec: float, history_keep_sec: float) -> None:
        mx, my = self.state()
        self.motion_pose_history.append((float(stamp_sec), float(mx), float(my)))
        cutoff = float(stamp_sec) - max(float(history_keep_sec), 0.5)
        while self.motion_pose_history and self.motion_pose_history[0][0] < cutoff:
            self.motion_pose_history.popleft()

    def map_displacement_span_m(self, stamp_sec: float, window_sec: float) -> float:
        """Max distance between sample pairs in [stamp_sec - window, stamp_sec].

        Uses the bounding-box diagonal (O(n)) instead of O(n²) pairs.
        """
        cutoff = float(stamp_sec) - max(float(window_sec), 1e-3)
        xs: list[float] = []
        ys: list[float] = []
        for t, x, y in self.motion_pose_history:
            if t >= cutoff:
                xs.append(x)
                ys.append(y)
        if len(xs) < 2:
            return 0.0
        return float(math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
