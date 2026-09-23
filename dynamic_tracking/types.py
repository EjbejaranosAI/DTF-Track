"""Tipos y estado Kalman compartidos por el nodo de tracking."""

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
    # Hit del /scan en la ventana del bbox (misma convencion que scan.angle_min + i*increment);
    # permite matching sin TF a mapa (fase polar en apply_yolo_auxiliary).
    laser_range_m: Optional[float] = None
    laser_bearing_rad: Optional[float] = None


@dataclass
class LidarMeasurement:
    map_x: float
    map_y: float
    # 0..1 geometric leg-like score (cluster span, range flatness, chord width)
    leg_candidate_score: float = 0.0
    # True si el segmento pasaria fusion spawn (p. ej. require_two); auditoria periodica del track.
    spawn_tier_pass: bool = False


@dataclass
class LidarDualExtract:
    """Strict clusters for spawn + lax clusters for track maintenance (one /scan pass)."""

    spawn: List[LidarMeasurement]
    maintenance: List[LidarMeasurement]
    n_segments: int = 0


@dataclass
class PendingLidarSpawn:
    """Acumula detecciones laser-only person en ~el mismo sitio (mapa) antes de crear Track."""

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
        # Ciclos seguidos sin cluster ni scan_verify (solo FOV); debounce antes de missed++.
        self.consecutive_no_lidar_presence = 0
        self.hits = 1
        self.prev_speed_mps = 0.0
        # LiDAR-only person hypothesis from leg geometry (cleared when camera updates)
        self.leg_hypothesis_active = False
        self.leg_hypothesis_score = 0.0
        # Ultima vez que este track recibio YOLO clase person (0 = nunca; refuerzo en RViz).
        self.last_rgb_person_sec = 0.0
        # FOV camara (proyeccion track->imagen).
        self.last_camera_fov_sec = 0.0
        self.last_outside_camera_fov_sec = 0.0
        # Ciclos seguidos en FOV sin match YOLO person (rechazo negativo).
        self.yolo_in_fov_reject_streak = 0
        # Ultimo correct() desde cluster LiDAR (0 = nunca).
        self.last_lidar_correct_sec = 0.0
        self.last_lidar_map_x = float(map_x)
        self.last_lidar_map_y = float(map_y)
        # Stamp de imagen en que last_bbox se actualizo con deteccion real (evita bbox fantasma en /image_annotated).
        self.last_image_bbox_stamp_sec = float(stamp_sec) if bbox is not None else 0.0
        # Muestras (stamp, map_x, map_y) tras correct() en mapa — desplazamiento real vs FP estatico.
        self.motion_pose_history: Deque[Tuple[float, float, float]] = deque(maxlen=48)
        # Anillo de posiciones (x,y) tras cada ciclo: duplicados temporales (Spencer) + trayectoria RViz.
        self.pose_trail_ring: Deque[Tuple[float, float]] = deque(maxlen=64)
        self.motion_dynamic_confirmed = False
        # Spencer occlusion: separate counter for out-of-FOV cycles.
        self.consecutive_occlusions = 0

        # EKF 2D CV (Spencer srl_nearest_neighbor_tracker): [x,y,vx,vy], medicion [x,y].
        self.kf = ConstantVelocityEKF(
            map_x, map_y, process_noise_diag=(1e-2, 1e-2, 0.55, 0.55), measurement_noise=5e-2
        )

    def predict(self, dt_sec: float) -> Tuple[float, float]:
        pred = self.kf.predict(dt_sec)
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, map_x: float, map_y: float, measurement_noise_scale: float = 1.0) -> None:
        """Corrige con medicion. scale>1 = confiar menos (solo-laser con RGB reciente)."""
        if measurement_noise_scale != 1.0:
            self.kf.measurementNoiseCov = self.kf._R_base * float(measurement_noise_scale)
        else:
            self.kf.measurementNoiseCov = self.kf._R_base.copy()
        self.kf.correct(map_x, map_y)

    def state(self) -> Tuple[float, float]:
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])

    def association_position(self) -> Tuple[float, float]:
        """Posicion previa a correct(): statePre tras predict() (Spencer / EKF)."""
        return float(self.kf.statePre[0, 0]), float(self.kf.statePre[1, 0])

    def mahalanobis_sq(self, meas_x: float, meas_y: float) -> float:
        """Spencer: distancia Mahalanobis al cuadrado con covarianza de innovacion."""
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
        """Una muestra por ciclo (estado Kalman en mapa/tracking_frame)."""
        mx, my = self.state()
        self.pose_trail_ring.append((float(mx), float(my)))

    def record_map_pose_after_correct(self, stamp_sec: float, history_keep_sec: float) -> None:
        mx, my = self.state()
        self.motion_pose_history.append((float(stamp_sec), float(mx), float(my)))
        cutoff = float(stamp_sec) - max(float(history_keep_sec), 0.5)
        while self.motion_pose_history and self.motion_pose_history[0][0] < cutoff:
            self.motion_pose_history.popleft()

    def map_displacement_span_m(self, stamp_sec: float, window_sec: float) -> float:
        """Max distancia entre pares de muestras en [stamp_sec - window, stamp_sec].

        Usa bounding-box diagonal (O(n)) en vez de pares O(n²).
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
