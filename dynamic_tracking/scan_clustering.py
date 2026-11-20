"""LaserScan -> clusters en frame de tracking (Spencer parity).

Alineado con old_spencer/spencer_people_tracking:
  - Segmentacion jump-distance (srl_laser_segmentation).
  - Filtros: min/max puntos por segmento, distancia media al sensor, anchura minima opcional.
  - Pose publicada: mediana cartesiana x,y por separado (use_median_for_pose en srl_laser_detectors).
  - Clasificador RF opcional (random_forest_detector.cpp): solo FOREGROUND si spencer_rf_only_positive.

La fusion con mapa (costmap, /map estatico) y el tracking Kalman estan en dynamic_tracking_node / track_pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
from sensor_msgs.msg import LaserScan

from .types import LidarMeasurement

if TYPE_CHECKING:
    from .spencer_detector import SpencerLaserDetector

TransformXY = Callable[[float, float, str, object], Optional[Tuple[float, float]]]


# Debug stats live in lidar_detection (multi-detector pipeline).
from .lidar_detection.types import LidarExtractDebugStats  # noqa: F401


def _cluster_points_to_xy(cluster: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(angle_rad, range_m) -> (x,y) en frame del laser (mismo convenio que el resto del nodo)."""
    return [(r * math.cos(a), r * math.sin(a)) for a, r in cluster]


def cluster_pose_laser_xy(
    cluster: List[Tuple[float, float]],
    pose_mode: str,
) -> Tuple[float, float, float]:
    """Pose 2D del cluster en frame laser + distancia al origen.

    - median: mediana de x y de y por separado (srl_laser_detectors, use_median_for_pose).
    - closest: punto del haz con rango minimo (comportamiento antiguo del paquete).
    """
    if not cluster:
        return 0.0, 0.0, 0.0
    mode = str(pose_mode).strip().lower()
    if mode == "closest":
        ang, r = min(cluster, key=lambda c: c[1])
        x = r * math.cos(ang)
        y = r * math.sin(ang)
        return x, y, float(r)
    xs = sorted(r * math.cos(a) for a, r in cluster)
    ys = sorted(r * math.sin(a) for a, r in cluster)
    n = len(xs)
    mid = n // 2
    if n % 2:
        mx, my = xs[mid], ys[mid]
    else:
        mx = 0.5 * (xs[mid - 1] + xs[mid])
        my = 0.5 * (ys[mid - 1] + ys[mid])
    return mx, my, math.hypot(mx, my)


def spencer_style_segment_prefilter(
    cluster: List[Tuple[float, float]],
    min_points: int,
    max_points: int,
    min_mean_range_m: float,
    max_mean_range_m: float,
    min_segment_width_m: float,
) -> bool:
    """Replica filtros de srl_laser_segmentation/ros_interface (min/max puntos, distancia media, anchura)."""
    n = len(cluster)
    if n < min_points:
        return False
    if max_points > 0 and n > max_points:
        return False
    xy = _cluster_points_to_xy(cluster)
    mx = sum(p[0] for p in xy) / n
    my = sum(p[1] for p in xy) / n
    d_mean = math.hypot(mx, my)
    if d_mean < min_mean_range_m or d_mean > max_mean_range_m:
        return False
    if min_segment_width_m > 1e-6:
        x0, y0 = xy[0]
        x1, y1 = xy[-1]
        w = math.hypot(x1 - x0, y1 - y0)
        if w < min_segment_width_m:
            return False
    return True


@dataclass
class LegHeuristicParams:
    """Heuristica 2D tipo piernas (sin ML): cluster angular + cuerda + rango plano."""

    enabled: bool = True
    min_span_deg: float = 2.5
    max_span_deg: float = 26.0
    max_range_spread_ratio: float = 0.16
    min_chord_m: float = 0.06
    max_chord_m: float = 0.55


def _trapezoid_01(x: float, a: float, b: float, c: float, d: float) -> float:
    """1 en [b,c], rampas en [a,b] y [c,d]; fuera de [a,d] es 0."""
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / max(b - a, 1e-9)
    if x <= c:
        return 1.0
    return (d - x) / max(d - c, 1e-9)


def _bimodal_range_boost(cluster: List[Tuple[float, float]]) -> float:
    """Dos grupos de rango similares separados por un salto (dos piernas)."""
    if len(cluster) < 5:
        return 0.0
    rs = sorted(c[1] for c in cluster)
    best_gap = 0.0
    best_i = -1
    for i in range(len(rs) - 1):
        g = rs[i + 1] - rs[i]
        if g > best_gap:
            best_gap = g
            best_i = i
    if best_gap < 0.11:
        return 0.0
    left = rs[: best_i + 1]
    right = rs[best_i + 1 :]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    m1 = sum(left) / len(left)
    m2 = sum(right) / len(right)
    mr = min(m1, m2)
    if mr < 1e-3:
        return 0.0
    if abs(m1 - m2) > 0.22 * mr:
        return 0.0
    return min(0.22, 0.08 + best_gap * 0.35)


def _linearity_residual(cluster: List[Tuple[float, float]]) -> float:
    """Residual normalizado de ajuste lineal LS en coordenadas cartesianas.
    Bajo = lineal (pared); alto = curvado/circular (pierna).
    Spencer Feature10."""
    n = len(cluster)
    if n < 3:
        return 1.0
    xs = [r * math.cos(a) for a, r in cluster]
    ys = [r * math.sin(a) for a, r in cluster]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12:
        return 1.0
    m = sxy / sxx
    residual = sum((y - my - m * (x - mx)) ** 2 for x, y in zip(xs, ys)) / n
    extent = sxx + syy
    if extent < 1e-12:
        return 1.0
    return residual / extent


def _compactness(cluster: List[Tuple[float, float]]) -> float:
    """Ratio de area/perimetro; alto = compacto/circular (pierna).
    Inspirado en Spencer Feature06 (circle fit) y Feature09 (aspect ratio)."""
    n = len(cluster)
    if n < 3:
        return 0.0
    xs = [r * math.cos(a) for a, r in cluster]
    ys = [r * math.sin(a) for a, r in cluster]
    mx = sum(xs) / n
    my = sum(ys) / n
    std_x = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    std_y = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if max(std_x, std_y) < 1e-6:
        return 0.0
    return (1.0 + min(std_x, std_y)) / (1.0 + max(std_x, std_y))


def compute_leg_candidate_score(
    cluster: List[Tuple[float, float]],
    params: LegHeuristicParams,
) -> float:
    """Score multi-feature inspirado en Spencer (RF/AdaBoost sobre ~50 features).

    Features usadas:
      1. Span angular + chord (original)
      2. Range spread ratio (original)
      3. Point count (Spencer F01) — piernas: 3-20 puntos
      4. Linearity (Spencer F10) — paredes lineales, piernas curvadas
      5. Compactness (Spencer F06/F09) — piernas son compactas/circulares
      6. Width (Spencer F05) — piernas: 0.06-0.55m
      7. Mean consecutive range diff (Spencer F02)
      8. Bimodal range boost (dos piernas)
    """
    if not params.enabled or len(cluster) < 2:
        return 0.0
    n = len(cluster)
    angles = [c[0] for c in cluster]
    ranges = [c[1] for c in cluster]
    span = max(angles) - min(angles)
    span_deg = math.degrees(span)
    mean_r = sum(ranges) / n
    spread = max(ranges) - min(ranges)
    if mean_r < 1e-6:
        return 0.0
    chord = 2.0 * mean_r * math.sin(0.5 * span)

    # F1: span angular
    span_ok = _trapezoid_01(
        span_deg,
        max(0.5, params.min_span_deg - 1.5),
        params.min_span_deg,
        params.max_span_deg,
        params.max_span_deg + 6.0,
    )
    # F2: range spread ratio
    ratio = spread / mean_r
    thr = params.max_range_spread_ratio
    flat_ok = 1.0 if ratio <= thr else max(0.0, 1.0 - (ratio - thr) / max(0.07, thr * 1.2))

    # F3: chord width
    chord_ok = _trapezoid_01(
        chord,
        max(0.03, params.min_chord_m - 0.04),
        params.min_chord_m,
        params.max_chord_m,
        params.max_chord_m + 0.12,
    )

    # F4: point count (Spencer F01) — piernas: 3-20 puntos a distancias típicas
    expected_pts = max(3, min(25, 12.0 / max(mean_r, 0.5)))
    npts_ok = _trapezoid_01(float(n), 2.0, 3.0, expected_pts * 1.5, expected_pts * 3.0)

    # F5: linearity (Spencer F10) — paredes tienen residual bajo, piernas alto
    lin_res = _linearity_residual(cluster)
    linearity_ok = min(1.0, lin_res * 8.0)

    # F6: compactness (Spencer F06/F09) — piernas son compactas
    compact = _compactness(cluster)
    compact_ok = _trapezoid_01(compact, 0.15, 0.3, 1.0, 1.01)

    # F7: mean consecutive range diff (Spencer F02)
    if n >= 3:
        diffs = [abs(ranges[i + 1] - ranges[i]) for i in range(n - 1)]
        mean_diff = sum(diffs) / len(diffs)
        diff_ok = _trapezoid_01(mean_diff, 0.0, 0.001, 0.04, 0.10)
    else:
        diff_ok = 0.5

    # Weighted combination (pseudo-AdaBoost: pesos manuales por importancia)
    score = (
        0.20 * span_ok
        + 0.10 * flat_ok
        + 0.15 * chord_ok
        + 0.10 * npts_ok
        + 0.20 * linearity_ok
        + 0.15 * compact_ok
        + 0.10 * diff_ok
    )
    boost = _bimodal_range_boost(cluster)
    score = min(1.0, score + boost)

    # Penalizaciones fuertes (veto)
    if span_deg > 16.0 and boost < 0.06:
        score = min(score, 0.40)
    beams_per_rad = (n / max(span, 0.02)) if span > 0 else 0.0
    if span_deg > 10.0 and beams_per_rad > 120.0:
        score *= 0.30
    # Muchos puntos + muy lineal = pared
    if n > 15 and lin_res < 0.01:
        score *= 0.25
    # Muy lejos: penalizacion suave (6m ya no castiga; piernas a 3–5m suelen quedar >0.55).
    if mean_r > 10.0:
        score *= 0.45
    elif mean_r > 8.0:
        score *= 0.78

    return float(max(0.0, min(1.0, score)))


def is_track_in_lidar_fov(
    scan: LaserScan,
    track_map_x: float,
    track_map_y: float,
    tf_tx: float,
    tf_ty: float,
    tf_yaw: float,
    fov_margin_deg: float = 5.0,
) -> bool:
    """Check if a track position falls within the LiDAR's angular FOV.

    Returns False if the person is behind the robot (outside sensor cone).
    Used for Spencer-style occlusion: tracks outside FOV are OCCLUDED,
    not MISSED, and should not be penalized as harshly.
    """
    n = len(scan.ranges)
    if n == 0:
        return False
    cos_y = math.cos(tf_yaw)
    sin_y = math.sin(tf_yaw)
    dx = track_map_x - tf_tx
    dy = track_map_y - tf_ty
    x_laser = cos_y * dx + sin_y * dy
    y_laser = -sin_y * dx + cos_y * dy
    angle = math.atan2(y_laser, x_laser)
    margin = math.radians(fov_margin_deg)
    fov_min = scan.angle_min + margin
    fov_max = scan.angle_min + (n - 1) * scan.angle_increment - margin
    return fov_min <= angle <= fov_max


def _shortest_angle_diff_rad(a: float, b: float) -> float:
    """Diferencia a-b en (-pi, pi] (robusto a saltos del scan)."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return d


def _angle_to_index_range(
    scan: LaserScan, center_angle: float, half_window: float
) -> Tuple[int, int]:
    """Convert angular window to scan index range (clamped). O(1)."""
    n = len(scan.ranges)
    inc = scan.angle_increment
    if inc == 0.0 or n == 0:
        return (0, 0)
    i_center = (center_angle - scan.angle_min) / inc
    i_half = half_window / abs(inc)
    lo = max(0, int(math.floor(i_center - i_half)))
    hi = min(n, int(math.ceil(i_center + i_half)) + 1)
    return (lo, hi)


def scan_window_best_range(
    scan: LaserScan, lidar_window_rad: float, center_angle: float
) -> Optional[Tuple[float, float]]:
    """Menor rango en ventana angular alrededor de center_angle (index-based)."""
    if scan.angle_increment == 0.0 or not scan.ranges:
        return None
    lo, hi = _angle_to_index_range(scan, center_angle, lidar_window_rad * 0.5)
    best_r: Optional[float] = None
    best_angle: Optional[float] = None
    rmin, rmax = scan.range_min, scan.range_max
    amin, ainc = scan.angle_min, scan.angle_increment
    ranges = scan.ranges
    for i in range(lo, hi):
        r = ranges[i]
        if not math.isfinite(r) or r < rmin or r > rmax:
            continue
        if best_r is None or r < best_r:
            best_r = float(r)
            best_angle = float(amin + i * ainc)
    if best_r is None or best_angle is None:
        return None
    return best_r, best_angle


def verify_track_in_scan(
    scan: LaserScan,
    track_map_x: float,
    track_map_y: float,
    tf_tx: float,
    tf_ty: float,
    tf_yaw: float,
    range_tolerance_m: float = 0.50,
    window_deg: float = 15.0,
    min_range_m: float = 0.25,
    max_range_m: float = 8.0,
) -> bool:
    """Verifica si el scan tiene un retorno donde esperamos ver un track.

    O(window) en vez de O(N) gracias al cálculo de índice directo.
    """
    cos_y = math.cos(tf_yaw)
    sin_y = math.sin(tf_yaw)
    dx = track_map_x - tf_tx
    dy = track_map_y - tf_ty
    x_laser = cos_y * dx + sin_y * dy
    y_laser = -sin_y * dx + cos_y * dy
    expected_range = math.sqrt(x_laser * x_laser + y_laser * y_laser)
    if expected_range < min_range_m or expected_range > max_range_m:
        return False
    expected_angle = math.atan2(y_laser, x_laser)
    half_window = math.radians(window_deg) * 0.5
    lo, hi = _angle_to_index_range(scan, expected_angle, half_window)
    best_diff = float("inf")
    rmin, rmax = scan.range_min, scan.range_max
    ranges = scan.ranges
    for i in range(lo, hi):
        r = ranges[i]
        if not math.isfinite(r) or r < rmin or r > rmax:
            continue
        diff = abs(r - expected_range)
        if diff < best_diff:
            best_diff = diff
    return best_diff <= range_tolerance_m


def _cluster_to_measurement(
    cluster: List[Tuple[float, float]],
    scan: LaserScan,
    min_track_distance_m: float,
    max_track_distance_m: float,
    scan_frame_id: str,
    transform_xy_to_map: TransformXY,
    leg_params: LegHeuristicParams,
    pose_mode: str = "median",
) -> Optional[LidarMeasurement]:
    x_local, y_local, d_pose = cluster_pose_laser_xy(cluster, pose_mode)
    if d_pose < min_track_distance_m or d_pose > max_track_distance_m:
        return None
    transformed = transform_xy_to_map(x_local, y_local, scan_frame_id, scan.header.stamp)
    if transformed is None:
        return None
    leg_score = compute_leg_candidate_score(cluster, leg_params)
    return LidarMeasurement(
        map_x=float(transformed[0]),
        map_y=float(transformed[1]),
        leg_candidate_score=leg_score,
    )


def repair_interleaved_returns(
    scan: LaserScan,
    half_window: int,
    max_range_m: float,
    jump_m: float,
) -> LaserScan:
    """Fill background beams interleaved inside a close object (TIAGo virtual laser).

    Beside the robot the merged /scan alternates beams between a person (~0.4 m) and
    what lies behind (0.75 m, 6 m, NaN), so jump segmentation shreds the person into
    1-3 point pieces. A beam is filled when there is a closer return on BOTH sides
    within half_window beams (morphological closing, so edges and isolated points do
    not grow), that return is < max_range_m and closer by more than jump_m.
    Far objects are untouched. Returns a new LaserScan (input is not modified).
    """
    if half_window <= 0 or not scan.ranges:
        return scan
    r = np.asarray(scan.ranges, dtype=np.float64)
    valid = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max)
    rv = np.where(valid, r, np.inf)
    n = rv.size
    padded = np.concatenate([np.full(half_window, np.inf), rv, np.full(half_window, np.inf)])
    left = np.min(np.stack([padded[k:k + n] for k in range(half_window)]), axis=0)
    right = np.min(
        np.stack([padded[k:k + n] for k in range(half_window + 1, 2 * half_window + 1)]), axis=0
    )
    near = np.maximum(left, right)
    with np.errstate(invalid="ignore"):
        fill = (near < max_range_m) & (rv - near > jump_m)
    if not fill.any():
        return scan
    out = LaserScan()
    out.header = scan.header
    out.angle_min, out.angle_max = scan.angle_min, scan.angle_max
    out.angle_increment, out.time_increment = scan.angle_increment, scan.time_increment
    out.scan_time = scan.scan_time
    out.range_min, out.range_max = scan.range_min, scan.range_max
    out.ranges = np.where(fill, near, r).astype(np.float32).tolist()
    out.intensities = scan.intensities
    return out


def _segment_scan(
    scan: LaserScan,
    lidar_cluster_jump_m: float,
) -> List[List[int]]:
    """Jump-distance segmentation (Spencer-style). Returns list of index-lists."""
    segments: List[List[int]] = []
    current: List[int] = []
    prev_range: Optional[float] = None
    for i, value in enumerate(scan.ranges):
        if not math.isfinite(value) or value < scan.range_min or value > scan.range_max:
            if current:
                segments.append(current)
                current = []
            prev_range = None
            continue
        if prev_range is not None and abs(value - prev_range) > lidar_cluster_jump_m:
            if current:
                segments.append(current)
                current = []
        current.append(i)
        prev_range = value
    if current:
        segments.append(current)
    return segments


