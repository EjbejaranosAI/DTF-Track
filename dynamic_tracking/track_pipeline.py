"""Actualizacion de tracks (Kalman + asociacion YOLO / LiDAR) y export a Detection2DArray.

Orden por ciclo (ver docs/PIPELINE.md):
  Modo `yolo_auxiliary_person_only` (cascada tipo Spencer: laser primero):
  1) predict(dt); 2) LiDAR como si YOLO estuviera apagado; 3) poda/dedup; 4) YOLO solo confirma
     person en tracks existentes (color/bbox imagen), sin correct() ni spawn desde camara.

  Modo legacy (YOLO en el tracker):
  1) predict(dt); 2) asociacion YOLO + correct opcional; 3) IoU imagen si aplica;
  4) LiDAR; 5) missed_frames, poda, flags dinamicos
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .association import assign_measurements, assoc_class_key, measurement_cost
from .identity import ReIdConfig, add_track_appearance

try:
    from scipy.optimize import linear_sum_assignment as _lsa_person_iou
    _HAS_SCIPY_PERSON_IOU = True
except Exception:
    _lsa_person_iou = None
    _HAS_SCIPY_PERSON_IOU = False

from .scan_clustering import is_track_in_lidar_fov, verify_track_in_scan
from .types import LidarMeasurement, PendingLidarSpawn, Track, YoloMeasurement

_BIG_IOU_COST = 1e6


def _nearest_leg_cluster_multi_anchor(
    anchors: List[Tuple[float, float]],
    lidar_meas: List[LidarMeasurement],
    used_mi: Set[int],
    min_leg_score: float,
    max_dist_m: float,
) -> Tuple[Optional[int], float]:
    best_mi: Optional[int] = None
    best_d = float(max_dist_m)
    for ax, ay in anchors:
        mi, d = _nearest_leg_cluster_index(
            ax, ay, lidar_meas, used_mi, min_leg_score, max_dist_m
        )
        if mi is not None and d < best_d:
            best_mi, best_d = mi, d
    if best_mi is None:
        return None, float(max_dist_m)
    return best_mi, best_d


def _nearest_leg_cluster_index(
    map_x: float,
    map_y: float,
    lidar_meas: List[LidarMeasurement],
    used_mi: Set[int],
    min_leg_score: float,
    max_dist_m: float,
) -> Tuple[Optional[int], float]:
    best_mi: Optional[int] = None
    best_d = float(max_dist_m)
    for mi, lm in enumerate(lidar_meas):
        if mi in used_mi:
            continue
        if float(lm.leg_candidate_score) < float(min_leg_score):
            continue
        d = measurement_cost(map_x, map_y, lm.map_x, lm.map_y)
        if d < best_d:
            best_d = d
            best_mi = mi
    if best_mi is None:
        return None, float(max_dist_m)
    return best_mi, best_d


def _record_person_lidar_position(tr: Track, map_x: float, map_y: float, stamp_sec: float) -> None:
    tr.last_lidar_correct_sec = float(stamp_sec)
    tr.last_lidar_map_x = float(map_x)
    tr.last_lidar_map_y = float(map_y)


def _person_lidar_micro_jitter(
    tr: Track, map_x: float, map_y: float, threshold_m: float
) -> bool:
    """True si la medicion casi no se movio respecto al ultimo correct() laser."""
    if threshold_m <= 0.0 or tr.class_name != "person":
        return False
    if float(getattr(tr, "last_lidar_correct_sec", 0.0)) <= 0.0:
        return False
    lx = float(getattr(tr, "last_lidar_map_x", tr.state()[0]))
    ly = float(getattr(tr, "last_lidar_map_y", tr.state()[1]))
    return measurement_cost(lx, ly, map_x, map_y) < threshold_m


def _other_person_track_closer(
    tracks: List[Track],
    person_tis: List[int],
    exclude_ti: int,
    lm: LidarMeasurement,
    my_dist_m: float,
) -> bool:
    for oti in person_tis:
        if oti == exclude_ti:
            continue
        ox, oy = tracks[oti].association_position()
        if measurement_cost(ox, oy, lm.map_x, lm.map_y) + 0.12 < my_dist_m:
            return True
    return False


def _person_has_nearby_leg_cluster(
    tr: Track,
    lidar_meas: List[LidarMeasurement],
    radius_m: float,
    min_leg_score: float,
) -> bool:
    if radius_m <= 0.0:
        return False
    mx, my = tr.state()
    for lm in lidar_meas:
        if float(lm.leg_candidate_score) < float(min_leg_score):
            continue
        if measurement_cost(mx, my, lm.map_x, lm.map_y) < radius_m:
            return True
    return False


def _person_track_dual_audit_pass(
    tr: Track,
    stamp_sec: float,
    lidar_meas: List[LidarMeasurement],
    *,
    audit_radius_m: float,
    audit_min_leg_score: float,
    yolo_seen_sec: float,
    corrected_spawn_tier: bool,
) -> bool:
    """Track existente: YOLO reciente o cluster maintenance cerca (1 detector basta; no exige spawn-tier)."""
    if corrected_spawn_tier:
        return True
    lr = float(getattr(tr, "last_rgb_person_sec", 0.0))
    if lr > 0.0 and yolo_seen_sec > 0.0 and (stamp_sec - lr) <= yolo_seen_sec:
        return True
    return _person_has_nearby_leg_cluster(
        tr, lidar_meas, audit_radius_m, audit_min_leg_score
    )


def _spawn_initiator_accepts_step(
    pending: PendingLidarSpawn,
    map_x: float,
    map_y: float,
    stamp_sec: float,
    max_step_m: float,
    max_vel_mps: float,
) -> bool:
    """Spencer logic_initiator: rechaza saltos imposibles entre scans consecutivos del mismo pending."""
    if pending.streak <= 0 or not pending.touched_this_cycle:
        return True
    dt = max(stamp_sec - pending.last_stamp_sec, 1e-4)
    dist = math.hypot(map_x - pending.map_x, map_y - pending.map_y)
    if max_step_m > 0.0 and dist > max_step_m:
        return False
    if max_vel_mps > 0.0 and (dist / dt) > max_vel_mps:
        return False
    return True


def _assign_person_yolo_iou_pairs(
    tracks: List[Track],
    track_indices: List[int],
    yolo_meas: List[YoloMeasurement],
    candidate_ym_indices: List[int],
    iou_min: float,
) -> List[Tuple[int, int]]:
    """Empareja tracks person vs detecciones YOLO person por IoU en imagen (Hungarian).

    Solo usa bbox imagen + clase person; ignora map_xy para el matching (evita permutas al girar).
    Devuelve pares (track_index, indice en yolo_meas).
    """
    person_tis = [ti for ti in track_indices if str(tracks[ti].class_name).lower() == "person"]
    person_ym_idx = [
        j
        for j in candidate_ym_indices
        if str(yolo_meas[j].class_name).lower() == "person" and yolo_meas[j].map_xy is not None
    ]
    if not person_tis or not person_ym_idx:
        return []
    n, m = len(person_tis), len(person_ym_idx)
    costs = np.full((n, m), _BIG_IOU_COST, dtype=np.float64)
    for ri, ti in enumerate(person_tis):
        bx = tracks[ti].last_bbox
        if bx is None:
            continue
        for cj, ym_idx in enumerate(person_ym_idx):
            iou = _bbox_iou_xyxy(bx, yolo_meas[ym_idx].bbox)
            if iou >= iou_min:
                costs[ri, cj] = 1.0 - iou
    if _HAS_SCIPY_PERSON_IOU:
        row_idx, col_idx = _lsa_person_iou(costs)
    else:
        row_idx, col_idx = _greedy_iou_assignment(costs)
    out: List[Tuple[int, int]] = []
    used_t: Set[int] = set()
    used_y: Set[int] = set()
    for r, c in zip(row_idx, col_idx):
        if costs[r, c] >= _BIG_IOU_COST - 1.0:
            continue
        ti = person_tis[r]
        ym_idx = person_ym_idx[c]
        if ti in used_t or ym_idx in used_y:
            continue
        used_t.add(ti)
        used_y.add(ym_idx)
        out.append((ti, ym_idx))
    return out


def _greedy_iou_assignment(costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = costs.shape
    pairs: List[Tuple[float, int, int]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if costs[r, c] < _BIG_IOU_COST - 1.0:
                pairs.append((costs[r, c], r, c))
    pairs.sort(key=lambda x: x[0])
    used_r: Set[int] = set()
    used_c: Set[int] = set()
    row_out: List[int] = []
    col_out: List[int] = []
    for _, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        row_out.append(r)
        col_out.append(c)
    return np.array(row_out, dtype=int), np.array(col_out, dtype=int)


def _bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def _image_only_soft_associate(
    tracks: List[Track],
    unmatched_track_indices: List[int],
    yolo_meas: List[YoloMeasurement],
    stamp_sec: float,
    min_iou: float,
) -> List[int]:
    """Si YOLO no tiene map_xy (TF/rayo fallo al mover el base), mantiene ID por IoU en imagen.
    No hace Kalman correct (no hay pose map); evita missed_frames y actualiza bbox."""
    mapless_idx = [i for i, ym in enumerate(yolo_meas) if ym.map_xy is None]
    if not mapless_idx or not unmatched_track_indices:
        return unmatched_track_indices

    pairs: List[Tuple[float, int, int]] = []
    for ti in unmatched_track_indices:
        tr = tracks[ti]
        if tr.last_bbox is None:
            continue
        for yi in mapless_idx:
            ym = yolo_meas[yi]
            if assoc_class_key(tr.class_name) != assoc_class_key(ym.class_name):
                continue
            iou = _bbox_iou_xyxy(tr.last_bbox, ym.bbox)
            if iou >= min_iou:
                pairs.append((iou, ti, yi))
    pairs.sort(key=lambda x: -x[0])

    used_t: Set[int] = set()
    used_y: Set[int] = set()
    for _, ti, yi in pairs:
        if ti in used_t or yi in used_y:
            continue
        used_t.add(ti)
        used_y.add(yi)
        tr = tracks[ti]
        ym = yolo_meas[yi]
        tr.last_bbox = ym.bbox
        tr.last_image_bbox_stamp_sec = float(stamp_sec)
        tr.last_seen_sec = stamp_sec
        tr.missed_frames = 0
        tr.consecutive_occlusions = 0
        tr.hits += 1
        tr.score = ym.score
        if str(ym.class_name).lower() == "person":
            tr.last_rgb_person_sec = float(stamp_sec)

    return [ti for ti in unmatched_track_indices if ti not in used_t]


def _axis_aligned_square_iou_map(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    side_m: float,
) -> float:
    """IoU en el plano mapa de dos cuadrados iguales, ejes alineados a x/y (como caja persona en RViz)."""
    if side_m <= 1e-9:
        return 0.0
    h = side_m * 0.5
    a_x1, a_x2 = ax - h, ax + h
    a_y1, a_y2 = ay - h, ay + h
    b_x1, b_x2 = bx - h, bx + h
    b_y1, b_y2 = by - h, by + h
    ix1 = max(a_x1, b_x1)
    iy1 = max(a_y1, b_y1)
    ix2 = min(a_x2, b_x2)
    iy2 = min(a_y2, b_y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    s2 = side_m * side_m
    union = s2 + s2 - inter
    if union <= 1e-12:
        return 0.0
    return float(inter / union)


def _dedup_overlapping_person_tracks_map(
    tracks: List[Track],
    person_box_side_m: float,
    merge_min_iou: float,
    lidar_only: bool,
) -> None:
    """NMS tipo IoU en mapa: un track `person` por solape fuerte (varios segmentos/spawns -> mismo punto)."""
    if merge_min_iou <= 0.0 or len(tracks) < 2:
        return
    candidates = [
        t
        for t in tracks
        if t.class_name == "person"
        and (not lidar_only or t.last_rgb_person_sec <= 0.0)
    ]
    if len(candidates) < 2:
        return
    def _merge_rank(t: Track) -> tuple:
        return (
            -1 if float(getattr(t, "last_rgb_person_sec", 0.0)) > 0.0 else 0,
            -t.hits,
            -float(getattr(t, "leg_hypothesis_score", 0.0)),
            t.track_id,
        )

    candidates.sort(key=_merge_rank)
    kept: List[Track] = []
    drop_ids: Set[int] = set()
    for tr in candidates:
        mx, my = tr.state()
        is_dup = False
        for kt in kept:
            kx, ky = kt.state()
            if (
                _axis_aligned_square_iou_map(mx, my, kx, ky, person_box_side_m)
                >= merge_min_iou
            ):
                is_dup = True
                break
        if is_dup:
            drop_ids.add(tr.track_id)
        else:
            kept.append(tr)
    if drop_ids:
        tracks[:] = [t for t in tracks if t.track_id not in drop_ids]


def _delete_temporal_duplicate_person_tracks(
    tracks: List[Track],
    *,
    enable: bool,
    compare_n: int,
    must_match: int,
    max_dist_m: float,
    lidar_only: bool,
) -> None:
    """Spencer NNH deleteDuplicateTracks: dos person muy cercanas en los ultimos N ciclos -> una sola ID."""
    if not enable or len(tracks) < 2 or compare_n < 2 or must_match < 1 or max_dist_m <= 0.0:
        return
    max_d2 = float(max_dist_m) * float(max_dist_m)
    indices = [
        i
        for i, t in enumerate(tracks)
        if t.class_name == "person" and (not lidar_only or t.last_rgb_person_sec <= 0.0)
    ]
    if len(indices) < 2:
        return
    drop_ids: Set[int] = set()
    for a in range(len(indices)):
        ia = indices[a]
        ta = tracks[ia]
        if ta.track_id in drop_ids:
            continue
        ring_a = getattr(ta, "pose_trail_ring", None)
        if ring_a is None or len(ring_a) < compare_n:
            continue
        for b in range(a + 1, len(indices)):
            ib = indices[b]
            tb = tracks[ib]
            if tb.track_id in drop_ids:
                continue
            ring_b = getattr(tb, "pose_trail_ring", None)
            if ring_b is None or len(ring_b) < compare_n:
                continue
            matches = 0
            for k in range(compare_n):
                x1, y1 = ring_a[-1 - k]
                x2, y2 = ring_b[-1 - k]
                dx = x1 - x2
                dy = y1 - y2
                if dx * dx + dy * dy <= max_d2:
                    matches += 1
            if matches >= must_match:
                ra = float(getattr(ta, "last_rgb_person_sec", 0.0)) > 0.0
                rb = float(getattr(tb, "last_rgb_person_sec", 0.0)) > 0.0
                if ra != rb:
                    drop_ids.add(ta.track_id if rb else tb.track_id)
                    break
                ha, hb = ta.hits, tb.hits
                if ha < hb or (ha == hb and ta.track_id > tb.track_id):
                    drop_ids.add(ta.track_id)
                    break
                drop_ids.add(tb.track_id)
    if drop_ids:
        tracks[:] = [t for t in tracks if t.track_id not in drop_ids]


def _lidar_person_hypothesis_from_leg(
    lm: LidarMeasurement,
    leg_hypothesis_min_score: float,
    leg_hypothesis_as_person: bool,
) -> Tuple[str, float, bool]:
    """class_name, score, leg_hypothesis_active."""
    if lm.leg_candidate_score >= leg_hypothesis_min_score:
        sc = min(0.42, 0.10 + 0.52 * lm.leg_candidate_score)
        if leg_hypothesis_as_person:
            return "person", sc, True
        return "unknown", sc, True
    return "unknown", 0.2, False


def _wrapped_abs_angle_diff_rad(a: float, b: float) -> float:
    d = a - b
    d = (d + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def _yolo_auxiliary_match_classes(
    tracks: List[Track],
    yolo_meas: List[YoloMeasurement],
    stamp_sec: float,
    max_map_dist_m: float,
    image_iou_min: float,
    match_classes: Set[str],
    on_match,
    *,
    min_track_hits: int = 1,
    distance_match_lidar_ray_only: bool = True,
    track_image_xy: Optional[Dict[int, Tuple[float, float]]] = None,
    image_point_bbox_margin_px: float = 0.0,
    track_laser_bearing_rad: Optional[Dict[int, float]] = None,
    yolo_meas_laser_bearing_rad: Optional[Dict[int, float]] = None,
    bearing_match_max_rad: float = 0.0,
    polar_match_enable: bool = False,
    polar_match_max_range_m: float = 1.0,
    polar_match_max_bearing_rad: float = 0.45,
    track_laser_polar: Optional[Dict[int, Tuple[float, float]]] = None,
    skip_track_indices: Optional[Set[int]] = None,
) -> Set[int]:
    """Asocia detecciones YOLO de `match_classes` a tracks (sin Kalman). Devuelve tracks matcheados."""
    if not tracks or not yolo_meas or not match_classes:
        return set()
    min_hits = max(1, int(min_track_hits))
    class_idx = [
        i
        for i, ym in enumerate(yolo_meas)
        if str(ym.class_name).lower() in match_classes
    ]
    if not class_idx:
        return set()
    used_ti: Set[int] = set()
    used_yi: Set[int] = set()
    skip_ti = skip_track_indices or set()

    def _track_eligible(ti: int) -> bool:
        return (
            ti < len(tracks)
            and tracks[ti].hits >= min_hits
            and ti not in skip_ti
            and ti not in used_ti
        )

    # Fase 0: proyeccion track→imagen (punto dentro de bbox YOLO).
    _bbox_m = max(0.0, float(image_point_bbox_margin_px))
    if track_image_xy:
        by_score = sorted(class_idx, key=lambda i: yolo_meas[i].score, reverse=True)
        for yi in by_score:
            ym = yolo_meas[yi]
            bx1, by1, bx2, by2 = ym.bbox
            best_ti = -1
            best_dist_sq = float("inf")
            bcx = (bx1 + bx2) * 0.5
            bcy = (by1 + by2) * 0.5
            for ti, (u, v) in track_image_xy.items():
                if not _track_eligible(ti):
                    continue
                if (bx1 - _bbox_m) <= u <= (bx2 + _bbox_m) and (by1 - _bbox_m) <= v <= (by2 + _bbox_m):
                    dsq = (u - bcx) ** 2 + (v - bcy) ** 2
                    if dsq < best_dist_sq:
                        best_dist_sq = dsq
                        best_ti = ti
            if best_ti >= 0 and bearing_match_max_rad > 1e-9:
                if (
                    track_laser_bearing_rad is not None
                    and yolo_meas_laser_bearing_rad is not None
                ):
                    b_tr = track_laser_bearing_rad.get(best_ti)
                    b_yo = yolo_meas_laser_bearing_rad.get(yi)
                    if b_tr is None or b_yo is None:
                        best_ti = -1
                    elif _wrapped_abs_angle_diff_rad(float(b_tr), float(b_yo)) > float(
                        bearing_match_max_rad
                    ):
                        best_ti = -1
            if best_ti >= 0:
                used_ti.add(best_ti)
                used_yi.add(yi)
                on_match(best_ti, ym)

    # Fase 1: map_xy del rayo laser (no rgb_only si distance_match_lidar_ray_only)
    with_map: List[int] = []
    for i in class_idx:
        if i in used_yi:
            continue
        ym = yolo_meas[i]
        if ym.map_xy is None:
            continue
        if bool(distance_match_lidar_ray_only) and str(ym.fusion_mode) != "lidar_rgb":
            continue
        with_map.append(i)
    with_map.sort(key=lambda i: yolo_meas[i].score, reverse=True)
    thr = max(0.05, float(max_map_dist_m))
    for yi in with_map:
        ym = yolo_meas[yi]
        mxy = ym.map_xy
        if mxy is None:
            continue
        mx, my = float(mxy[0]), float(mxy[1])
        best_ti = -1
        best_d = thr + 1.0
        for ti, tr in enumerate(tracks):
            if not _track_eligible(ti):
                continue
            tx, ty = tr.state()
            d = math.hypot(mx - tx, my - ty)
            if d < best_d:
                best_d = d
                best_ti = ti
        if best_ti >= 0 and best_d <= thr:
            used_ti.add(best_ti)
            used_yi.add(yi)
            on_match(best_ti, ym)

    # Fase 1b: (r, theta) laser del hit YOLO vs track proyectado al laser (confirmacion profundidad /scan).
    if (
        bool(polar_match_enable)
        and track_laser_polar is not None
        and polar_match_max_range_m > 1e-6
        and polar_match_max_bearing_rad > 1e-9
    ):
        thr_r = max(0.05, float(polar_match_max_range_m))
        thr_a = float(polar_match_max_bearing_rad)
        polar_cand = [
            i
            for i in class_idx
            if i not in used_yi
            and yolo_meas[i].laser_range_m is not None
            and yolo_meas[i].laser_bearing_rad is not None
        ]
        polar_cand.sort(key=lambda i: yolo_meas[i].score, reverse=True)
        for yi in polar_cand:
            ym = yolo_meas[yi]
            ry = float(ym.laser_range_m)  # type: ignore[arg-type]
            ay = float(ym.laser_bearing_rad)  # type: ignore[arg-type]
            best_ti = -1
            best_cost = float("inf")
            for ti in range(len(tracks)):
                if ti in used_ti or not _track_eligible(ti):
                    continue
                pp = track_laser_polar.get(ti)
                if pp is None:
                    continue
                rt, at = float(pp[0]), float(pp[1])
                da = _wrapped_abs_angle_diff_rad(at, ay)
                dr = abs(rt - ry)
                if da <= thr_a and dr <= thr_r:
                    cost = da + 0.35 * (dr / max(thr_r, 0.1))
                    if cost < best_cost:
                        best_cost = cost
                        best_ti = ti
            if best_ti >= 0:
                used_ti.add(best_ti)
                used_yi.add(yi)
                on_match(best_ti, ym)

    # Fase 2: IoU en imagen (tracks con last_bbox)
    remaining = [i for i in class_idx if i not in used_yi]
    remaining.sort(key=lambda i: yolo_meas[i].score, reverse=True)
    iou_thr = float(image_iou_min)
    for yi in remaining:
        ym = yolo_meas[yi]
        best_ti = -1
        best_iou = 0.0
        for ti, tr in enumerate(tracks):
            if ti in used_ti or tr.last_bbox is None or not _track_eligible(ti):
                continue
            iou = _bbox_iou_xyxy(tr.last_bbox, ym.bbox)
            if iou > best_iou:
                best_iou = iou
                best_ti = ti
        if best_ti >= 0 and best_iou >= iou_thr:
            used_ti.add(best_ti)
            used_yi.add(yi)
            on_match(best_ti, ym)

    # Fase 3: mismo angulo en /scan que el rayo al centro (o base) del bbox YOLO.
    if (
        bearing_match_max_rad > 1e-9
        and track_laser_bearing_rad is not None
        and yolo_meas_laser_bearing_rad is not None
    ):
        thr = float(bearing_match_max_rad)
        bearing_cand = [
            i
            for i in class_idx
            if i not in used_yi and yolo_meas_laser_bearing_rad.get(i) is not None
        ]
        bearing_cand.sort(key=lambda i: yolo_meas[i].score, reverse=True)
        for yi in bearing_cand:
            ym = yolo_meas[yi]
            b_y = float(yolo_meas_laser_bearing_rad[yi])
            best_ti = -1
            best_dang = thr + 1.0
            for ti, tr in enumerate(tracks):
                if ti in used_ti or not _track_eligible(ti):
                    continue
                b_t = track_laser_bearing_rad.get(ti)
                if b_t is None:
                    continue
                dang = _wrapped_abs_angle_diff_rad(b_t, b_y)
                if dang < best_dang:
                    best_dang = dang
                    best_ti = ti
            if best_ti >= 0 and best_dang <= thr:
                used_ti.add(best_ti)
                used_yi.add(yi)
                on_match(best_ti, ym)

    return used_ti


def apply_yolo_auxiliary_person_confirmation(
    tracks: List[Track],
    yolo_meas: List[YoloMeasurement],
    stamp_sec: float,
    max_map_dist_m: float,
    image_iou_min: float,
    *,
    min_track_hits: int = 1,
    distance_match_lidar_ray_only: bool = True,
    track_image_xy: Optional[Dict[int, Tuple[float, float]]] = None,
    image_point_bbox_margin_px: float = 0.0,
    track_laser_bearing_rad: Optional[Dict[int, float]] = None,
    yolo_meas_laser_bearing_rad: Optional[Dict[int, float]] = None,
    bearing_match_max_rad: float = 0.0,
    polar_match_enable: bool = False,
    polar_match_max_range_m: float = 1.0,
    polar_match_max_bearing_rad: float = 0.45,
    track_laser_polar: Optional[Dict[int, Tuple[float, float]]] = None,
    reid_cfg: Optional[ReIdConfig] = None,
) -> Set[int]:
    """Sin tocar Kalman: confirma person en tracks laser existentes. Devuelve indices matcheados."""

    def _on_person(ti: int, ym: YoloMeasurement) -> None:
        tr = tracks[ti]
        tr.last_rgb_person_sec = float(stamp_sec)
        tr.yolo_in_fov_reject_streak = 0
        tr.last_bbox = ym.bbox
        tr.last_image_bbox_stamp_sec = float(stamp_sec)
        tr.last_seen_sec = stamp_sec
        tr.score = float(ym.score)
        cn = str(tr.class_name).lower()
        if cn in ("person", "unknown"):
            tr.class_name = "person"
        tr.leg_hypothesis_active = False
        if ym.embedding is not None and reid_cfg is not None:
            add_track_appearance(tr, ym.embedding, stamp_sec, reid_cfg)

    return _yolo_auxiliary_match_classes(
        tracks,
        yolo_meas,
        stamp_sec,
        max_map_dist_m,
        image_iou_min,
        {"person"},
        _on_person,
        min_track_hits=min_track_hits,
        distance_match_lidar_ray_only=distance_match_lidar_ray_only,
        track_image_xy=track_image_xy,
        image_point_bbox_margin_px=image_point_bbox_margin_px,
        track_laser_bearing_rad=track_laser_bearing_rad,
        yolo_meas_laser_bearing_rad=yolo_meas_laser_bearing_rad,
        bearing_match_max_rad=bearing_match_max_rad,
        polar_match_enable=polar_match_enable,
        polar_match_max_range_m=polar_match_max_range_m,
        polar_match_max_bearing_rad=polar_match_max_bearing_rad,
        track_laser_polar=track_laser_polar,
    )


def apply_yolo_auxiliary_person_rejection(
    tracks: List[Track],
    stamp_sec: float,
    track_image_xy: Optional[Dict[int, Tuple[float, float]]],
    person_matched_indices: Set[int],
    *,
    reject_streak_frames: int = 8,
    min_track_hits: int = 1,
    prune_missed_frames: int = 9999,
) -> None:
    """En FOV camara: sin YOLO person alineado → streak; al umbral marca para poda (FP mesa)."""
    if reject_streak_frames <= 0 or not track_image_xy:
        return
    min_hits = max(1, int(min_track_hits))
    in_fov = track_image_xy
    for ti, tr in enumerate(tracks):
        if tr.hits < min_hits or str(tr.class_name).lower() != "person":
            continue
        if ti in person_matched_indices:
            tr.yolo_in_fov_reject_streak = 0
            continue
        if ti not in in_fov:
            tr.yolo_in_fov_reject_streak = 0
            continue
        # Ya confirmado por YOLO alguna vez: no rechazar por ausencia puntual en imagen.
        if float(getattr(tr, "last_rgb_person_sec", 0.0)) > 0.0:
            tr.yolo_in_fov_reject_streak = 0
            continue
        tr.yolo_in_fov_reject_streak += 1
        if tr.yolo_in_fov_reject_streak >= int(reject_streak_frames):
            tr.missed_frames = max(int(tr.missed_frames), int(prune_missed_frames))


def update_yolo_auxiliary_camera_fov_state(
    tracks: List[Track],
    stamp_sec: float,
    track_image_xy: Optional[Dict[int, Tuple[float, float]]],
    *,
    min_track_hits: int = 1,
) -> None:
    """Marca tracks dentro/fuera del FOV camara (color verde-azulado al salir del FOV)."""
    min_hits = max(1, int(min_track_hits))
    in_fov = track_image_xy or {}
    for ti, tr in enumerate(tracks):
        if tr.hits < min_hits or str(tr.class_name).lower() != "person":
            continue
        if ti in in_fov:
            tr.last_camera_fov_sec = float(stamp_sec)
        else:
            tr.last_outside_camera_fov_sec = float(stamp_sec)


def update_yolo_auxiliary_fov_unseen(
    tracks: List[Track],
    stamp_sec: float,
    track_image_xy: Optional[Dict[int, Tuple[float, float]]],
    *,
    min_track_hits: int = 1,
) -> None:
    """Alias retrocompatible: actualiza estado FOV camara."""
    update_yolo_auxiliary_camera_fov_state(
        tracks, stamp_sec, track_image_xy, min_track_hits=min_track_hits
    )


def update_tracks(
    tracks: List[Track],
    next_track_id: int,
    yolo_meas: List[YoloMeasurement],
    lidar_meas: List[LidarMeasurement],
    dt_sec: float,
    stamp_sec: float,
    enable_lidar_fusion: bool,
    assoc_threshold: float,
    max_missed_frames: int,
    leg_hypothesis_min_score: float = 0.62,
    lidar_assoc_threshold: float = 1.6,
    person_lidar_assoc_threshold: float = 0.0,
    person_max_missed_frames: int = 45,
    spawn_unknown_from_lidar: bool = False,
    leg_hypothesis_as_person: bool = False,
    enable_image_only_assoc_fallback: bool = True,
    image_only_assoc_min_iou: float = 0.22,
    person_dynamic_window_sec: float = 2.0,
    person_dynamic_min_displacement_m: float = 0.32,
    person_dynamic_pose_history_sec: float = 4.0,
    person_dynamic_classes: Tuple[str, ...] = ("person",),
    person_lidar_innovation_gate_m: float = 0.0,
    person_lidar_innovation_gate_stale_rgb_sec: float = 3.5,
    # Si |cluster - prediccion| > umbral (m), no correct() aunque Hungaro emparejo (evita salto a muro al girar).
    person_lidar_max_correction_jump_m: float = 0.0,
    # Tras correct(), si |v| < esto, anular velocidad Kalman (caja quieta en mapa).
    person_static_zero_velocity_speed_m: float = 0.0,
    # Suaviza saltos de bbox verde: YOLO+map_xy jitter al mover el robot.
    person_yolo_measurement_noise_scale: float = 3.0,
    # Sin rayo laser (rgb_only): casi no mover el Kalman (proyeccion camara + TF inestables al mover base).
    person_yolo_rgb_only_measurement_noise_scale: float = 14.0,
    # 0 = off. Si |yolo - prediccion| > gate (m), no corrige Kalman (evita TF suelto).
    person_yolo_innovation_gate_m: float = 0.0,
    scan_verify_data=None,
    scan_verify_range_tolerance_m: float = 0.50,
    scan_verify_window_deg: float = 15.0,
    # Ciclos sin cluster ni scan_verify antes de missed++ (FOV). 0 = comportamiento anterior.
    person_missed_debounce_frames: int = 0,
    # Spencer Logic Initiator: separate limits for never-RGB tracks
    person_lidar_only_max_missed_frames: int = 20,
    lidar_initiation_max_age_static_sec: float = 8.0,
    # Tracks que ya se alejaron > esto de su nacimiento no se podan por estaticos. 0 = off.
    lidar_initiation_static_min_travel_m: float = 0.0,
    # Spencer occlusion: tracks outside LiDAR FOV are OCCLUDED (not MISSED).
    max_occlusions_before_deletion: int = 120,
    # Spencer: tracks maduros aguantan mas ciclos oclusos (detras del robot). 0 = mismo limite que arriba.
    max_occlusions_mature_track: int = 0,
    # Spencer: Mahalanobis + chi² gating for LiDAR association.
    use_mahalanobis_lidar: bool = False,
    # Ventana s tras ultimo YOLO person: no corregir Kalman con laser (solo scan verify) → OL estable.
    person_hold_skip_lidar_correct_sec: float = 0.0,
    # Persona ya matcheada por YOLO: sigue entrando en asociacion laser (posicion fiable al girar).
    person_lidar_refinement_after_yolo: bool = True,
    # Multiplica ruido medicion YOLO en persona si hay refinamiento laser (YOLO solo ancla debil).
    person_yolo_weak_scale_with_lidar_refinement: float = 8.0,
    # Si YOLO no matcheo por distancia en mapa pero el bbox solapa (IoU) un track sin matchear, unificar (menos IDs duplicados).
    yolo_merge_unmatched_iou: float = 0.0,
    # Spencer: con clusters en el ciclo, no corregir Kalman con YOLO en persona; laser primero, YOLO si falla laser.
    person_yolo_defer_position_to_lidar: bool = True,
    # Con defer+laser: antes del Hungaro en mapa, emparejar person<->person YOLO por IoU imagen (estable al girar).
    person_yolo_match_by_iou_first: bool = True,
    person_yolo_assoc_iou_min: float = 0.12,
    # Con fusion laser: YOLO no hace correct() en mapa (ninguna clase) y no abre tracks desde detecciones camara.
    # Solo actualiza etiqueta/bbox/RGB; la posicion Kalman queda 100% laser + predict.
    yolo_lidar_fusion_label_only: bool = False,
    # Tras correct() laser: last_update_source=lidar-only (como sin YOLO); verde RViz por last_rgb reciente.
    yolo_visual_lidar_parity: bool = False,
    # No crear track laser-only person cerca de un person con RGB reciente (reduce FP en muros junto al confirmado).
    person_lidar_spawn_suppress_radius_m: float = 0.0,
    person_lidar_spawn_suppress_rgb_sec: float = 0.0,
    # 0 = usar leg_hypothesis_min_score; >0 umbral minimo para spawn laser-only person.
    lidar_spawn_person_min_leg_score: float = 0.0,
    # >0: en un mismo ciclo no crear dos tracks laser si los centros quedan mas cerca (evita #25/#26 duplicados).
    lidar_spawn_same_frame_dedup_m: float = 0.0,
    # Lista mutable en el nodo; 0 frames = crear person laser al instante (comportamiento previo).
    pending_lidar_spawns: Optional[List[PendingLidarSpawn]] = None,
    # Fase 2 dual-tier: solo estos clusters abren track (None = unmatched maintenance).
    lidar_spawn_meas: Optional[List[LidarMeasurement]] = None,
    lidar_spawn_confirm_frames: int = 0,
    lidar_spawn_confirm_radius_m: float = 0.42,
    spawn_initiator_max_step_displacement_m: float = 0.0,
    spawn_initiator_max_velocity_mps: float = 0.0,
    person_track_dual_audit_enable: bool = False,
    person_track_dual_audit_interval_frames: int = 8,
    person_track_dual_audit_radius_m: float = 0.42,
    person_yolo_seen_green_sec: float = 0.45,
    person_track_leg_snap_enable: bool = True,
    person_track_leg_snap_max_dist_m: float = 0.90,
    person_track_leg_snap_min_leg_score: float = 0.65,
    person_lidar_micro_stable_m: float = 0.10,
    person_track_leg_snap_min_jump_for_velocity_m: float = 0.45,
    person_track_leg_snap_max_velocity_mps: float = 1.0,
    person_track_leg_snap_require_spawn_tier_jump_m: float = 0.55,
    scan_verify_require_nearby_cluster: bool = True,
    scan_verify_nearby_cluster_m: float = 0.55,
    scan_verify_nearby_min_leg_score: float = 0.62,
    # 0 = off. IoU en mapa entre cajas person (lado = person_box_side_m): quita tracks duplicados mismo blanco.
    person_track_map_merge_min_iou: float = 0.0,
    person_track_map_merge_box_m: float = 0.6,
    person_track_map_merge_lidar_only: bool = True,
    duplicate_track_temporal_enable: bool = False,
    duplicate_track_compare_samples: int = 10,
    duplicate_track_must_match: int = 6,
    duplicate_track_max_dist_m: float = 0.18,
    duplicate_track_lidar_only: bool = True,
) -> int:
    """Actualiza `tracks` in place y devuelve el siguiente `track_id` libre."""
    dyn_classes = {str(c).lower() for c in person_dynamic_classes}
    _confirm_n = int(lidar_spawn_confirm_frames)
    if pending_lidar_spawns is not None and _confirm_n > 0:
        for _p in pending_lidar_spawns:
            _p.touched_this_cycle = False

    for tr in tracks:
        tr.predict(dt_sec)
        tr.lidar_audit_counter = int(getattr(tr, "lidar_audit_counter", 0)) + 1

    track_indices = list(range(len(tracks)))
    yolo_association_to_meas_idx: List[int] = []
    for idx, ym in enumerate(yolo_meas):
        if ym.map_xy is None:
            continue
        yolo_association_to_meas_idx.append(idx)

    person_yolo_matched_ti: Set[int] = set()
    # Cualquier track que matcheo YOLO este ciclo (cualquier clase): debe seguir en el pool laser;
    # si solo usaramos person_yolo_matched_ti, un match YOLO no-person excluiria el track del Hungaro laser.
    all_yolo_matched_ti: Set[int] = set()
    _refine_lidar = (
        bool(person_lidar_refinement_after_yolo)
        and enable_lidar_fusion
        and len(lidar_meas) > 0
    )
    _weak_yolo = float(person_yolo_weak_scale_with_lidar_refinement)
    # Persona: YOLO solo confirma (clase/bbox/RGB); el Kalman en mapa lo mueve solo el laser (o predict si no hay match).
    # No exigir len(lidar_meas)>0: si un ciclo no trae clusters, igual no debe proyectar YOLO sobre el estado.
    _defer_yolo_pos = bool(person_yolo_defer_position_to_lidar) and bool(enable_lidar_fusion)
    _yolo_label_only = bool(yolo_lidar_fusion_label_only) and bool(enable_lidar_fusion)

    def _apply_yolo_match(ti: int, ym_idx: int) -> None:
        tr = tracks[ti]
        ym = yolo_meas[ym_idx]
        if ym.map_xy is None:
            return
        all_yolo_matched_ti.add(ti)
        if str(ym.class_name).lower() == "person":
            person_yolo_matched_ti.add(ti)
        mx, my = ym.map_xy[0], ym.map_xy[1]
        yolo_gate = float(person_yolo_innovation_gate_m)
        if yolo_gate > 0.0 and str(ym.class_name).lower() == "person":
            px, py = tr.association_position()
            if measurement_cost(px, py, mx, my) > yolo_gate:
                tr.class_name = ym.class_name
                tr.score = ym.score
                tr.last_bbox = ym.bbox
                tr.last_image_bbox_stamp_sec = float(stamp_sec)
                tr.last_seen_sec = stamp_sec
                tr.last_rgb_person_sec = float(stamp_sec)
                tr.hits += 1
                tr.missed_frames = 0
                tr.consecutive_occlusions = 0
                tr.leg_hypothesis_active = False
                tr.leg_hypothesis_score = 0.0
                if ym.fusion_mode == "lidar_rgb":
                    tr.last_update_source = "camera+lidar"
                else:
                    tr.last_update_source = "camera-only"
                tr.record_map_pose_after_correct(stamp_sec, person_dynamic_pose_history_sec)
                return
        y_scale = 1.0
        y_scale_base = 1.0
        if str(ym.class_name).lower() == "person":
            y_scale_base = float(person_yolo_measurement_noise_scale)
            if ym.fusion_mode == "rgb_only":
                y_scale_base = max(
                    y_scale_base, float(person_yolo_rgb_only_measurement_noise_scale)
                )
            y_scale = y_scale_base
            if _refine_lidar and ti in person_yolo_matched_ti:
                y_scale *= max(1.0, _weak_yolo)
        is_person = str(ym.class_name).lower() == "person"
        skip_kalman_yolo = _yolo_label_only or (_defer_yolo_pos and is_person)
        if not skip_kalman_yolo:
            tr.correct(mx, my, measurement_noise_scale=y_scale)
        tr.class_name = ym.class_name
        tr.score = ym.score
        tr.last_bbox = ym.bbox
        tr.last_image_bbox_stamp_sec = float(stamp_sec)
        tr.last_seen_sec = stamp_sec
        if is_person:
            tr.last_rgb_person_sec = float(stamp_sec)
        tr.hits += 1
        tr.missed_frames = 0
        tr.consecutive_occlusions = 0
        tr.leg_hypothesis_active = False
        tr.leg_hypothesis_score = 0.0
        if ym.fusion_mode == "lidar_rgb":
            tr.last_update_source = "camera+lidar"
        else:
            tr.last_update_source = "camera-only"
        if not skip_kalman_yolo:
            tr.record_map_pose_after_correct(stamp_sec, person_dynamic_pose_history_sec)

    iou_matched_ti: Set[int] = set()
    iou_matched_ym: Set[int] = set()
    use_iou_first = (
        bool(person_yolo_match_by_iou_first)
        and bool(person_yolo_defer_position_to_lidar)
        and bool(enable_lidar_fusion)
        and len(yolo_association_to_meas_idx) > 0
    )
    if use_iou_first:
        for ti, ym_idx in _assign_person_yolo_iou_pairs(
            tracks,
            track_indices,
            yolo_meas,
            yolo_association_to_meas_idx,
            float(person_yolo_assoc_iou_min),
        ):
            _apply_yolo_match(ti, ym_idx)
            iou_matched_ti.add(ti)
            iou_matched_ym.add(ym_idx)

    track_indices_f = [ti for ti in track_indices if ti not in iou_matched_ti]
    yolo_assoc_meas_idx = [j for j in yolo_association_to_meas_idx if j not in iou_matched_ym]
    yolo_association_f = [
        (yolo_meas[j].map_xy[0], yolo_meas[j].map_xy[1], assoc_class_key(yolo_meas[j].class_name))
        for j in yolo_assoc_meas_idx
    ]

    cam_matches, unmatched_tracks, unmatched_yolo = assign_measurements(
        tracks, track_indices_f, yolo_association_f, assoc_threshold
    )

    for ti, mi_col in cam_matches:
        ym_idx = yolo_assoc_meas_idx[mi_col]
        _apply_yolo_match(ti, ym_idx)

    _iou_merge_thr = float(yolo_merge_unmatched_iou)
    if _iou_merge_thr > 0.0 and unmatched_yolo and unmatched_tracks:
        pairs_merge: List[Tuple[float, int, int]] = []
        for mi in unmatched_yolo:
            ym = yolo_meas[yolo_assoc_meas_idx[mi]]
            if ym.map_xy is None:
                continue
            ck = assoc_class_key(ym.class_name)
            for ti in unmatched_tracks:
                tr = tracks[ti]
                if tr.last_bbox is None:
                    continue
                if tr.class_name != "unknown" and assoc_class_key(tr.class_name) != ck:
                    continue
                iou = _bbox_iou_xyxy(tr.last_bbox, ym.bbox)
                if iou >= _iou_merge_thr:
                    pairs_merge.append((iou, ti, mi))
        pairs_merge.sort(key=lambda x: -x[0])
        used_merge_t: Set[int] = set()
        used_merge_m: Set[int] = set()
        for _, ti, mi in pairs_merge:
            if ti in used_merge_t or mi in used_merge_m:
                continue
            used_merge_t.add(ti)
            used_merge_m.add(mi)
            ym_idx = yolo_assoc_meas_idx[mi]
            _apply_yolo_match(ti, ym_idx)
        if used_merge_t:
            unmatched_tracks = [ti for ti in unmatched_tracks if ti not in used_merge_t]
        if used_merge_m:
            unmatched_yolo = [mi for mi in unmatched_yolo if mi not in used_merge_m]

    # Minimum distance to avoid spawning duplicates near an existing track
    _min_spawn_dist = max(0.35, assoc_threshold * 0.5)
    _min_lidar_spawn_dist = max(0.58, assoc_threshold * 0.62)
    if not _yolo_label_only:
        for mi in unmatched_yolo:
            ym = yolo_meas[yolo_assoc_meas_idx[mi]]
            if ym.map_xy is None:
                continue
            mx, my = ym.map_xy
            # Skip if too close to any existing track of the same class
            too_close = False
            for etr in tracks:
                ex, ey = etr.association_position()
                if measurement_cost(mx, my, ex, ey) < _min_spawn_dist:
                    if assoc_class_key(etr.class_name) == assoc_class_key(ym.class_name):
                        too_close = True
                        break
            if too_close:
                continue
            tr = Track(
                track_id=next_track_id,
                class_name=ym.class_name,
                map_x=mx,
                map_y=my,
                score=ym.score,
                stamp_sec=stamp_sec,
                bbox=ym.bbox,
            )
            tr.last_update_source = "camera+lidar" if ym.fusion_mode == "lidar_rgb" else "camera-only"
            if str(ym.class_name).lower() == "person":
                tr.last_rgb_person_sec = float(stamp_sec)
            tr.record_map_pose_after_correct(stamp_sec, person_dynamic_pose_history_sec)
            next_track_id += 1
            tracks.append(tr)

    if enable_image_only_assoc_fallback and yolo_meas:
        unmatched_tracks = _image_only_soft_associate(
            tracks,
            unmatched_tracks,
            yolo_meas,
            stamp_sec,
            image_only_assoc_min_iou,
        )

    if enable_lidar_fusion and lidar_meas:
        # Spencer: alta precision en asociacion person↔cluster (no 0.5× umbral: eso metia muros).
        lidar_eligible = list(unmatched_tracks)
        if person_lidar_refinement_after_yolo:
            for ti in all_yolo_matched_ti:
                if ti not in lidar_eligible:
                    lidar_eligible.append(ti)
        _score_floor = max(0.58, leg_hypothesis_min_score * 0.88)
        high_score_indices = [
            mi for mi, lm in enumerate(lidar_meas)
            if lm.leg_candidate_score >= _score_floor
        ]
        hold_sec = float(person_hold_skip_lidar_correct_sec)
        scan_verify_person: List[int] = []
        person_for_lidar: List[int] = []
        other_eligible: List[int] = []
        for ti in lidar_eligible:
            tr = tracks[ti]
            if tr.class_name != "person":
                other_eligible.append(ti)
                continue
            skip_hold_for_refine = person_lidar_refinement_after_yolo and ti in all_yolo_matched_ti
            if (
                hold_sec > 0.0
                and not skip_hold_for_refine
                and tr.last_rgb_person_sec > 0.0
                and (stamp_sec - tr.last_rgb_person_sec) <= hold_sec
            ):
                scan_verify_person.append(ti)
                continue
            person_for_lidar.append(ti)
        person_eligible = person_for_lidar

        def _lidar_pair_allowed(tr: Track, mx: float, my: float) -> bool:
            gate = float(person_lidar_innovation_gate_m)
            if gate <= 0.0:
                return True
            if tr.class_name != "person":
                return True
            # Con YOLO desactivado last_rgb_person_sec=0 → gate nunca aplica (comportamiento estable).
            # Con YOLO activo last_rgb>0 y gate>0: se podian rechazar clusters laser legitimos si la prediccion
            # o el TF al girar desalineaban >gate m → sin correct() laser, drift/occlusion (cajas "van con el robot").
            # Si la posicion en mapa la define solo el laser (defer / label-only), no apretar DA laser con RGB.
            if _yolo_label_only or _defer_yolo_pos:
                return True
            # Spencer-like: gate solo con RGB reciente; si no hubo RGB o paso stale_win, confiar en laser (map).
            if tr.last_rgb_person_sec <= 0.0:
                return True
            stale_win = float(person_lidar_innovation_gate_stale_rgb_sec)
            if stale_win > 0.0 and (stamp_sec - tr.last_rgb_person_sec) > stale_win:
                return True
            px, py = tr.association_position()
            return measurement_cost(px, py, mx, my) <= gate

        # Phase 1: Person tracks → only high-score clusters (legs).
        all_lidar_matches: list = []
        used_lidar_meas: set = set()
        remaining_person = person_eligible
        if person_eligible and high_score_indices:
            hs_assoc = [(lidar_meas[mi].map_x, lidar_meas[mi].map_y, None) for mi in high_score_indices]
            p_thr = person_lidar_assoc_threshold if person_lidar_assoc_threshold > 0.0 else lidar_assoc_threshold
            pm, p_unmatched, hs_unmatched = assign_measurements(
                tracks, person_eligible, hs_assoc, p_thr,
                pair_allowed=_lidar_pair_allowed,
                use_mahalanobis=use_mahalanobis_lidar,
            )
            for ti, hsi in pm:
                real_mi = high_score_indices[hsi]
                all_lidar_matches.append((ti, real_mi))
                used_lidar_meas.add(real_mi)
            remaining_person = p_unmatched

        # Phase 1b: Personas sin match tras fase 1 (solo clusters leg_candidate_score "high").
        # Por defecto solo last_rgb<=0: si ya hubo YOLO, no usar suelo laxo (evita muros en persona "confirmada").
        # Con defer/label-only la posicion en mapa es laser: person con last_rgb>0 que fallan fase 1 NO deben
        # quedar fuera de 1b (no entran en fase 2 = solo no-person) → sin correct() laser ese ciclo → deriva.
        person_phase1b_eligible = [
            ti
            for ti in remaining_person
            if tracks[ti].class_name == "person"
            and (
                tracks[ti].last_rgb_person_sec <= 0.0
                or _defer_yolo_pos
                or _yolo_label_only
            )
        ]
        _hi_floor = max(0.58, leg_hypothesis_min_score * 0.88)
        # Sin YOLO, todos los person son "never_rgb": si el suelo es ~0.52×leg, segmentos de muro
        # (~0.45–0.65) siguen matcheando y arrastran la caja. Acotar: 1b no más laxo que hi−δ.
        _never_rgb_soft_cap_below_hi = 0.08
        _soft_floor = max(
            0.38,
            min(leg_hypothesis_min_score * 0.52, _hi_floor - 1e-3),
            _hi_floor - _never_rgb_soft_cap_below_hi,
        )
        if person_phase1b_eligible:
            soft_indices = [
                mi
                for mi in range(len(lidar_meas))
                if mi not in used_lidar_meas and lidar_meas[mi].leg_candidate_score >= _soft_floor
            ]
            if soft_indices:
                p_thr_b = person_lidar_assoc_threshold if person_lidar_assoc_threshold > 0.0 else lidar_assoc_threshold
                soft_assoc = [
                    (lidar_meas[mi].map_x, lidar_meas[mi].map_y, None) for mi in soft_indices
                ]
                pm_b, _, _ = assign_measurements(
                    tracks,
                    person_phase1b_eligible,
                    soft_assoc,
                    p_thr_b,
                    pair_allowed=_lidar_pair_allowed,
                    use_mahalanobis=use_mahalanobis_lidar,
                )
                matched_1b: Set[int] = set()
                for ti, ssi in pm_b:
                    real_mi = soft_indices[ssi]
                    all_lidar_matches.append((ti, real_mi))
                    used_lidar_meas.add(real_mi)
                    matched_1b.add(ti)
                if matched_1b:
                    remaining_person = [ti for ti in remaining_person if ti not in matched_1b]

        # Phase 2: Non-person tracks → all remaining clusters.
        remaining_other = other_eligible
        if other_eligible:
            all_assoc = [
                (lidar_meas[mi].map_x, lidar_meas[mi].map_y, None)
                for mi in range(len(lidar_meas)) if mi not in used_lidar_meas
            ]
            remaining_idx_map = [mi for mi in range(len(lidar_meas)) if mi not in used_lidar_meas]
            if all_assoc:
                om, o_unmatched, o_um_meas = assign_measurements(
                    tracks, other_eligible, all_assoc, lidar_assoc_threshold,
                )
                for ti, ai in om:
                    real_mi = remaining_idx_map[ai]
                    all_lidar_matches.append((ti, real_mi))
                    used_lidar_meas.add(real_mi)
                remaining_other = o_unmatched

        person_pred_xy: Dict[int, Tuple[float, float]] = {}
        for ti in person_eligible:
            person_pred_xy[ti] = tracks[ti].association_position()

        # Apply corrections for all LiDAR matches.
        lidar_corrected_ti: Set[int] = set()
        lidar_position_corrected_ti: Set[int] = set()
        lidar_match_mi: Dict[int, int] = {}
        _max_lidar_jump = float(person_lidar_max_correction_jump_m)
        _static_v0 = float(person_static_zero_velocity_speed_m)
        for ti, mi in all_lidar_matches:
            lidar_match_mi[ti] = mi
            tr = tracks[ti]
            lm = lidar_meas[mi]
            px, py = tr.association_position()
            jump = math.hypot(lm.map_x - px, lm.map_y - py)
            skip_correct = (
                _max_lidar_jump > 0.0
                and tr.class_name == "person"
                and jump > _max_lidar_jump * 1.45
            )
            position_updated = False
            if skip_correct:
                used_lidar_meas.discard(mi)
            elif (
                _max_lidar_jump > 0.0
                and tr.class_name == "person"
                and jump > _max_lidar_jump
            ):
                # Giro brusco: aceptar cluster con mas ruido de medicion (no congelar caja).
                tr.correct(lm.map_x, lm.map_y, measurement_noise_scale=3.5)
                lidar_position_corrected_ti.add(ti)
                _record_person_lidar_position(tr, lm.map_x, lm.map_y, stamp_sec)
                position_updated = True
            else:
                _micro_m = float(person_lidar_micro_stable_m)
                if not _person_lidar_micro_jitter(tr, lm.map_x, lm.map_y, _micro_m):
                    tr.correct(lm.map_x, lm.map_y)
                    lidar_position_corrected_ti.add(ti)
                    _record_person_lidar_position(tr, lm.map_x, lm.map_y, stamp_sec)
                    position_updated = True
                    if _static_v0 > 0.0 and jump < 0.22:
                        vx, vy = tr.velocity()
                        if math.hypot(vx, vy) < _static_v0:
                            tr.zero_velocity()
            if position_updated or tr.class_name != "person":
                lidar_corrected_ti.add(ti)
                tr.last_seen_sec = stamp_sec
                tr.missed_frames = 0
                tr.consecutive_no_lidar_presence = 0
                tr.consecutive_occlusions = 0
                tr.hits += 1
            if bool(yolo_visual_lidar_parity):
                tr.last_update_source = "lidar-only"
            elif ti in all_yolo_matched_ti:
                tr.last_update_source = "camera+lidar"
            else:
                tr.last_update_source = "lidar-only"
            if _defer_yolo_pos and ti in person_yolo_matched_ti:
                tr.record_map_pose_after_correct(stamp_sec, person_dynamic_pose_history_sec)
            tr.leg_hypothesis_score = max(tr.leg_hypothesis_score, lm.leg_candidate_score)
            if tr.class_name == "unknown" and lm.leg_candidate_score >= leg_hypothesis_min_score:
                cls, sc, hyp = _lidar_person_hypothesis_from_leg(
                    lm, leg_hypothesis_min_score, leg_hypothesis_as_person
                )
                tr.class_name = cls
                tr.score = sc
                tr.leg_hypothesis_active = hyp
            elif (
                tr.class_name == "person"
                and tr.last_bbox is None
                and lm.leg_candidate_score >= leg_hypothesis_min_score
            ):
                tr.leg_hypothesis_active = True

        if person_track_leg_snap_enable and person_eligible:
            _snap_max = float(person_track_leg_snap_max_dist_m)
            _snap_min = float(person_track_leg_snap_min_leg_score)
            snap_queue: List[Tuple[float, int, int]] = []
            for ti in person_eligible:
                tr = tracks[ti]
                if tr.class_name != "person":
                    continue
                pred_x, pred_y = person_pred_xy.get(ti, tr.association_position())
                st_x, st_y = tr.state()
                anchors = [(pred_x, pred_y), (st_x, st_y)]
                allowed_used = set(used_lidar_meas)
                own_mi = lidar_match_mi.get(ti)
                if own_mi is not None:
                    allowed_used.discard(own_mi)
                mi, dist = _nearest_leg_cluster_multi_anchor(
                    anchors, lidar_meas, allowed_used, _snap_min, _snap_max
                )
                if mi is None:
                    continue
                lm = lidar_meas[mi]
                if ti in lidar_position_corrected_ti and own_mi == mi:
                    sx, sy = tr.state()
                    if measurement_cost(sx, sy, lm.map_x, lm.map_y) < float(
                        person_lidar_micro_stable_m
                    ):
                        continue
                _req_spawn_jump = float(person_track_leg_snap_require_spawn_tier_jump_m)
                if _req_spawn_jump > 0.0 and dist > _req_spawn_jump and not lm.spawn_tier_pass:
                    continue
                if _other_person_track_closer(tracks, person_eligible, ti, lm, dist):
                    continue
                snap_queue.append((dist, ti, mi))
            snap_queue.sort(key=lambda item: item[0])
            for _dist, ti, mi in snap_queue:
                if mi in used_lidar_meas and lidar_match_mi.get(ti) != mi:
                    continue
                tr = tracks[ti]
                lm = lidar_meas[mi]
                if lidar_match_mi.get(ti) is not None:
                    used_lidar_meas.discard(lidar_match_mi[ti])
                snap_from_x, snap_from_y = st_x, st_y
                if not _person_lidar_micro_jitter(tr, lm.map_x, lm.map_y, float(person_lidar_micro_stable_m)):
                    tr.correct(lm.map_x, lm.map_y)
                    _record_person_lidar_position(tr, lm.map_x, lm.map_y, stamp_sec)
                snap_jump = math.hypot(lm.map_x - snap_from_x, lm.map_y - snap_from_y)
                _vel_jump_min = float(person_track_leg_snap_min_jump_for_velocity_m)
                _vel_cap = float(person_track_leg_snap_max_velocity_mps)
                if snap_jump > _vel_jump_min and dt_sec > 1e-3:
                    vx = (lm.map_x - snap_from_x) / dt_sec
                    vy = (lm.map_y - snap_from_y) / dt_sec
                    sp = math.hypot(vx, vy)
                    if _vel_cap > 0.0 and sp > _vel_cap:
                        vx *= _vel_cap / sp
                        vy *= _vel_cap / sp
                    tr.set_velocity(vx, vy)
                elif _static_v0 > 0.0 and snap_jump < 0.22:
                    vx, vy = tr.velocity()
                    if math.hypot(vx, vy) < _static_v0:
                        tr.zero_velocity()
                lidar_position_corrected_ti.add(ti)
                lidar_corrected_ti.add(ti)
                used_lidar_meas.add(mi)
                tr.last_seen_sec = stamp_sec
                tr.last_lidar_correct_sec = float(stamp_sec)
                tr.missed_frames = 0
                tr.consecutive_no_lidar_presence = 0
                tr.consecutive_occlusions = 0
                tr.hits += 1
                if bool(yolo_visual_lidar_parity):
                    tr.last_update_source = "lidar-only"
                tr.leg_hypothesis_score = max(tr.leg_hypothesis_score, lm.leg_candidate_score)

        unmatched_lidar_meas = [mi for mi in range(len(lidar_meas)) if mi not in used_lidar_meas]
        unmatched_tracks = remaining_person + remaining_other + scan_verify_person
        _spawn_assoc_thr = (
            float(person_lidar_assoc_threshold)
            if float(person_lidar_assoc_threshold) > 0.0
            else float(lidar_assoc_threshold)
        )
        if lidar_spawn_meas is not None:
            spawn_pool: List[LidarMeasurement] = list(lidar_spawn_meas)
        else:
            spawn_pool = [lidar_meas[mi] for mi in unmatched_lidar_meas]
        _spawn_thr = (
            float(lidar_spawn_person_min_leg_score)
            if float(lidar_spawn_person_min_leg_score) > 0.0
            else float(leg_hypothesis_min_score)
        )
        _spawn_suppress_r = float(person_lidar_spawn_suppress_radius_m)
        _spawn_suppress_sec = float(person_lidar_spawn_suppress_rgb_sec)
        _spawn_dedup = float(lidar_spawn_same_frame_dedup_m)
        _confirm_r = float(lidar_spawn_confirm_radius_m)
        _spawn_pts_frame: List[Tuple[float, float]] = []
        _spawn_order = sorted(
            range(len(spawn_pool)),
            key=lambda i: spawn_pool[i].leg_candidate_score,
            reverse=True,
        )
        for si in _spawn_order:
            lm = spawn_pool[si]
            if lidar_spawn_meas is not None:
                assoc_blocked = False
                for mi in used_lidar_meas:
                    um = lidar_meas[mi]
                    if measurement_cost(lm.map_x, lm.map_y, um.map_x, um.map_y) < _spawn_assoc_thr:
                        assoc_blocked = True
                        break
                if assoc_blocked:
                    continue
            cls, sc, hyp = _lidar_person_hypothesis_from_leg(
                lm, _spawn_thr, leg_hypothesis_as_person
            )
            if cls == "unknown" and not spawn_unknown_from_lidar:
                continue
            if _spawn_suppress_r > 0.0 and _spawn_suppress_sec > 0.0:
                spawn_blocked = False
                for etr in tracks:
                    if etr.class_name != "person":
                        continue
                    if etr.last_rgb_person_sec <= 0.0:
                        continue
                    if stamp_sec - etr.last_rgb_person_sec > _spawn_suppress_sec:
                        continue
                    ex, ey = etr.association_position()
                    if measurement_cost(lm.map_x, lm.map_y, ex, ey) < _spawn_suppress_r:
                        spawn_blocked = True
                        break
                if spawn_blocked:
                    continue
            # Skip if too close to any existing track
            too_close = False
            for etr in tracks:
                ex, ey = etr.association_position()
                if measurement_cost(lm.map_x, lm.map_y, ex, ey) < _min_lidar_spawn_dist:
                    too_close = True
                    break
            if too_close:
                continue
            if _spawn_dedup > 0.0:
                dup = False
                for sx, sy in _spawn_pts_frame:
                    if measurement_cost(lm.map_x, lm.map_y, sx, sy) < _spawn_dedup:
                        dup = True
                        break
                if dup:
                    continue

            if cls == "person" and pending_lidar_spawns is not None and _confirm_n > 0:
                matched: Optional[PendingLidarSpawn] = None
                for p in pending_lidar_spawns:
                    if measurement_cost(lm.map_x, lm.map_y, p.map_x, p.map_y) < _confirm_r:
                        matched = p
                        break
                if matched is not None:
                    if _spawn_initiator_accepts_step(
                        matched,
                        lm.map_x,
                        lm.map_y,
                        stamp_sec,
                        float(spawn_initiator_max_step_displacement_m),
                        float(spawn_initiator_max_velocity_mps),
                    ):
                        if not matched.touched_this_cycle:
                            matched.touched_this_cycle = True
                            matched.streak += 1
                    else:
                        matched.streak = 1
                        matched.created_stamp_sec = stamp_sec
                    matched.map_x = lm.map_x
                    matched.map_y = lm.map_y
                    matched.last_stamp_sec = stamp_sec
                    matched.leg_candidate_score = max(matched.leg_candidate_score, lm.leg_candidate_score)
                    matched.class_name = cls
                    matched.score = sc
                    matched.leg_hypothesis_active = hyp
                    if matched.streak >= _confirm_n:
                        tr = Track(
                            track_id=next_track_id,
                            class_name=cls,
                            map_x=lm.map_x,
                            map_y=lm.map_y,
                            score=sc,
                            stamp_sec=stamp_sec,
                            bbox=None,
                        )
                        tr.last_update_source = "lidar-only"
                        tr.leg_hypothesis_active = hyp
                        tr.leg_hypothesis_score = lm.leg_candidate_score
                        next_track_id += 1
                        tracks.append(tr)
                        pending_lidar_spawns.remove(matched)
                else:
                    npend = PendingLidarSpawn(
                        map_x=lm.map_x,
                        map_y=lm.map_y,
                        streak=1,
                        last_stamp_sec=stamp_sec,
                        created_stamp_sec=stamp_sec,
                        leg_candidate_score=lm.leg_candidate_score,
                        class_name=cls,
                        score=sc,
                        leg_hypothesis_active=hyp,
                        touched_this_cycle=True,
                    )
                    pending_lidar_spawns.append(npend)
                    if npend.streak >= _confirm_n:
                        tr = Track(
                            track_id=next_track_id,
                            class_name=cls,
                            map_x=lm.map_x,
                            map_y=lm.map_y,
                            score=sc,
                            stamp_sec=stamp_sec,
                            bbox=None,
                        )
                        tr.last_update_source = "lidar-only"
                        tr.leg_hypothesis_active = hyp
                        tr.leg_hypothesis_score = lm.leg_candidate_score
                        next_track_id += 1
                        tracks.append(tr)
                        pending_lidar_spawns.remove(npend)
                if _spawn_dedup > 0.0:
                    _spawn_pts_frame.append((lm.map_x, lm.map_y))
                continue

            tr = Track(
                track_id=next_track_id,
                class_name=cls,
                map_x=lm.map_x,
                map_y=lm.map_y,
                score=sc,
                stamp_sec=stamp_sec,
                bbox=None,
            )
            tr.last_update_source = "lidar-only"
            tr.leg_hypothesis_active = hyp
            tr.leg_hypothesis_score = lm.leg_candidate_score
            next_track_id += 1
            tracks.append(tr)
            if _spawn_dedup > 0.0:
                _spawn_pts_frame.append((lm.map_x, lm.map_y))

    if pending_lidar_spawns is not None and _confirm_n > 0:
        pending_lidar_spawns[:] = [p for p in pending_lidar_spawns if p.touched_this_cycle]

    # --- Scan verification (presencia laser sin correct) + auditoria dual-tier periodica ---
    _audit_on = bool(person_track_dual_audit_enable)
    _audit_iv = max(1, int(person_track_dual_audit_interval_frames))
    _audit_r = float(person_track_dual_audit_radius_m)
    _yolo_seen = float(person_yolo_seen_green_sec)
    if scan_verify_data is not None and unmatched_tracks:
        scan, tf_tx, tf_ty, tf_yaw = scan_verify_data
        still_unmatched = []
        for ti in unmatched_tracks:
            tr = tracks[ti]
            if tr.class_name != "person":
                still_unmatched.append(ti)
                continue
            audit_due = (
                _audit_on
                and int(getattr(tr, "lidar_audit_counter", 0)) % _audit_iv == 0
            )
            if audit_due and not _person_track_dual_audit_pass(
                tr,
                stamp_sec,
                lidar_meas,
                audit_radius_m=_audit_r,
                audit_min_leg_score=float(scan_verify_nearby_min_leg_score),
                yolo_seen_sec=_yolo_seen,
                corrected_spawn_tier=False,
            ):
                still_unmatched.append(ti)
                continue
            if bool(scan_verify_require_nearby_cluster):
                near_legs = _person_has_nearby_leg_cluster(
                    tr,
                    lidar_meas,
                    float(scan_verify_nearby_cluster_m),
                    float(scan_verify_nearby_min_leg_score),
                )
                if not near_legs:
                    still_unmatched.append(ti)
                    continue
            mx, my = tr.state()
            confirmed = verify_track_in_scan(
                scan,
                mx,
                my,
                tf_tx,
                tf_ty,
                tf_yaw,
                range_tolerance_m=float(scan_verify_range_tolerance_m),
                window_deg=float(scan_verify_window_deg),
            )
            if not confirmed:
                still_unmatched.append(ti)
                continue
            tr.last_seen_sec = stamp_sec
            tr.missed_frames = 0
            tr.consecutive_no_lidar_presence = 0
            tr.consecutive_occlusions = 0
            tr.hits += 1
        unmatched_tracks = still_unmatched

    _miss_debounce = max(0, int(person_missed_debounce_frames))

    for ti in unmatched_tracks:
        tr = tracks[ti]
        # Spencer FOV-aware occlusion: distinguish "behind robot" from "truly missed".
        in_fov = True
        if scan_verify_data is not None:
            scan, tf_tx, tf_ty, tf_yaw = scan_verify_data
            mx, my = tr.state()
            in_fov = is_track_in_lidar_fov(scan, mx, my, tf_tx, tf_ty, tf_yaw)
        if in_fov:
            # Sin cluster ni verify: debounce antes de missed++ (evita parpadeo con require_two).
            tr.consecutive_no_lidar_presence += 1
            if _miss_debounce <= 0 or tr.consecutive_no_lidar_presence > _miss_debounce:
                tr.missed_frames += 1
            tr.consecutive_occlusions = 0
        else:
            # Track is outside sensor FOV (behind robot) → OCCLUDED (Spencer).
            # Don't increment missed_frames — the person might still be there.
            tr.consecutive_occlusions += 1
            tr.consecutive_no_lidar_presence = 0
        # Spencer: copy prediction → posterior (covariance grows naturally).
        tr.occluded_update()
        # Velocity damping only after many missed frames (not occluded ones).
        if tr.missed_frames >= 8:
            tr.zero_velocity()

    if person_dynamic_min_displacement_m > 0.0:
        for tr in tracks:
            if tr.class_name.lower() in dyn_classes:
                span = tr.map_displacement_span_m(stamp_sec, person_dynamic_window_sec)
                tr.motion_dynamic_confirmed = span >= person_dynamic_min_displacement_m
            else:
                tr.motion_dynamic_confirmed = False
    else:
        for tr in tracks:
            tr.motion_dynamic_confirmed = False

    # --- Pruning (Spencer-inspired tiered max_missed + occlusion + maturity) ---
    # Spencer: mature tracks (>100 total matches) get 120 frame tolerance vs 20.
    _MATURE_HIT_THRESHOLD = 50  # lowered from Spencer's 100 (our cycle rate is lower)

    def _occlusion_limit_for_track(tr: Track) -> int:
        if (
            max_occlusions_mature_track > 0
            and tr.hits >= _MATURE_HIT_THRESHOLD
        ):
            return max_occlusions_mature_track
        return max_occlusions_before_deletion

    def _track_max_missed(tr: Track) -> int:
        if tr.class_name != "person":
            return max_missed_frames
        if tr.last_rgb_person_sec > 0.0:
            # Mature person tracks get extra persistence (Spencer: 120 for mature).
            if tr.hits >= _MATURE_HIT_THRESHOLD:
                return max(person_max_missed_frames, int(person_max_missed_frames * 1.5))
            return person_max_missed_frames
        return person_lidar_only_max_missed_frames

    surviving: List[Track] = []
    for tr in tracks:
        if tr.missed_frames > _track_max_missed(tr):
            continue
        # Spencer: occlusion limit (for tracks behind the robot).
        if tr.consecutive_occlusions > _occlusion_limit_for_track(tr):
            continue
        # Spencer Logic Initiator: never-RGB LiDAR person tracks that are static
        # and old enough → prune (wall/furniture), unless they already walked away
        # from where they were born (a real person standing still).
        if (
            tr.class_name == "person"
            and tr.last_rgb_person_sec <= 0.0
            and lidar_initiation_max_age_static_sec > 0.0
            and not (
                lidar_initiation_static_min_travel_m > 0.0
                and tr.max_travel_from_spawn_m > lidar_initiation_static_min_travel_m
            )
        ):
            age = stamp_sec - getattr(tr, "created_sec", stamp_sec)
            if age > lidar_initiation_max_age_static_sec:
                vx, vy = tr.velocity()
                if math.hypot(vx, vy) < 0.10:
                    continue
        surviving.append(tr)
    tracks[:] = surviving
    for tr in tracks:
        tr.append_pose_trail_sample()
    _delete_temporal_duplicate_person_tracks(
        tracks,
        enable=bool(duplicate_track_temporal_enable),
        compare_n=max(2, int(duplicate_track_compare_samples)),
        must_match=max(1, int(duplicate_track_must_match)),
        max_dist_m=float(duplicate_track_max_dist_m),
        lidar_only=bool(duplicate_track_lidar_only),
    )
    _dedup_overlapping_person_tracks_map(
        tracks,
        float(person_track_map_merge_box_m),
        float(person_track_map_merge_min_iou),
        bool(person_track_map_merge_lidar_only),
    )
    return next_track_id


def tracks_to_detection_array(
    tracks: List[Track],
    min_confirmed_hits: int,
    header,
    dynamic_person_suffix_in_class_id: bool = False,
    require_visual_confirmation: bool = True,
    fov_only: bool = False,
    overlay_max_age_sec: float = 0.0,
    frame_sec: float = 0.0,
) -> Detection2DArray:
    """Todos los tracks confirmados (Kalman) como Detection2DArray.

    fov_only=False (default, p.ej. para /detections/tracked): incluye
    tambien tracks solo-LiDAR sin last_bbox (fuera del FOV de la camara,
    o aun no confirmados por YOLO) -- se publican con bbox en blanco
    (0,0)-(0,0), la vista completa de todo lo que el tracker ve.

    fov_only=True (p.ej. para un topic separado tipo /detections/tracked_fov,
    pensado para consumidores que solo saben razonar sobre lo que hay en
    imagen, como vlm_hri): descarta tracks sin last_bbox, y los que tengan
    un last_bbox mas viejo que overlay_max_age_sec (evita "caja congelada"
    de una persona que ya salio de cuadro) -- mismo guard que ya usa
    build_dot_yolo_crop_array para /dot_yolo_crops.
    """
    arr = Detection2DArray()
    arr.header = header
    for tr in tracks:
        if tr.hits < min_confirmed_hits:
            continue
        # Spencer: person tracks sin RGB son tentative → no publicar
        if require_visual_confirmation and tr.class_name == "person" and tr.last_rgb_person_sec <= 0.0:
            continue
        if fov_only:
            if tr.last_bbox is None:
                continue
            if overlay_max_age_sec > 0.0 and tr.last_image_bbox_stamp_sec > 0.0:
                age = frame_sec - tr.last_image_bbox_stamp_sec
                if age > overlay_max_age_sec:
                    continue
        det = Detection2D()
        if tr.last_bbox is not None:
            x1, y1, x2, y2 = tr.last_bbox
            det.bbox.center.position.x = float((x1 + x2) / 2.0)
            det.bbox.center.position.y = float((y1 + y2) / 2.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

        hyp = ObjectHypothesisWithPose()
        cname = tr.class_name
        if (
            dynamic_person_suffix_in_class_id
            and cname.lower() == "person"
            and getattr(tr, "motion_dynamic_confirmed", False)
        ):
            cname = "person_dyn"
        hyp.hypothesis.class_id = f"{cname}#{tr.track_id}"
        hyp.hypothesis.score = float(tr.score)
        map_x, map_y = tr.state()
        hyp.pose.pose.position.x = float(map_x)
        hyp.pose.pose.position.y = float(map_y)
        det.results.append(hyp)
        arr.detections.append(det)
    return arr
