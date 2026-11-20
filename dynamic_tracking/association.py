"""Hungarian (globally optimal) association of measurements <-> tracks.

Uses scipy.optimize.linear_sum_assignment (Jonker-Volgenant O(n³)).
Falls back to a greedy sort if scipy is not available.
"""

import math
from typing import Callable, List, Optional, Tuple

import numpy as np

from .types import Track

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def assoc_class_key(class_name: str) -> int:
    """Process-stable hash to match YOLO class <-> track."""
    return hash(str(class_name).strip().lower())


def measurement_cost(track_x: float, track_y: float, meas_x: float, meas_y: float) -> float:
    dx = track_x - meas_x
    dy = track_y - meas_y
    return math.sqrt(dx * dx + dy * dy)


# Spencer: chi²(99%, 2 DOF) gate for Mahalanobis association.
CHI2_99_2DOF = 9.21

_BIG_COST = 1e6


def assign_measurements(
    tracks: List[Track],
    track_indices: List[int],
    measurements: List[Tuple[float, float, Optional[int]]],
    assoc_threshold: float,
    per_track_max_cost: Optional[Callable[[Track], float]] = None,
    pair_allowed: Optional[Callable[[Track, float, float], bool]] = None,
    use_mahalanobis: bool = False,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Globally optimal (Hungarian) or greedy association with gating.

    measurements: (map_x, map_y, optional class hash).
    use_mahalanobis: if True, use Mahalanobis distance with chi² gate (Spencer).
    """
    n_tracks = len(track_indices)
    n_meas = len(measurements)
    if n_tracks == 0 or n_meas == 0:
        return [], list(track_indices), list(range(n_meas))

    cost_matrix = np.full((n_tracks, n_meas), _BIG_COST, dtype=np.float64)
    threshold_matrix = np.empty(n_tracks, dtype=np.float64)

    for row, ti in enumerate(track_indices):
        tr = tracks[ti]
        if use_mahalanobis:
            thr = CHI2_99_2DOF
        else:
            thr = per_track_max_cost(tr) if per_track_max_cost is not None else assoc_threshold
        threshold_matrix[row] = thr
        tr_x, tr_y = tr.association_position()
        for col, (m_x, m_y, class_hash) in enumerate(measurements):
            if class_hash is not None and tr.class_name != "unknown" and assoc_class_key(tr.class_name) != class_hash:
                continue
            if pair_allowed is not None and not pair_allowed(tr, m_x, m_y):
                continue
            if use_mahalanobis:
                c = tr.mahalanobis_sq(m_x, m_y)
            else:
                c = measurement_cost(tr_x, tr_y, m_x, m_y)
            if c <= thr:
                cost_matrix[row, col] = c

    if _HAS_SCIPY:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
    else:
        row_idx, col_idx = _greedy_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    used_t = set()
    used_m = set()
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] >= _BIG_COST:
            continue
        ti = track_indices[r]
        matches.append((ti, c))
        used_t.add(ti)
        used_m.add(c)

    unmatched_tracks = [ti for ti in track_indices if ti not in used_t]
    unmatched_meas = [mi for mi in range(n_meas) if mi not in used_m]
    return matches, unmatched_tracks, unmatched_meas


def _greedy_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy fallback if scipy is not available."""
    n_rows, n_cols = cost_matrix.shape
    pairs = []
    for r in range(n_rows):
        for c in range(n_cols):
            if cost_matrix[r, c] < _BIG_COST:
                pairs.append((cost_matrix[r, c], r, c))
    pairs.sort(key=lambda x: x[0])
    used_r = set()
    used_c = set()
    row_out = []
    col_out = []
    for _, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        row_out.append(r)
        col_out.append(c)
    return np.array(row_out, dtype=int), np.array(col_out, dtype=int)
