"""Associate two leg-like LiDAR clusters to a person track (for pose_estimation)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .association import measurement_cost
from .types import LidarMeasurement


def leg_pair_for_track(
    map_x: float,
    map_y: float,
    lidar_meas: Sequence[LidarMeasurement],
    *,
    min_leg_score: float = 0.58,
    max_dist_m: float = 0.9,
    min_separation_m: float = 0.14,
    max_separation_m: float = 0.72,
) -> Tuple[bool, float, float, float, float]:
    """
    Find two distinct leg clusters near ``(map_x, map_y)``.

    Returns (valid, left_x, left_y, right_x, right_y) in ``map``.
    Left/right: smaller bearing from track centroid = "left" (arbitrary but stable).
    """
    mx, my = float(map_x), float(map_y)
    cands: List[Tuple[float, float, float, float]] = []
    for lm in lidar_meas:
        sc = float(lm.leg_candidate_score)
        if sc < float(min_leg_score):
            continue
        lx, ly = float(lm.map_x), float(lm.map_y)
        d = measurement_cost(mx, my, lx, ly)
        if d <= float(max_dist_m):
            cands.append((sc, lx, ly, d))

    if len(cands) < 2:
        return False, 0.0, 0.0, 0.0, 0.0

    cands.sort(key=lambda t: (-t[0], t[3]))

    best: Optional[Tuple[float, float, float, float]] = None
    best_sep = -1.0
    for i in range(len(cands)):
        _, x1, y1, _ = cands[i]
        for j in range(i + 1, len(cands)):
            _, x2, y2, _ = cands[j]
            sep = math.hypot(x2 - x1, y2 - y1)
            if sep < float(min_separation_m) or sep > float(max_separation_m):
                continue
            if sep > best_sep:
                best_sep = sep
                best = (x1, y1, x2, y2)

    if best is None:
        return False, 0.0, 0.0, 0.0, 0.0

    x1, y1, x2, y2 = best
    a1 = math.atan2(y1 - my, x1 - mx)
    a2 = math.atan2(y2 - my, x2 - mx)
    if a1 <= a2:
        return True, x1, y1, x2, y2
    return True, x2, y2, x1, y1


def merge_leg_pairs(
    meas: Sequence[LidarMeasurement],
    max_separation_m: float,
) -> List[LidarMeasurement]:
    """Merge the two legs of one person into a single measurement at their midpoint.

    Each LiDAR measurement is one leg, so a walking person alternates left/right
    and the track jitters (and the second leg can open its own ID). Pairs are
    formed greedily, closest first, between measurements at most
    ``max_separation_m`` apart (TIAGo data: one person's legs 0.16-0.57 m, 100 %
    <= 0.6 m). Unpaired measurements (one leg visible) are kept as they are.
    """
    n = len(meas)
    if max_separation_m <= 0.0 or n < 2:
        return list(meas)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(meas[i].map_x - meas[j].map_x, meas[i].map_y - meas[j].map_y)
            if d <= max_separation_m:
                pairs.append((d, i, j))
    if not pairs:
        return list(meas)
    pairs.sort()
    used = [False] * n
    out: List[LidarMeasurement] = []
    for _, i, j in pairs:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        a, b = meas[i], meas[j]
        out.append(
            LidarMeasurement(
                map_x=0.5 * (a.map_x + b.map_x),
                map_y=0.5 * (a.map_y + b.map_y),
                leg_candidate_score=max(a.leg_candidate_score, b.leg_candidate_score),
                spawn_tier_pass=a.spawn_tier_pass or b.spawn_tier_pass,
            )
        )
    out.extend(m for k, m in enumerate(meas) if not used[k])
    return out
