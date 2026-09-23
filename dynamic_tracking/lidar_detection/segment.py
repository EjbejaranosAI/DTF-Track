"""Scan segmentation (Spencer jump-distance parity)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from sensor_msgs.msg import LaserScan

from ..scan_clustering import (
    LegHeuristicParams,
    cluster_pose_laser_xy,
    spencer_style_segment_prefilter,
)


def segment_scan(scan: LaserScan, jump_m: float) -> List[List[int]]:
    """Jump-distance segmentation. Returns list of beam index lists."""
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
        if prev_range is not None and abs(value - prev_range) > jump_m:
            if current:
                segments.append(current)
                current = []
        current.append(i)
        prev_range = value
    if current:
        segments.append(current)
    return segments


def cluster_from_indices(
    scan: LaserScan, idx_list: List[int]
) -> List[Tuple[float, float]]:
    return [
        (scan.angle_min + i * scan.angle_increment, float(scan.ranges[i]))
        for i in idx_list
    ]


def prepare_scan_arrays(scan: LaserScan) -> Tuple[np.ndarray, np.ndarray, int]:
    n_beams = len(scan.ranges)
    angles = np.array(
        [scan.angle_min + i * scan.angle_increment for i in range(n_beams)],
        dtype=np.float64,
    )
    ranges = np.array(scan.ranges, dtype=np.float64)
    return angles, ranges, n_beams


def passes_segment_prefilter(
    cluster: List[Tuple[float, float]],
    *,
    min_points: int,
    max_points: int,
    min_range_m: float,
    max_range_m: float,
    min_segment_width_m: float,
) -> bool:
    return spencer_style_segment_prefilter(
        cluster,
        min_points,
        max_points,
        min_range_m,
        max_range_m,
        min_segment_width_m,
    )


def segment_pose_and_distance(
    cluster: List[Tuple[float, float]],
    pose_mode: str,
) -> Tuple[float, float, float]:
    return cluster_pose_laser_xy(cluster, pose_mode)


# Re-export for detectors that need leg heuristics params type
__all__ = [
    "LegHeuristicParams",
    "cluster_from_indices",
    "passes_segment_prefilter",
    "prepare_scan_arrays",
    "segment_pose_and_distance",
    "segment_scan",
]
