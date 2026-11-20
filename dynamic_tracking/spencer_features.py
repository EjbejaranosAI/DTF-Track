"""Port of Spencer srl_laser_features to Python/NumPy.

Computes the same 54 feature dimensions used by the pre-trained Random Forest
model shipped with ``srl_laser_detectors``.  Each ``FeatureXX`` class mirrors
the identically-named C++ class; dimension indices and string descriptions are
kept identical so that the saved OpenCV ``cv::ml::RTrees`` model can be used
directly.

References
----------
Timm Linder, Social Robotics Lab, University of Freiburg.
srl_laser_features (BSD-3 license).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Segment data structure (mirrors srl_laser_features/segment.h)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A contiguous group of laser hits extracted from one LaserScan."""

    points: np.ndarray  # (N,2) cartesian [x, y]
    ranges: np.ndarray  # (N,) metric distances
    indices: np.ndarray  # (N,) original scan indices (uint)

    mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    median: np.ndarray = field(default_factory=lambda: np.zeros(2))

    preceding_point: Optional[np.ndarray] = None  # last hit of previous segment
    succeeding_point: Optional[np.ndarray] = None  # first hit of next segment

    mean_angle: float = 0.0
    min_angle: float = 0.0
    max_angle: float = 0.0
    angular_distance_to_closest_boundary: float = 0.0


# ---------------------------------------------------------------------------
# Individual features (F00 – F16, F33)
# Each returns an np.ndarray whose length equals the number of dimensions.
# ---------------------------------------------------------------------------

_RNG00 = random.Random(666)


def f00(seg: Segment) -> np.ndarray:
    """F00 – Random coin (baseline noise feature)."""
    return np.array([float(_RNG00.randint(0, 1))], dtype=np.float32)


def f01(seg: Segment) -> np.ndarray:
    """F01 – Number of points."""
    return np.array([float(len(seg.points))], dtype=np.float32)


def f02(seg: Segment) -> np.ndarray:
    """F02 – Mean and std of consecutive range differences."""
    n = len(seg.ranges)
    if n < 2:
        return np.array([-1.0, -1.0], dtype=np.float32)
    diffs = np.abs(np.diff(seg.ranges))
    m = float(np.mean(diffs))
    s = float(np.std(diffs, ddof=1)) if n >= 3 else -1.0
    return np.array([m, s], dtype=np.float32)


def f03(seg: Segment) -> np.ndarray:
    """F03 – RMS spread about centroid (mean)."""
    n = len(seg.points)
    if n < 2:
        return np.array([0.0], dtype=np.float32)
    d = np.linalg.norm(seg.points - seg.mean, axis=1)
    return np.array([float(np.sqrt(np.mean(d ** 2)))], dtype=np.float32)


def f04(seg: Segment) -> np.ndarray:
    """F04 – RMS spread about median."""
    n = len(seg.points)
    if n < 2:
        return np.array([0.0], dtype=np.float32)
    d = np.linalg.norm(seg.points - seg.median, axis=1)
    return np.array([float(np.sqrt(np.mean(d ** 2)))], dtype=np.float32)


def f05(seg: Segment) -> np.ndarray:
    """F05 – Width (chord), min/max edge, ratio (extended=True, 4 dims)."""
    pts = seg.points
    n = len(pts)
    if n < 2:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    width = float(np.linalg.norm(pts[-1] - pts[0]))
    edges = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mn = float(np.min(edges))
    mx = float(np.max(edges))
    ratio = mn / mx if mx > 0 else 0.0
    return np.array([width, mn, mx, ratio], dtype=np.float32)


def _circle_fit(pts: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic least-squares circle fit. Returns (cx, cy, r)."""
    n = len(pts)
    if n < 3:
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        return cx, cy, 0.0
    A = np.column_stack([-2.0 * pts[:, 0], -2.0 * pts[:, 1], np.ones(n)])
    b = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    try:
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        return cx, cy, 0.0
    xc, yc, c = float(res[0]), float(res[1]), float(res[2])
    r2 = xc * xc + yc * yc - c
    r = math.sqrt(max(r2, 0.0))
    return xc, yc, r


def f06(seg: Segment) -> np.ndarray:
    """F06 – Circle fit (extended=True, 7 dims)."""
    pts = seg.points
    n = len(pts)
    if n < 3:
        return np.zeros(7, dtype=np.float32)
    cx, cy, r = _circle_fit(pts)
    center = np.array([cx, cy])
    dists = np.linalg.norm(pts - center, axis=1)
    residuals = dists - r
    sq_res = residuals ** 2
    circularity = float(np.sum(sq_res))
    mean_sq = float(np.mean(sq_res))
    min_diff = float(np.min(np.abs(residuals)))
    max_diff = float(np.max(np.abs(residuals)))
    ratio = min_diff / max_diff if max_diff > 0 else 0.0
    # dim 6: "Area of rectangles" – never computed in Spencer C++ (stays 0)
    return np.array(
        [circularity, r, mean_sq, min_diff, max_diff, ratio, 0.0],
        dtype=np.float32,
    )


def f07(seg: Segment) -> np.ndarray:
    """F07 – Boundary length, perimeter, regularity (3 dims)."""
    pts = seg.points
    n = len(pts)
    if n < 2:
        return np.zeros(3, dtype=np.float32)
    edges = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    boundary = float(np.sum(edges))
    closing = float(np.linalg.norm(pts[-1] - pts[0]))
    perimeter = boundary + closing
    regularity = math.sqrt(perimeter / max(n - 2, 1)) if n > 2 else 0.0
    return np.array([boundary, perimeter, regularity], dtype=np.float32)


def f08(seg: Segment) -> np.ndarray:
    """F08 – Mean curvature (Menger) and mean angular difference (2 dims)."""
    pts = seg.points
    n = len(pts)
    if n < 3:
        return np.zeros(2, dtype=np.float32)
    curvatures: List[float] = []
    angles: List[float] = []
    for i in range(1, n - 1):
        v0 = pts[i] - pts[i - 1]
        v1 = pts[i + 1] - pts[i]
        d0 = float(np.linalg.norm(v0))
        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(pts[i + 1] - pts[i - 1]))
        if d0 < 1e-12 or d1 < 1e-12 or d2 < 1e-12:
            continue
        # Menger curvature = 4*A / (d0*d1*d2), A via Heron's formula
        s = (d0 + d1 + d2) * 0.5
        area2 = s * (s - d0) * (s - d1) * (s - d2)
        area = math.sqrt(max(area2, 0.0))
        curvatures.append(4.0 * area / (d0 * d1 * d2))
        cos_a = np.dot(v0, v1) / (d0 * d1)
        cos_a = max(-1.0, min(1.0, cos_a))
        angles.append(math.acos(cos_a))
    mc = float(np.mean(curvatures)) if curvatures else 0.0
    ma = float(np.mean(angles)) if angles else 0.0
    return np.array([mc, ma], dtype=np.float32)


def f09(seg: Segment) -> np.ndarray:
    """F09 – Aspect ratio (anisotropy of spread)."""
    pts = seg.points
    n = len(pts)
    if n < 2:
        return np.array([0.0], dtype=np.float32)
    diff = pts - seg.mean
    sx = float(np.std(diff[:, 0], ddof=1)) if n > 1 else 0.0
    sy = float(np.std(diff[:, 1], ddof=1)) if n > 1 else 0.0
    mn, mx = min(sx, sy), max(sx, sy)
    ratio = (1.0 + mn) / (1.0 + mx)
    return np.array([ratio], dtype=np.float32)


def f10(seg: Segment) -> np.ndarray:
    """F10 – Line fit (extended=True, 5 dims)."""
    pts = seg.points
    n = len(pts)
    if n < 2:
        return np.zeros(5, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    mx_val = float(np.mean(x))
    sxx = float(np.sum((x - mx_val) ** 2))
    if sxx < 1e-12:
        # Nearly vertical line: use x = const model
        residuals = x - mx_val
    else:
        A = np.column_stack([x, np.ones(n)])
        try:
            res, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return np.zeros(5, dtype=np.float32)
        m, q = float(res[0]), float(res[1])
        residuals = m * x + q - y
    rss = float(np.sum(residuals ** 2))
    mse = float(np.mean(residuals ** 2))
    min_r = float(np.min(residuals))
    max_r = float(np.max(residuals))
    abs_min = abs(min_r)
    abs_max = abs(max_r)
    ratio = abs_min / abs_max if abs_max > 0 else 0.0
    return np.array([rss, mse, min_r, max_r, ratio], dtype=np.float32)


def _bspline_eval(ctrl: np.ndarray, degree: int, num_samples: int) -> np.ndarray:
    """Evaluate uniform B-spline with given control points."""
    n_ctrl = len(ctrl)
    if n_ctrl <= degree:
        return ctrl.copy()
    n_knots = n_ctrl + degree + 1
    knots = np.linspace(0.0, 1.0, n_knots)
    samples = np.zeros((num_samples, 2))
    t_min = knots[degree]
    t_max = knots[n_ctrl]
    if t_max <= t_min:
        return ctrl.copy()
    ts = np.linspace(t_min, t_max - 1e-10, num_samples)
    for si, t in enumerate(ts):
        # de Boor's algorithm
        k = int(np.searchsorted(knots, t, side="right")) - 1
        k = max(degree, min(k, n_ctrl - 1))
        d = [ctrl[j].copy() for j in range(max(0, k - degree), min(n_ctrl, k + 1))]
        offset = max(0, k - degree)
        for r in range(1, degree + 1):
            for j in range(len(d) - 1, r - 1, -1):
                idx = j + offset
                left = knots[idx + 1 + degree - r] if (idx + 1 + degree - r) < n_knots else knots[-1]
                right = knots[idx] if idx < n_knots else knots[-1]
                denom = left - right
                if abs(denom) < 1e-15:
                    alpha = 0.0
                else:
                    alpha = (t - right) / denom
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
        samples[si] = d[-1]
    return samples


def f11(seg: Segment) -> np.ndarray:
    """F11 – B-spline approximation error (quadratic=2, cubic=3). 2 dims."""
    pts = seg.points
    n = len(pts)
    if n < 4:
        return np.zeros(2, dtype=np.float32)
    errs = []
    for deg in (2, 3):
        samples = _bspline_eval(pts, deg, n)
        dists = np.linalg.norm(pts - samples[:n], axis=1)
        errs.append(float(np.mean(dists)))
    return np.array(errs, dtype=np.float32)


def f12(seg: Segment) -> np.ndarray:
    """F12 – Distance from sensor to segment center."""
    return np.array([float(np.linalg.norm(seg.mean))], dtype=np.float32)


def f13(seg: Segment) -> np.ndarray:
    """F13 – Shoelace polygon area."""
    pts = seg.points
    n = len(pts)
    if n < 3:
        return np.array([0.0], dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    return np.array([area], dtype=np.float32)


def f14(seg: Segment) -> np.ndarray:
    """F14 – Range span + local minima counts at 14 noise levels (16 dims)."""
    r = seg.ranges
    n = len(r)
    rmin = float(np.min(r)) if n > 0 else 0.0
    rmax = float(np.max(r)) if n > 0 else 0.0
    diff = rmax - rmin
    ratio = rmin / rmax if rmax > 0 else 0.0

    minima_counts = np.zeros(14, dtype=np.float32)
    if n >= 3:
        for k in range(14):
            noise = 0.01 * (k + 1)
            count = 0
            for i in range(1, n - 1):
                if r[i] < r[i - 1] - noise and r[i] < r[i + 1] - noise:
                    count += 1
            minima_counts[k] = float(count)

    result = np.zeros(16, dtype=np.float32)
    result[0] = diff
    result[1:15] = minima_counts
    result[15] = ratio
    return result


def f16(seg: Segment) -> np.ndarray:
    """F16 – Angular placement in FOV (4 dims)."""
    return np.array(
        [
            seg.angular_distance_to_closest_boundary,
            seg.mean_angle,
            seg.min_angle,
            seg.max_angle,
        ],
        dtype=np.float32,
    )


def f33(seg: Segment) -> np.ndarray:
    """F33 – Gaps to neighboring segments (2 dims)."""
    pts = seg.points
    if seg.preceding_point is not None:
        d_prev = float(np.linalg.norm(pts[0] - seg.preceding_point))
    else:
        d_prev = -1.0
    if seg.succeeding_point is not None:
        d_next = float(np.linalg.norm(pts[-1] - seg.succeeding_point))
    else:
        d_next = -1.0
    return np.array([d_prev, d_next], dtype=np.float32)


# Ordered list of feature functions matching the model's expected ordering:
# F00(1) + F01(1) + F02(2) + F03(1) + F04(1) + F05(4) + F06(7) + F07(3)
# + F08(2) + F09(1) + F10(5) + F11(2) + F12(1) + F13(1) + F14(16) + F16(4) + F33(2) = 54
_FEATURE_FUNCS = [f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13, f14, f16, f33]

TOTAL_DIMS = 54


def compute_feature_vector(seg: Segment) -> np.ndarray:
    """Compute all 54 feature dimensions for one segment (noetic RF models)."""
    parts = [fn(seg) for fn in _FEATURE_FUNCS]
    vec = np.concatenate(parts)
    assert len(vec) == TOTAL_DIMS, f"Expected {TOTAL_DIMS} dims, got {len(vec)}"
    return vec.astype(np.float32)


def feature_values_by_name(seg: Segment) -> dict:
    """Map Spencer ``used_feature_dimensions`` labels to scalar values (icra16 models)."""
    f5 = f05(seg)
    f6 = f06(seg)
    f7 = f07(seg)
    f8 = f08(seg)
    f9 = f09(seg)
    f10v = f10(seg)
    f11v = f11(seg)
    f12v = f12(seg)
    f13v = f13(seg)
    f14v = f14(seg)
    f16v = f16(seg)
    f33v = f33(seg)
    f2 = f02(seg)
    return {
        "Throw a coin.": float(f00(seg)[0]),
        "Number of laser end points.": float(f01(seg)[0]),
        "Mean of difference in range of consecutive beams.": float(f2[0]),
        "Variance of difference in range of consecutive beams.": float(f2[1]),
        "Variance from center of gravity.": float(f03(seg)[0]),
        "Mean average deviation from median.": float(f04(seg)[0]),
        "Width.": float(f5[0]),
        "Min distance between points.": float(f5[1]),
        "Max distance between points.": float(f5[2]),
        "Ratio between min and max distance between points.": float(f5[3]),
        "Circularity.": float(f6[0]),
        "Radius of fitted circle.": float(f6[1]),
        "Variance of distance from points to center of circle.": float(f6[2]),
        "Minimal difference to radius.": float(f6[3]),
        "Maximal difference to radius.": float(f6[4]),
        "Ratio between min and max difference.": float(f6[5]),
        "Area of rectangles to fitted circle center.": float(f6[6]),
        "Boundary length.": float(f7[0]),
        "Perimeter.": float(f7[1]),
        "Contour regularity.": float(f7[2]),
        "Mean curvature.": float(f8[0]),
        "Mean angular difference.": float(f8[1]),
        "Aspect ratio.": float(f9[0]),
        "Linearity.": float(f10v[0]),
        "Linearity mean error.": float(f10v[1]),
        "Linearity min error.": float(f10v[2]),
        "Linearity max error.": float(f10v[3]),
        "Linearity ratio of min and max error.": float(f10v[4]),
        "Error of fitted quadratic b-spline.": float(f11v[0]),
        "Error of fitted cubic b-spline.": float(f11v[1]),
        "Distance.": float(f12v[0]),
        "Polygon Area.": float(f13v[0]),
        "Difference of longest and shortest beam.": float(f14v[0]),
        **{
            f"Number of local minima (noise {0.01 * (k + 1):.2f}).": float(f14v[1 + k])
            for k in range(14)
        },
        "Ratio between shortest and longest beam.": float(f14v[15]),
        "Jump distance to previous segment.": float(f33v[0]),
        "Jump distance to next segment.": float(f33v[1]),
    }


def read_model_feature_dimension_names(model_path: str) -> List[str]:
    """Parse ``used_feature_dimensions`` from a Spencer FileStorage model."""
    import cv2

    fs = cv2.FileStorage(str(model_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return []
    try:
        node = fs.getNode("used_feature_dimensions")
        if node.empty():
            return []
        return [str(node.at(i).string()) for i in range(node.size())]
    finally:
        fs.release()


def compute_feature_vector_for_names(seg: Segment, names: List[str]) -> np.ndarray:
    """Feature row for legacy icra16 models (subset/order from training)."""
    lut = feature_values_by_name(seg)
    row = np.array([lut.get(n, -1.0) for n in names], dtype=np.float32)
    return row


# ---------------------------------------------------------------------------
# Segment extraction from LaserScan
# ---------------------------------------------------------------------------

def build_segments_from_scan(
    angles: np.ndarray,
    ranges: np.ndarray,
    cluster_indices: List[List[int]],
    angle_min: float,
    angle_max: float,
) -> List[Optional[Segment]]:
    """Build ``Segment`` objects from pre-clustered scan indices.

    The result is aligned with ``cluster_indices``: element i describes cluster i,
    or is None when that cluster has < 2 points. Callers index it by cluster
    position, so skipped clusters must not shift the following ones.

    Parameters
    ----------
    angles : (N,) array of beam angles for the full scan.
    ranges : (N,) array of beam ranges for the full scan.
    cluster_indices : List of lists of scan indices per cluster.
    angle_min, angle_max : FOV boundaries of the laser.
    """
    aligned: List[Optional[Segment]] = []
    segments: List[Segment] = []
    for idx_list in cluster_indices:
        if len(idx_list) < 2:
            aligned.append(None)
            continue
        idx = np.array(idx_list, dtype=np.intp)
        a = angles[idx]
        r = ranges[idx]
        # Spencer convention: y = -sin(phi)*rho (mirror about x-axis)
        xs = r * np.cos(a)
        ys = -r * np.sin(a)
        pts = np.column_stack([xs, ys])
        mean = np.mean(pts, axis=0)
        median = np.median(pts, axis=0)
        # Angular features computed from cartesian (Spencer style)
        seg = Segment(
            points=pts,
            ranges=r,
            indices=idx,
            mean=mean,
            median=median,
            mean_angle=float(np.arctan2(mean[1], mean[0])),
            min_angle=float(np.min(np.arctan2(ys, xs))),
            max_angle=float(np.max(np.arctan2(ys, xs))),
        )
        d_lo = abs(seg.min_angle - np.arctan2(-np.sin(angle_min), np.cos(angle_min)))
        d_hi = abs(np.arctan2(-np.sin(angle_max), np.cos(angle_max)) - seg.max_angle)
        seg.angular_distance_to_closest_boundary = min(abs(d_lo), abs(d_hi))
        segments.append(seg)
        aligned.append(seg)

    # Fill preceding/succeeding points (neighbor context for Feature33)
    for i, seg in enumerate(segments):
        if i > 0:
            seg.preceding_point = segments[i - 1].points[-1].copy()
        if i < len(segments) - 1:
            seg.succeeding_point = segments[i + 1].points[0].copy()

    return aligned
