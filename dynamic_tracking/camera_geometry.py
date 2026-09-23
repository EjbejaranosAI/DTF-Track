"""Camera intrinsics, bearing, projection, and polar geometry (extracted from node)."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sensor_msgs.msg import CameraInfo

from .tf_utils import TFHelper
from .types import Track, YoloMeasurement


def intrinsics_fov_fallback(fov_rad: float, w: int, h: int) -> Tuple[float, float, float, float]:
    """(fx, fy, cx, cy) from horizontal FOV when CameraInfo is unavailable."""
    fx = (w * 0.5) / max(math.tan(fov_rad * 0.5), 1e-6)
    return fx, fx, float(w) * 0.5, float(h) * 0.5


def scaled_intrinsics(
    ci: Optional[CameraInfo], w: int, h: int, fov_rad: float,
) -> Tuple[float, float, float, float]:
    """K scaled to actual image size.  Falls back to FOV if CameraInfo is invalid."""
    if ci is not None and len(ci.k) >= 9:
        iw0, ih0 = float(ci.width), float(ci.height)
        if iw0 >= 1 and ih0 >= 1:
            sx, sy = float(w) / iw0, float(h) / ih0
            fx = float(ci.k[0]) * sx
            fy = float(ci.k[4]) * sy
            cx = float(ci.k[2]) * sx
            cy = float(ci.k[5]) * sy
            if fx >= 1 and fy >= 1:
                return fx, fy, cx, cy
    return intrinsics_fov_fallback(fov_rad, w, h)


# ── Bearing / polar from optical ray ────────────────────────────────

def bearing_laser_from_optical_ray(
    u: float,
    v: float,
    w: int,
    h: int,
    stamp,
    optical_frame: str,
    laser_frame: str,
    ci: Optional[CameraInfo],
    fov_rad: float,
    angle_offset_rad: float,
    tf: TFHelper,
    extra_timeout: float = 0.25,
) -> Optional[float]:
    """Angle (rad) in laser frame of a pixel (u,v) projected through optical→laser TF."""
    if not laser_frame or not optical_frame:
        return None
    fx, fy, cx, cy = scaled_intrinsics(ci, w, h, fov_rad)
    nx = (float(u) - cx) / max(fx, 1e-6)
    ny = (float(v) - cy) / max(fy, 1e-6)
    nz = 1.0
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    d_opt = np.array([nx / norm, ny / norm, nz / norm], dtype=np.float64)

    rt = tf.lookup_R_t_cached(laser_frame, optical_frame, stamp)
    if rt is None:
        rt = tf.lookup_R_t(laser_frame, optical_frame, stamp)
    if rt is None:
        rt = tf.lookup_R_t(laser_frame, optical_frame, None, extra_timeout=extra_timeout)
    if rt is None:
        return None

    R, tvec = rt
    d_l = R @ d_opt
    o_l = tvec
    if abs(float(d_l[2])) > 1e-5:
        s = -float(o_l[2]) / float(d_l[2])
        if s > 0:
            px = float(o_l[0] + s * d_l[0])
            py = float(o_l[1] + s * d_l[1])
            return math.atan2(py, px) + angle_offset_rad
    horiz = math.hypot(float(d_l[0]), float(d_l[1]))
    if horiz < 1e-9:
        return None
    return math.atan2(float(d_l[1]), float(d_l[0])) + angle_offset_rad


# ── Bearing / polar from track ground point ──────────────────────────

def bearing_laser_from_track(
    tx: float, ty: float, stamp,
    laser_frame: str, tracking_frame: str,
    angle_offset_rad: float,
    tf: TFHelper,
) -> Optional[float]:
    """atan2(py, px) in laser frame for (tx, ty, 0) in tracking_frame."""
    rt = tf.laser_from_tracking_cached(laser_frame, tracking_frame, stamp)
    if rt is None:
        return None
    R, tvec = rt
    pl = R @ np.array([tx, ty, 0.0], dtype=np.float64) + tvec
    return math.atan2(float(pl[1]), float(pl[0])) + angle_offset_rad


def laser_polar_from_track(
    tx: float, ty: float, stamp,
    laser_frame: str, tracking_frame: str,
    tf: TFHelper,
) -> Optional[Tuple[float, float]]:
    """(range_m, angle_rad) in scan plane."""
    rt = tf.laser_from_tracking_cached(laser_frame, tracking_frame, stamp)
    if rt is None:
        return None
    R, tvec = rt
    pl = R @ np.array([tx, ty, 0.0], dtype=np.float64) + tvec
    x, y = float(pl[0]), float(pl[1])
    return math.hypot(x, y), math.atan2(y, x)


def all_tracks_laser_polar(
    tracks: list,
    stamp,
    laser_frame: str,
    tracking_frame: str,
    tf: TFHelper,
) -> Dict[int, Tuple[float, float]]:
    """Polar (r, θ) in laser frame for every track — 1 TF + N multiplications."""
    out: Dict[int, Tuple[float, float]] = {}
    rt = tf.laser_from_tracking_cached(laser_frame, tracking_frame, stamp)
    if rt is None:
        return out
    R, tvec = rt
    for i, tr in enumerate(tracks):
        tx, ty = tr.state()
        pl = R @ np.array([tx, ty, 0.0], dtype=np.float64) + tvec
        x, y = float(pl[0]), float(pl[1])
        out[i] = (math.hypot(x, y), math.atan2(y, x))
    return out


# ── Project tracks to image (u,v) ───────────────────────────────────

def project_tracks_to_image(
    tracks: List[Track],
    w: int,
    h: int,
    optical_frame: str,
    stamp,
    ci: Optional[CameraInfo],
    projection_height_m: float,
    tracking_frame: str,
    tf: TFHelper,
    extra_timeout: float = 0.25,
) -> Dict[int, Tuple[float, float]]:
    """Returns {track_idx: (u, v)} for tracks visible in image."""
    if ci is None or len(ci.k) < 9:
        return {}
    iw0, ih0 = float(ci.width), float(ci.height)
    if iw0 < 1 or ih0 < 1:
        return {}
    sx, sy = float(w) / iw0, float(h) / ih0
    fx = float(ci.k[0]) * sx
    fy = float(ci.k[4]) * sy
    cx_cam = float(ci.k[2]) * sx
    cy_cam = float(ci.k[5]) * sy
    if fx < 1 or fy < 1:
        return {}

    rt = tf.lookup_R_t(optical_frame, tracking_frame, stamp, extra_timeout=extra_timeout)
    if rt is None:
        return {}
    R_co, t_co = rt

    hz_main = max(0.0, projection_height_m)
    z_candidates = _unique_heights([hz_main, 0.55 * hz_main if hz_main > 1e-6 else 0.45, 0.45, 0.0])
    result: Dict[int, Tuple[float, float]] = {}
    for ti, tr in enumerate(tracks):
        tx, ty = tr.state()
        for hz in z_candidates:
            p_cam = R_co @ np.array([tx, ty, hz], dtype=np.float64) + t_co
            z = float(p_cam[2])
            if z < 0.12:
                continue
            u = fx * float(p_cam[0]) / z + cx_cam
            v = fy * float(p_cam[1]) / z + cy_cam
            if 0 <= u < w and 0 <= v < h:
                result[ti] = (u, v)
                break
    return result


def _unique_heights(vals) -> List[float]:
    seen: set = set()
    out: List[float] = []
    for h in vals:
        key = round(h, 4)
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out
