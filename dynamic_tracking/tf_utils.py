"""Unified TF helpers — replaces 6 variants of try/except TF lookup in the node."""

import math
import time
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformException

from .geometry import quat_to_rotmat


class TFHelper:
    """Thin wrapper around tf2 Buffer that caches lookups and provides fallbacks.

    All public methods return ``None`` on failure (no exceptions leak).
    """

    def __init__(self, tf_buffer: Buffer, timeout_sec: float = 0.05) -> None:
        self.buf = tf_buffer
        self.timeout_sec = timeout_sec
        self._last_good: Dict[str, Tuple[float, float, float, float]] = {}
        self._stale_max_sec = 2.0

        # Per-frame caches (call ``reset_frame_caches()`` at the top of each frame)
        self._planar_key: Optional[Tuple[str, int, int]] = None
        self._planar_val: Optional[Tuple[float, float, float]] = None
        self._rt_key: Optional[Tuple[str, str, int, int]] = None
        self._rt_val: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._laser_track_key: Optional[Tuple[str, int, int]] = None
        self._laser_track_val: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def reset_frame_caches(self) -> None:
        self._planar_key = None
        self._planar_val = None
        self._rt_key = None
        self._rt_val = None
        self._laser_track_key = None
        self._laser_track_val = None

    # ------------------------------------------------------------------
    # Low-level: single lookup
    # ------------------------------------------------------------------
    @staticmethod
    def stamp_msg(stamp):
        """Convert any stamp-like to ``builtin_interfaces.msg.Time``."""
        if stamp is None:
            from builtin_interfaces.msg import Time as BuiltinTime
            return BuiltinTime()
        if hasattr(stamp, "to_msg"):
            return stamp.to_msg()
        return stamp

    def _raw_lookup(self, target: str, source: str, stamp, timeout_sec: Optional[float] = None):
        """Single TF lookup, returns TransformStamped or None."""
        tout = Duration(seconds=timeout_sec if timeout_sec is not None else self.timeout_sec)
        try:
            return self.buf.lookup_transform(target, source, self.stamp_msg(stamp), timeout=tout)
        except TransformException:
            return None

    # ------------------------------------------------------------------
    # (R, t) lookup — used for full 3-D transforms
    # ------------------------------------------------------------------
    def lookup_R_t(
        self, target: str, source: str, stamp, *, extra_timeout: float = 0.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (R_3x3, t_3) or None.  Falls back to ``Time()`` on stamp miss."""
        t = self._raw_lookup(target, source, stamp)
        if t is None:
            t = self._raw_lookup(target, source, Time())
        if t is None and extra_timeout > 0:
            t = self._raw_lookup(target, source, Time(), timeout_sec=extra_timeout)
        if t is None:
            return None
        R = quat_to_rotmat(t.transform.rotation)
        tr = np.array(
            [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
            dtype=np.float64,
        )
        return R, tr

    def lookup_R_t_cached(
        self, target: str, source: str, stamp,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Per-frame cached version of ``lookup_R_t``."""
        s, ns = int(stamp.sec), int(stamp.nanosec)
        key = (target, source, s, ns)
        is_latest = (s == 0 and ns == 0)
        if is_latest or self._rt_key != key:
            self._rt_key = key
            self._rt_val = self.lookup_R_t(target, source, stamp)
        return self._rt_val

    # ------------------------------------------------------------------
    # Laser ← tracking  (1 lookup / frame reused by all tracks)
    # ------------------------------------------------------------------
    def laser_from_tracking_cached(
        self, laser_frame: str, tracking_frame: str, stamp,
        extra_timeout: float = 0.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not laser_frame:
            return None
        s, ns = int(stamp.sec), int(stamp.nanosec)
        key = (laser_frame, s, ns)
        is_latest = (s == 0 and ns == 0)
        if is_latest or self._laser_track_key != key:
            self._laser_track_key = key
            self._laser_track_val = self.lookup_R_t(
                laser_frame, tracking_frame, stamp, extra_timeout=extra_timeout,
            )
        return self._laser_track_val

    # ------------------------------------------------------------------
    # Planar (tx, ty, yaw) with stale-cache fallback
    # ------------------------------------------------------------------
    def _extract_planar(self, t) -> Tuple[float, float, float]:
        q = t.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return t.transform.translation.x, t.transform.translation.y, math.atan2(siny, cosy)

    def lookup_planar(
        self, target: str, source: str, stamp,
    ) -> Optional[Tuple[float, float, float]]:
        """(tx, ty, yaw) — stamp → latest → stale cache."""
        t = self._raw_lookup(target, source, stamp)
        if t is not None:
            val = self._extract_planar(t)
            self._last_good[f"{target}<-{source}"] = (*val, time.monotonic())
            return val
        t = self._raw_lookup(target, source, Time())
        if t is not None:
            val = self._extract_planar(t)
            self._last_good[f"{target}<-{source}"] = (*val, time.monotonic())
            return val
        cached = self._last_good.get(f"{target}<-{source}")
        if cached is not None and (time.monotonic() - cached[3]) < self._stale_max_sec:
            return (cached[0], cached[1], cached[2])
        return None

    def lookup_planar_stamped_only(
        self, target: str, source: str, stamp,
    ) -> Optional[Tuple[float, float, float]]:
        """(tx, ty, yaw) at stamp only; None if unavailable."""
        t = self._raw_lookup(target, source, stamp, timeout_sec=0.001)
        if t is not None:
            val = self._extract_planar(t)
            self._last_good[f"{target}<-{source}"] = (*val, time.monotonic())
            return val
        return None
    
    def lookup_planar_extrapolated(
        self, target: str, source: str, stamp, max_extrap_sec: float, window_sec: float = 0.1,
    ) -> Optional[Tuple[float, float, float]]:
        """(tx, ty, yaw) at ``stamp`` when the TF chain lags behind it.

        On the TIAGo the odometry TF (odom -> base_footprint) arrives ~65 ms after the
        /scan, so the stamped lookup fails and ``latest`` is off by the robot motion in
        that gap (~6.6 deg at 100 deg/s). Extrapolate the latest transform with the
        velocity seen over the previous ``window_sec`` (taken from the TF buffer, i.e.
        the odometry itself). None if the gap is larger than ``max_extrap_sec``.
        """
        from builtin_interfaces.msg import Time as BuiltinTime

        t1 = self._raw_lookup(target, source, Time())
        if t1 is None:
            return None
        s1 = float(t1.header.stamp.sec) + float(t1.header.stamp.nanosec) * 1e-9
        s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        p1 = self._extract_planar(t1)
        gap = s - s1
        if gap <= 0.0:
            return p1
        if gap > max_extrap_sec:
            return None
        s0 = s1 - window_sec
        t0 = self._raw_lookup(
            target, source, BuiltinTime(sec=int(s0), nanosec=int((s0 % 1.0) * 1e9))
        )
        if t0 is None:
            return p1
        p0 = self._extract_planar(t0)
        w = s1 - (float(t0.header.stamp.sec) + float(t0.header.stamp.nanosec) * 1e-9)
        if w <= 1e-3:
            return p1
        dyaw = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
        k = gap / w
        return (p1[0] + (p1[0] - p0[0]) * k, p1[1] + (p1[1] - p0[1]) * k, p1[2] + dyaw * k)

    def lookup_planar_stamped_with_info(
        self, target: str, source: str, stamp,
    ) -> Tuple[Optional[Tuple[float, float, float]], str]:
        """Like lookup_planar_stamped_only but returns (result, info_string).
        
        Useful for debugging TF availability issues.
        Uses very short timeout (1ms) to avoid blocking the pipeline.
        """
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9 if hasattr(stamp, 'sec') else 0.0
        # Very short timeout to avoid blocking
        t = self._raw_lookup(target, source, stamp, timeout_sec=0.001)
        if t is not None:
            val = self._extract_planar(t)
            self._last_good[f"{target}<-{source}"] = (*val, time.monotonic())
            return val, f"OK@stamp"
        
        # Try latest to see if TF chain exists at all (also short timeout)
        t_latest = self._raw_lookup(target, source, Time(), timeout_sec=0.001)
        if t_latest is not None:
            return None, f"STAMP_TOO_OLD"
        return None, f"NO_TF"

    def lookup_planar_cached(
        self, target: str, source: str, stamp,
    ) -> Optional[Tuple[float, float, float]]:
        """Per-frame cached ``lookup_planar``."""
        key = (source, int(stamp.sec), int(stamp.nanosec))
        if self._planar_key != key:
            self._planar_key = key
            self._planar_val = self.lookup_planar(target, source, stamp)
        return self._planar_val

    # ------------------------------------------------------------------
    # Convenience: transform a 2-D local point to tracking_frame
    # ------------------------------------------------------------------
    def transform_xy(
        self, x_local: float, y_local: float, source_frame: str, stamp, tracking_frame: str,
    ) -> Optional[Tuple[float, float]]:
        """Transform at stamp; None if TF@stamp unavailable."""
        tf_data = self.lookup_planar_stamped_only(tracking_frame, source_frame, stamp)
        if tf_data is None:
            return None
        tx, ty, yaw = tf_data
        c, s = math.cos(yaw), math.sin(yaw)
        return tx + c * x_local - s * y_local, ty + s * x_local + c * y_local
    
    def transform_xy_with_fallback(
        self, x_local: float, y_local: float, source_frame: str, stamp, tracking_frame: str,
    ) -> Optional[Tuple[float, float]]:
        """Transform at stamp with fallback to latest TF."""
        tf_data = self.lookup_planar_cached(tracking_frame, source_frame, stamp)
        if tf_data is None:
            return None
        tx, ty, yaw = tf_data
        c, s = math.cos(yaw), math.sin(yaw)
        return tx + c * x_local - s * y_local, ty + s * x_local + c * y_local

    def transform_xy_stamped_only(
        self, x_local: float, y_local: float, source_frame: str, stamp, tracking_frame: str,
    ) -> Optional[Tuple[float, float]]:
        """Transform using ONLY the TF at the given stamp, NO fallback to latest.
        
        Returns None if TF at exact stamp is not available.
        """
        tf_data = self.lookup_planar_stamped_only(tracking_frame, source_frame, stamp)
        if tf_data is None:
            return None
        tx, ty, yaw = tf_data
        c, s = math.cos(yaw), math.sin(yaw)
        return tx + c * x_local - s * y_local, ty + s * x_local + c * y_local

    def transform_xy_latest(
        self, x_local: float, y_local: float, source_frame: str, tracking_frame: str,
    ) -> Optional[Tuple[float, float]]:
        """Like transform_xy but always uses Time() (latest TF).

        This is the robust approach used by lidar_object_detection_ros2: the
        latest TF is always available regardless of AMCL publish rate, so the
        lookup never fails when the chain laser→base→odom→map exists.
        """
        t = self._raw_lookup(tracking_frame, source_frame, Time())
        if t is None:
            return None
        tx, ty, yaw = self._extract_planar(t)
        key = f"{tracking_frame}<-{source_frame}"
        self._last_good[key] = (tx, ty, yaw, time.monotonic())
        c, s = math.cos(yaw), math.sin(yaw)
        return tx + c * x_local - s * y_local, ty + s * x_local + c * y_local

    # ------------------------------------------------------------------
    # Convenience: tracking → map (for static-map filter)
    # ------------------------------------------------------------------
    def tracking_to_map(
        self, tx: float, ty: float, tracking_frame: str, map_frame: str,
    ) -> Optional[Tuple[float, float]]:
        """Planar tracking→map using latest TF (with stale fallback). Timeout 0.08 s."""
        if tracking_frame == map_frame:
            return (tx, ty)
        t = self._raw_lookup(map_frame, tracking_frame, Time(), timeout_sec=0.08)
        key = f"{map_frame}<-{tracking_frame}"
        if t is not None:
            val = self._extract_planar(t)
            self._last_good[key] = (*val, time.monotonic())
        else:
            cached = self._last_good.get(key)
            if cached is not None and (time.monotonic() - cached[3]) < 5.0:
                val = (cached[0], cached[1], cached[2])
            else:
                return None
        c, s = math.cos(val[2]), math.sin(val[2])
        return val[0] + c * tx - s * ty, val[1] + s * tx + c * ty

    # ------------------------------------------------------------------
    # Full 3-D point transform (camera → tracking)
    # ------------------------------------------------------------------
    def transform_point_3d(
        self, x: float, y: float, z: float, source: str, stamp, target: str,
    ) -> Optional[Tuple[float, float]]:
        """Returns (map_x, map_y) — 3-D rotation then drop z."""
        rt = self.lookup_R_t(target, source, stamp)
        if rt is None:
            return None
        R, t_vec = rt
        p = R @ np.array([x, y, z], dtype=np.float64) + t_vec
        return float(p[0]), float(p[1])


class MarkerRemapTF:
    """Manages the TF lookup + stale-cache for ``odom → map`` marker remapping."""

    def __init__(self) -> None:
        self._cache: Optional[Tuple[str, str, np.ndarray, np.ndarray,
                                     Tuple[float, float, float, float], float]] = None

    def lookup(
        self,
        tf_helper: TFHelper,
        from_frame: str,
        to_frame: str,
        stamp,
        *,
        prefer_latest: bool = False,
        allow_latest_fallback: bool = False,
        stamp_fallbacks: Optional[Sequence] = None,
        timeout_sec: float = 0.2,
        stale_cache_sec: float = 0.75,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, object]]:
        """Returns (R, t, quaternion-like) or None.  Updates internal stale cache."""
        raw = TFHelper.stamp_msg
        latest = Time().to_msg()
        stamp_msg = raw(stamp)
        fallbacks = tuple(stamp_fallbacks) if stamp_fallbacks else ()

        def _try(s):
            return tf_helper._raw_lookup(to_frame, from_frame, s, timeout_sec=timeout_sec)

        tmsg = None
        if prefer_latest:
            tmsg = _try(latest) or _try(stamp_msg)
            if tmsg is None:
                for fb in fallbacks:
                    tmsg = _try(raw(fb))
                    if tmsg is not None:
                        break
            if tmsg is None and allow_latest_fallback:
                tmsg = _try(latest)
        else:
            tmsg = _try(stamp_msg)
            if tmsg is None:
                for fb in fallbacks:
                    tmsg = _try(raw(fb))
                    if tmsg is not None:
                        break
            if tmsg is None and allow_latest_fallback:
                tmsg = _try(latest)

        if tmsg is not None:
            R = quat_to_rotmat(tmsg.transform.rotation)
            tt = tmsg.transform.translation
            t_vec = np.array([float(tt.x), float(tt.y), float(tt.z)], dtype=np.float64)
            q = tmsg.transform.rotation
            self._cache = (
                from_frame, to_frame, R.copy(), t_vec.copy(),
                (float(q.x), float(q.y), float(q.z), float(q.w)),
                time.monotonic(),
            )
            return R, t_vec, q

        if stale_cache_sec > 0 and self._cache is not None:
            cf, ct, R, t_vec, qtu, tmono = self._cache
            if cf == from_frame and ct == to_frame and (time.monotonic() - tmono) <= stale_cache_sec:
                q = SimpleNamespace(x=qtu[0], y=qtu[1], z=qtu[2], w=qtu[3])
                return R, t_vec, q

        return None
