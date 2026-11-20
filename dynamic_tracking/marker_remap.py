"""Remap MarkerArray from one frame to another (e.g. odom → map for RViz)."""

from typing import Optional, Sequence

import numpy as np
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import quat_multiply_xyzw, quat_to_rotmat
from .tf_utils import MarkerRemapTF, TFHelper


def remap_marker_array(
    arr: MarkerArray,
    from_frame: str,
    to_frame: str,
    stamp,
    tf_helper: TFHelper,
    remap_tf: MarkerRemapTF,
    *,
    prefer_latest: bool = False,
    allow_latest_fallback: bool = False,
    stamp_fallbacks: Optional[Sequence] = None,
    timeout_sec: float = 0.2,
    stale_cache_sec: float = 0.75,
    logger=None,
    _warned_ref: Optional[list] = None,
) -> MarkerArray:
    """Transform markers in-place from ``from_frame`` to ``to_frame``."""
    if from_frame == to_frame or not arr.markers:
        return arr

    result = remap_tf.lookup(
        tf_helper, from_frame, to_frame, stamp,
        prefer_latest=prefer_latest,
        allow_latest_fallback=allow_latest_fallback,
        stamp_fallbacks=stamp_fallbacks,
        timeout_sec=timeout_sec,
        stale_cache_sec=stale_cache_sec,
    )
    if result is None:
        if logger is not None and _warned_ref is not None and not _warned_ref[0]:
            logger.warning(
                f"marker_visualization_frame: no TF {from_frame!r}->{to_frame!r}, "
                "publishing markers in tracking_frame"
            )
            _warned_ref[0] = True
        return arr

    R_tf, t_tf, q_tf = result
    _apply_transform(arr, to_frame, R_tf, t_tf, q_tf)
    return arr


def _orient_identity(o) -> bool:
    return (
        abs(float(o.w) - 1.0) < 1e-3
        and abs(float(o.x)) < 1e-6
        and abs(float(o.y)) < 1e-6
        and abs(float(o.z)) < 1e-6
    )


def _apply_transform(arr, to_frame, R_tf, t_tf, q_tf):
    for m in arr.markers:
        m.header.frame_id = to_frame
        if m.action == Marker.DELETEALL:
            continue

        old_o = m.pose.orientation
        old_p = m.pose.position
        pos = np.array([float(old_p.x), float(old_p.y), float(old_p.z)], dtype=np.float64)
        identity = _orient_identity(old_o)

        if m.type == Marker.TEXT_VIEW_FACING:
            p = R_tf @ pos + t_tf
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p[0]), float(p[1]), float(p[2])
            continue

        wireframe_bbox = (
            m.type == Marker.LINE_LIST
            and m.ns == "dynamic_tracks_bbox"
            and len(m.points) > 0
            and not identity
        )
        if wireframe_bbox:
            R_m = quat_to_rotmat(old_o)
            for p in m.points:
                vl = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
                vm = R_tf @ (R_m @ vl + pos) + t_tf
                p.x, p.y, p.z = float(vm[0]), float(vm[1]), float(vm[2])
            _reset_pose(m)
            continue

        if len(m.points) > 0 and identity:
            for p in m.points:
                v = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
                vm = R_tf @ v + t_tf
                p.x, p.y, p.z = float(vm[0]), float(vm[1]), float(vm[2])
            _reset_pose(m)
            continue

        p_map = R_tf @ pos + t_tf
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p_map[0]), float(p_map[1]), float(p_map[2])

        if identity:
            _reset_orientation(m)
        else:
            qx, qy, qz, qw = quat_multiply_xyzw(
                float(q_tf.x), float(q_tf.y), float(q_tf.z), float(q_tf.w),
                float(old_o.x), float(old_o.y), float(old_o.z), float(old_o.w),
            )
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.pose.orientation.w = qw


def _reset_pose(m):
    m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.0
    _reset_orientation(m)


def _reset_orientation(m):
    m.pose.orientation.x = m.pose.orientation.y = m.pose.orientation.z = 0.0
    m.pose.orientation.w = 1.0
