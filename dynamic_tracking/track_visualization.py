"""Colours per fusion mode and MarkerArray for RViz."""

import math
from typing import Dict, List, Optional, Tuple

from geometry_msgs.msg import Point
from rclpy.duration import Duration
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .types import LidarMeasurement, Track


def is_lidar_track_moving(
    tr: Track, motion_speed_threshold_mps: float, motion_accel_threshold_mps2: float
) -> bool:
    if tr.last_update_source != "lidar-only":
        return False
    vx, vy = tr.velocity()
    sp = math.hypot(vx, vy)
    accel = abs(sp - tr.prev_speed_mps)
    return sp > motion_speed_threshold_mps or accel > motion_accel_threshold_mps2


def person_yolo_seen_recent(
    tr: Track, stamp_sec: float, seen_green_sec: float
) -> bool:
    """True if YOLO marked person on this track less than seen_green_sec ago (same clock as RViz)."""
    if seen_green_sec <= 0.0:
        return False
    lr = float(getattr(tr, "last_rgb_person_sec", 0.0))
    if lr <= 0.0:
        return False
    # header_markers may use the /scan stamp and last_rgb the image one: use |dt| (usually ~0.03–0.05 s).
    return abs(float(stamp_sec) - lr) <= float(seen_green_sec)


def yolo_stamp_recent(last_sec: float, stamp_sec: float, window_sec: float) -> bool:
    if window_sec <= 0.0 or last_sec <= 0.0:
        return False
    return abs(float(stamp_sec) - float(last_sec)) <= float(window_sec)


def person_yolo_laser_coast(tr: Track, stamp_sec: float, coast_sec: float) -> bool:
    """YOLO confirmed person at some point; while the ID lives, teal coast (laser only)."""
    del stamp_sec, coast_sec  # track lifetime, not a time window
    if str(tr.class_name).lower() != "person":
        return False
    return float(getattr(tr, "last_rgb_person_sec", 0.0)) > 0.0


def person_lidar_holding_after_rgb(tr: Track, stamp_sec: float, hold_sec: float) -> bool:
    """This track had a YOLO person; the last update is laser only (camera lost sight). Orange state."""
    if hold_sec <= 0.0:
        return False
    if tr.class_name != "person" or tr.last_update_source != "lidar-only":
        return False
    last_rgb = float(getattr(tr, "last_rgb_person_sec", 0.0))
    return last_rgb > 0.0 and (stamp_sec - last_rgb) <= float(hold_sec)


# Alias for compatibility with existing YAML parameters
def person_rgb_coasts_as_cl(tr: Track, stamp_sec: float, coast_sec: float) -> bool:
    return person_lidar_holding_after_rgb(tr, stamp_sec, coast_sec)


def track_color_bgr(
    tr: Track,
    motion_speed_threshold_mps: float,
    motion_accel_threshold_mps2: float,
    *,
    stamp_sec: Optional[float] = None,
    person_rgb_coast_cl_sec: float = 0.0,
    yolo_visual_lidar_parity: bool = False,
    person_yolo_seen_green_sec: float = 0.5,
    marker_color_coast_bgr: Tuple[int, int, int] = (180, 200, 40),
) -> Tuple[int, int, int]:
    if tr.class_name == "unknown":
        return (180, 180, 180)
    if str(tr.class_name).lower() == "person" and yolo_visual_lidar_parity and stamp_sec is not None:
        if person_yolo_seen_recent(tr, float(stamp_sec), float(person_yolo_seen_green_sec)):
            return (0, 255, 0)
        if person_yolo_laser_coast(tr, float(stamp_sec), float(person_rgb_coast_cl_sec)):
            return marker_color_coast_bgr
        return (220, 90, 40)
    # YOLO as auxiliary: same shades as LiDAR-only; green only while YOLO sees a person.
    if yolo_visual_lidar_parity and str(tr.class_name).lower() not in ("person", "unknown"):
        if is_lidar_track_moving(tr, motion_speed_threshold_mps, motion_accel_threshold_mps2):
            return (180, 110, 60)
        if getattr(tr, "leg_hypothesis_active", False):
            return (200, 100, 50)
        return (160, 120, 70)

    if tr.class_name == "unknown":
        return (180, 180, 180)
    # Legacy green: camera+laser fusion in the same cycle (laser ray inside the bbox).
    if tr.last_update_source == "camera+lidar":
        return (0, 255, 0)
    if stamp_sec is not None and person_yolo_laser_coast(
        tr, float(stamp_sec), float(person_rgb_coast_cl_sec)
    ):
        return marker_color_coast_bgr
    if tr.last_update_source == "camera-only":
        return (0, 220, 255)  # coral / no laser ray in the bbox
    if tr.last_update_source == "lidar-only":
        # Blue: laser-only person never confirmed by YOLO (tentative dynamic).
        if tr.class_name == "person":
            return (220, 90, 40)  # strong blue (BGR)
        if getattr(tr, "leg_hypothesis_active", False):
            return (200, 100, 50)
        if is_lidar_track_moving(tr, motion_speed_threshold_mps, motion_accel_threshold_mps2):
            return (180, 110, 60)
        return (160, 120, 70)  # unknown lidar-only
    return (200, 200, 200)


def marker_rgb_from_track(
    tr: Track,
    motion_speed_threshold_mps: float,
    motion_accel_threshold_mps2: float,
    *,
    stamp_sec: Optional[float] = None,
    person_rgb_coast_cl_sec: float = 0.0,
    yolo_visual_lidar_parity: bool = False,
    person_yolo_seen_green_sec: float = 0.5,
    marker_color_coast_bgr: Tuple[int, int, int] = (180, 200, 40),
) -> Tuple[float, float, float]:
    b, g, r = track_color_bgr(
        tr,
        motion_speed_threshold_mps,
        motion_accel_threshold_mps2,
        stamp_sec=stamp_sec,
        person_rgb_coast_cl_sec=person_rgb_coast_cl_sec,
        yolo_visual_lidar_parity=yolo_visual_lidar_parity,
        person_yolo_seen_green_sec=person_yolo_seen_green_sec,
        marker_color_coast_bgr=marker_color_coast_bgr,
    )
    return r / 255.0, g / 255.0, b / 255.0


def fusion_tag_ui(last_update_source: str) -> str:
    """No spaces: CL=cam+laser, L=laser only, C=camera only."""
    return {
        "camera+lidar": "CL",
        "lidar-only": "L",
        "camera-only": "C",
    }.get(last_update_source, "?")


def class_tag_ui(
    class_name: str, uncertain_person: bool, motion_dynamic: bool = False
) -> str:
    """Short class label; person -> dyn / dyn? / dym (real motion on the map)."""
    cn = (class_name or "?").lower()
    if cn == "person":
        if uncertain_person:
            return "dyn?"
        if motion_dynamic:
            return "dym"
        return "dyn"
    if cn == "unknown":
        return "unk"
    raw = class_name or "?"
    return raw if len(raw) <= 4 else raw[:3]


def uncertain_static_lidar_person(
    tr: Track,
    stamp_sec: float,
    *,
    marker_uncertain_static_lidar_person: bool,
    person_lidar_uncertain_stale_rgb_sec: float,
    motion_speed_threshold_mps: float,
    motion_accel_threshold_mps2: float,
    person_rgb_coast_cl_sec: float = 0.0,
    yolo_visual_lidar_parity: bool = False,
    person_yolo_seen_green_sec: float = 0.5,
) -> bool:
    """True = same condition as the dyn? label (doubtful laser-only person, e.g. a wall)."""
    if not marker_uncertain_static_lidar_person:
        return False
    if yolo_visual_lidar_parity and person_yolo_seen_recent(
        tr, float(stamp_sec), float(person_yolo_seen_green_sec)
    ):
        return False
    if tr.class_name != "person" or tr.last_update_source != "lidar-only":
        return False
    if person_lidar_holding_after_rgb(tr, stamp_sec, person_rgb_coast_cl_sec):
        return False
    moving = is_lidar_track_moving(tr, motion_speed_threshold_mps, motion_accel_threshold_mps2)
    last_rgb = float(getattr(tr, "last_rgb_person_sec", 0.0))
    stale_rgb = last_rgb <= 0.0 or (stamp_sec - last_rgb) > float(
        person_lidar_uncertain_stale_rgb_sec
    )
    return (not moving) or stale_rgb


def track_label_compact_line(
    tr: Track,
    stamp_sec: float,
    *,
    motion_speed_threshold_mps: float,
    motion_accel_threshold_mps2: float,
    marker_uncertain_static_lidar_person: bool,
    person_lidar_uncertain_stale_rgb_sec: float,
    person_rgb_coast_cl_sec: float = 0.0,
    yolo_visual_lidar_parity: bool = False,
    person_yolo_seen_green_sec: float = 0.5,
) -> str:
    """RViz label: Person only with recent YOLO or within the coast window; not because of an old spurious match."""
    is_person = str(tr.class_name).lower() == "person"

    if yolo_visual_lidar_parity and is_person and stamp_sec is not None:
        if person_yolo_seen_recent(tr, float(stamp_sec), float(person_yolo_seen_green_sec)):
            return f"ID:{tr.track_id}-Person"
        if person_yolo_laser_coast(tr, float(stamp_sec), float(person_rgb_coast_cl_sec)):
            return f"ID:{tr.track_id}-Person"

    if is_person and tr.last_update_source == "lidar-only":
        return f"ID:{tr.track_id}-DynObct"

    hyp = "~" if getattr(tr, "leg_hypothesis_active", False) else ""
    uncertain_person = uncertain_static_lidar_person(
        tr,
        stamp_sec,
        marker_uncertain_static_lidar_person=marker_uncertain_static_lidar_person,
        person_lidar_uncertain_stale_rgb_sec=person_lidar_uncertain_stale_rgb_sec,
        motion_speed_threshold_mps=motion_speed_threshold_mps,
        motion_accel_threshold_mps2=motion_accel_threshold_mps2,
        person_rgb_coast_cl_sec=person_rgb_coast_cl_sec,
        yolo_visual_lidar_parity=yolo_visual_lidar_parity,
        person_yolo_seen_green_sec=person_yolo_seen_green_sec,
    )
    motion_dyn = bool(getattr(tr, "motion_dynamic_confirmed", False))
    cls = class_tag_ui(tr.class_name, uncertain_person, motion_dyn)
    if tr.last_update_source == "camera+lidar":
        tag = "CL"
    elif person_lidar_holding_after_rgb(tr, stamp_sec, person_rgb_coast_cl_sec):
        tag = "OL"
    else:
        tag = fusion_tag_ui(tr.last_update_source)
    return f"#{tr.track_id}{hyp}{cls}{tag}"


def marker_lifetime_msg(marker_lifetime_sec: float):
    if marker_lifetime_sec <= 0:
        return Duration(seconds=0).to_msg()
    return Duration(seconds=marker_lifetime_sec).to_msg()


def _quaternion_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _bbox_yaw_from_velocity(
    vx: float, vy: float, *, use_heading: bool, min_speed_mps: float
) -> float:
    if not use_heading:
        return 0.0
    if math.hypot(vx, vy) < float(min_speed_mps):
        return 0.0
    return math.atan2(float(vy), float(vx))


def _wireframe_box_line_points(hx: float, hy: float, hz: float) -> List[Point]:
    """Edges of an OBB centred at the origin of the marker frame (LINE_LIST = vertex pairs)."""
    c = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    pts: List[Point] = []
    for a, b in edges:
        xa, ya, za = c[a]
        xb, yb, zb = c[b]
        p0 = Point()
        p0.x, p0.y, p0.z = xa, ya, za
        p1 = Point()
        p1.x, p1.y, p1.z = xb, yb, zb
        pts.append(p0)
        pts.append(p1)
    return pts


def _subsample_trail_indices(n: int, max_points: int) -> List[int]:
    if n <= 0 or max_points <= 0:
        return []
    if n <= max_points:
        return list(range(n))
    out: List[int] = []
    last = -1
    for j in range(max_points):
        idx = int(round(j * (n - 1) / max(1, max_points - 1)))
        if idx != last:
            out.append(idx)
        last = idx
    return out


def build_track_marker_array(
    tracks: List[Track],
    header,
    tracking_frame: str,
    publish_unknown_markers: bool,
    unknown_min_speed_mps: float,
    lidar_only_markers_require_motion: bool,
    motion_speed_threshold_mps: float,
    motion_accel_threshold_mps2: float,
    person_box_size_xy_m: float,
    person_box_height_m: float,
    unknown_box_size_xy_m: float,
    unknown_box_height_m: float,
    marker_lifetime_sec: float,
    marker_text_scale_z: float = 0.28,
    marker_compact_map_labels: bool = True,
    marker_text_height_m: float = 1.55,
    marker_text_use_track_color: bool = False,
    marker_text_color_r: float = 0.06,
    marker_text_color_g: float = 0.06,
    marker_text_color_b: float = 0.10,
    marker_label_ring_radius_m: float = 0.35,
    marker_label_z_stagger_m: float = 0.06,
    marker_uncertain_static_lidar_person: bool = True,
    person_lidar_uncertain_stale_rgb_sec: float = 4.0,
    person_rgb_coast_cl_sec: float = 0.0,
    marker_hide_uncertain_static_lidar_person: bool = False,
    marker_hide_person_if_missed_frames: int = 0,
    # Spencer filter_visually_confirmed_tracks: person tracks need RGB to be visible.
    require_visual_confirmation: bool = True,
    min_confirmed_hits: int = 3,
    yolo_visual_lidar_parity: bool = False,
    person_yolo_seen_green_sec: float = 0.5,
    marker_color_coast_bgr: Tuple[int, int, int] = (180, 200, 40),
    marker_bbox_alpha: float = 0.82,
    marker_bbox_wireframe: bool = False,
    marker_bbox_line_width_m: float = 0.022,
    marker_bbox_use_velocity_heading: bool = True,
    marker_bbox_heading_min_speed_mps: float = 0.06,
    publish_track_trajectory: bool = True,
    track_trajectory_z_m: float = 0.06,
    track_trajectory_line_width_m: float = 0.035,
    track_trajectory_alpha: float = 0.65,
    publish_track_history_cones: bool = False,
    track_history_max_markers: int = 28,
    track_history_cylinder_height_min_m: float = 0.035,
    track_history_cylinder_height_max_m: float = 0.14,
    track_history_cylinder_diameter_min_m: float = 0.05,
    track_history_cylinder_diameter_max_m: float = 0.12,
    track_history_alpha_min: float = 0.12,
    track_history_alpha_max: float = 0.52,
    track_history_z_m: float = 0.02,
    label_stamp_sec: Optional[float] = None,
) -> MarkerArray:
    marker_arr = MarkerArray()

    # DELETEALL only deletes the given namespace: the boxes are in dynamic_tracks_*,
    # not in dynamic_tracking; without this RViz piles up thousands of old markers.
    for ns in (
        "dynamic_tracking",
        "dynamic_tracks_bbox",
        "dynamic_tracks_text",
        "dynamic_tracks_trail",
        "dynamic_tracks_history",
    ):
        clear_marker = Marker()
        clear_marker.header = header
        clear_marker.header.frame_id = tracking_frame
        clear_marker.ns = ns
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_arr.markers.append(clear_marker)

    lt_msg = marker_lifetime_msg(marker_lifetime_sec)
    stamp_sec = float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
    if label_stamp_sec is not None:
        stamp_sec = float(label_stamp_sec)
    marker_id = 1
    for tr in tracks:
        if tr.hits < min_confirmed_hits:
            continue
        # Spencer filter_visually_confirmed_tracks: person tracks without RGB
        # are "tentative" (they exist for internal association, but are invisible).
        if require_visual_confirmation and tr.class_name == "person" and tr.last_rgb_person_sec <= 0.0:
            continue
        if (
            marker_hide_person_if_missed_frames > 0
            and tr.class_name == "person"
            and tr.missed_frames >= marker_hide_person_if_missed_frames
        ):
            continue

        map_x, map_y = tr.display_position()
        vx, vy = tr.velocity()
        speed = math.sqrt(vx * vx + vy * vy)

        is_unknown = tr.class_name == "unknown"
        if is_unknown and not publish_unknown_markers:
            continue
        if is_unknown and unknown_min_speed_mps > 0.0 and speed < unknown_min_speed_mps:
            continue
        holding = person_lidar_holding_after_rgb(tr, stamp_sec, person_rgb_coast_cl_sec)
        if (
            tr.last_update_source == "lidar-only"
            and not is_lidar_track_moving(tr, motion_speed_threshold_mps, motion_accel_threshold_mps2)
            and lidar_only_markers_require_motion
            and not holding
        ):
            continue
        if marker_hide_uncertain_static_lidar_person and uncertain_static_lidar_person(
            tr,
            stamp_sec,
            marker_uncertain_static_lidar_person=marker_uncertain_static_lidar_person,
            person_lidar_uncertain_stale_rgb_sec=person_lidar_uncertain_stale_rgb_sec,
            motion_speed_threshold_mps=motion_speed_threshold_mps,
            motion_accel_threshold_mps2=motion_accel_threshold_mps2,
            person_rgb_coast_cl_sec=person_rgb_coast_cl_sec,
            yolo_visual_lidar_parity=yolo_visual_lidar_parity,
            person_yolo_seen_green_sec=person_yolo_seen_green_sec,
        ):
            continue

        # Bbox: solid CUBE or wireframe LINE_LIST; pose at the centre with yaw from the Kalman velocity (map).
        body = Marker()
        body.header = header
        body.header.frame_id = tracking_frame
        body.ns = "dynamic_tracks_bbox"
        body.id = marker_id
        marker_id += 1
        body.action = Marker.ADD
        if is_unknown:
            box_xy = float(unknown_box_size_xy_m)
            box_h = float(unknown_box_height_m)
        else:
            box_xy = float(person_box_size_xy_m)
            box_h = float(person_box_height_m)
        hz_geom = 0.5 * box_h
        # Only the wireframe usually benefits from the yaw; the square CUBE in XY was identical with q=1 before.
        # Applying yaw to the solid box with a noisy velocity can look unstable in RViz.
        use_heading = (
            bool(marker_bbox_wireframe) and bool(marker_bbox_use_velocity_heading)
        )
        yaw = _bbox_yaw_from_velocity(
            vx,
            vy,
            use_heading=use_heading,
            min_speed_mps=float(marker_bbox_heading_min_speed_mps),
        )
        qx, qy, qz, qw = _quaternion_yaw(yaw)
        body.pose.position.x = float(map_x)
        body.pose.position.y = float(map_y)
        body.pose.position.z = float(hz_geom)
        body.pose.orientation.x = qx
        body.pose.orientation.y = qy
        body.pose.orientation.z = qz
        body.pose.orientation.w = qw

        mr, mg, mb = marker_rgb_from_track(
            tr,
            motion_speed_threshold_mps,
            motion_accel_threshold_mps2,
            stamp_sec=stamp_sec,
            person_rgb_coast_cl_sec=person_rgb_coast_cl_sec,
            yolo_visual_lidar_parity=yolo_visual_lidar_parity,
            person_yolo_seen_green_sec=person_yolo_seen_green_sec,
            marker_color_coast_bgr=marker_color_coast_bgr,
        )
        body.color.r = mr
        body.color.g = mg
        body.color.b = mb
        body.color.a = max(0.0, min(1.0, float(marker_bbox_alpha)))

        if marker_bbox_wireframe:
            body.type = Marker.LINE_LIST
            hx = 0.5 * box_xy
            hy = 0.5 * box_xy
            body.points = _wireframe_box_line_points(hx, hy, hz_geom)
            lw = max(0.004, float(marker_bbox_line_width_m))
            body.scale.x = lw
            body.scale.y = lw
            body.scale.z = lw
        else:
            body.type = Marker.CUBE
            body.scale.x = box_xy
            body.scale.y = box_xy
            body.scale.z = box_h

        body.lifetime = lt_msg
        marker_arr.markers.append(body)

        text = Marker()
        text.header = body.header
        text.ns = "dynamic_tracks_text"
        text.id = marker_id
        marker_id += 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        slots = 16
        ang = 2.0 * math.pi * ((tr.track_id % slots) / float(slots))
        rr = float(marker_label_ring_radius_m)
        if rr > 1e-6:
            tcx = map_x + rr * math.cos(ang)
            tcy = map_y + rr * math.sin(ang)
        else:
            tcx, tcy = map_x, map_y
        z_st = float(marker_text_height_m) + float(marker_label_z_stagger_m) * float(tr.track_id % 8)
        text.pose.position.x = float(tcx)
        text.pose.position.y = float(tcy)
        text.pose.position.z = z_st
        text.pose.orientation.x = 0.0
        text.pose.orientation.y = 0.0
        text.pose.orientation.z = 0.0
        text.pose.orientation.w = 1.0
        text.scale.z = float(marker_text_scale_z)
        if marker_text_use_track_color:
            text.color.r = mr
            text.color.g = mg
            text.color.b = mb
        else:
            text.color.r = float(marker_text_color_r)
            text.color.g = float(marker_text_color_g)
            text.color.b = float(marker_text_color_b)
        text.color.a = 1.0
        text.text = track_label_compact_line(
            tr,
            stamp_sec,
            motion_speed_threshold_mps=motion_speed_threshold_mps,
            motion_accel_threshold_mps2=motion_accel_threshold_mps2,
            marker_uncertain_static_lidar_person=marker_uncertain_static_lidar_person,
            person_lidar_uncertain_stale_rgb_sec=person_lidar_uncertain_stale_rgb_sec,
            person_rgb_coast_cl_sec=person_rgb_coast_cl_sec,
            yolo_visual_lidar_parity=yolo_visual_lidar_parity,
            person_yolo_seen_green_sec=person_yolo_seen_green_sec,
        )
        if getattr(tr, "reid_name", None):
            text.text += f" ({tr.reid_name})"
        text.lifetime = lt_msg
        marker_arr.markers.append(text)

        if publish_track_trajectory:
            ring = getattr(tr, "pose_trail_ring", None)
            if ring is not None and len(ring) >= 2:
                trail = Marker()
                trail.header = header
                trail.header.frame_id = tracking_frame
                trail.ns = "dynamic_tracks_trail"
                trail.id = int(tr.track_id)
                trail.type = Marker.LINE_STRIP
                trail.action = Marker.ADD
                trail.scale.x = max(0.005, float(track_trajectory_line_width_m))
                trail.pose.orientation.w = 1.0
                zt = float(track_trajectory_z_m)
                for px, py in ring:
                    pt = Point()
                    pt.x = float(px)
                    pt.y = float(py)
                    pt.z = zt
                    trail.points.append(pt)
                trail.color.r = mr
                trail.color.g = mg
                trail.color.b = mb
                trail.color.a = max(0.0, min(1.0, float(track_trajectory_alpha)))
                trail.lifetime = lt_msg
                marker_arr.markers.append(trail)

        # History: vertical cylinders (Marker has no CONE in ROS 2); older = smaller and fainter.
        if publish_track_history_cones:
            ring = getattr(tr, "pose_trail_ring", None)
            if ring is not None and len(ring) >= 2:
                samples = list(ring)
                n = len(samples)
                cap = max(2, int(track_history_max_markers))
                idxs = _subsample_trail_indices(n, cap)
                h0 = float(track_history_cylinder_height_min_m)
                h1 = float(track_history_cylinder_height_max_m)
                d0 = float(track_history_cylinder_diameter_min_m)
                d1 = float(track_history_cylinder_diameter_max_m)
                a0 = float(track_history_alpha_min)
                a1 = float(track_history_alpha_max)
                zf = float(track_history_z_m)
                mseg = len(idxs)
                for j, si in enumerate(idxs):
                    px, py = samples[si]
                    u = float(j) / float(mseg - 1) if mseg > 1 else 1.0
                    ch = h0 + (h1 - h0) * u
                    cd = d0 + (d1 - d0) * u
                    ca = a0 + (a1 - a0) * u
                    cyl = Marker()
                    cyl.header = header
                    cyl.header.frame_id = tracking_frame
                    cyl.ns = "dynamic_tracks_history"
                    cyl.id = int((tr.track_id & 0xFFFF) * cap + j)
                    cyl.type = Marker.CYLINDER
                    cyl.action = Marker.ADD
                    cz = zf + 0.5 * ch
                    cyl.pose.position.x = float(px)
                    cyl.pose.position.y = float(py)
                    cyl.pose.position.z = float(cz)
                    cyl.pose.orientation.w = 1.0
                    cyl.scale.x = float(cd)
                    cyl.scale.y = float(cd)
                    cyl.scale.z = float(ch)
                    cyl.color.r = mr
                    cyl.color.g = mg
                    cyl.color.b = mb
                    cyl.color.a = max(0.0, min(1.0, ca))
                    cyl.lifetime = lt_msg
                    marker_arr.markers.append(cyl)

    return marker_arr


def build_yolo_laser_polar_debug_marker_array(
    header: Header,
    yolo_person_polar: List[Tuple[float, float]],
    track_polar: Optional[Dict[int, Tuple[float, float]]],
    *,
    marker_lifetime_sec: float = 0.0,
    line_width_m: float = 0.022,
    z_yolo_m: float = 0.07,
    z_track_m: float = 0.02,
) -> MarkerArray:
    """Rays in the /scan frame: YOLO hits (green) vs track (r,theta) (cyan). Debug the polar phase."""
    arr = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.ns = "yolo_polar_debug"
    clear.id = 0
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)

    lt_msg = marker_lifetime_msg(marker_lifetime_sec)

    def _ray_points(rng: float, ang: float, z: float) -> Tuple[Point, Point]:
        x = float(rng) * math.cos(float(ang))
        y = float(rng) * math.sin(float(ang))
        p0 = Point()
        p0.x, p0.y, p0.z = 0.0, 0.0, float(z)
        p1 = Point()
        p1.x, p1.y, p1.z = x, y, float(z)
        return p0, p1

    if yolo_person_polar:
        m_y = Marker()
        m_y.header = header
        m_y.ns = "yolo_polar_debug"
        m_y.id = 1
        m_y.type = Marker.LINE_LIST
        m_y.action = Marker.ADD
        m_y.pose.orientation.w = 1.0
        m_y.scale.x = max(0.004, float(line_width_m))
        m_y.color.r = 0.15
        m_y.color.g = 1.0
        m_y.color.b = 0.25
        m_y.color.a = 0.92
        m_y.lifetime = lt_msg
        for rng, ang in yolo_person_polar:
            if not math.isfinite(rng) or not math.isfinite(ang):
                continue
            p0, p1 = _ray_points(rng, ang, z_yolo_m)
            m_y.points.append(p0)
            m_y.points.append(p1)
        if m_y.points:
            arr.markers.append(m_y)

    if track_polar:
        m_t = Marker()
        m_t.header = header
        m_t.ns = "yolo_polar_debug"
        m_t.id = 2
        m_t.type = Marker.LINE_LIST
        m_t.action = Marker.ADD
        m_t.pose.orientation.w = 1.0
        m_t.scale.x = max(0.004, float(line_width_m) * 0.92)
        m_t.color.r = 0.2
        m_t.color.g = 0.75
        m_t.color.b = 1.0
        m_t.color.a = 0.88
        m_t.lifetime = lt_msg
        for _ti, (rng, ang) in sorted(track_polar.items()):
            if not math.isfinite(rng) or not math.isfinite(ang):
                continue
            p0, p1 = _ray_points(rng, ang, z_track_m)
            m_t.points.append(p0)
            m_t.points.append(p1)
        if m_t.points:
            arr.markers.append(m_t)

    return arr


def build_lidar_cluster_marker_array(
    measurements: List[LidarMeasurement],
    header,
    tracking_frame: str,
    marker_lifetime_sec: float,
    sphere_diameter_m: float = 0.14,
) -> MarkerArray:
    """Cyan spheres at (map_x, map_y) of each LiDAR cluster after filters (laser predictions in RViz)."""
    arr = MarkerArray()
    clear = Marker()
    clear.header.stamp = header.stamp
    clear.header.frame_id = tracking_frame
    clear.ns = "lidar_cluster_hypotheses"
    clear.id = 0
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    if not measurements:
        return arr
    lt_msg = marker_lifetime_msg(marker_lifetime_sec)
    for i, lm in enumerate(measurements, start=1):
        m = Marker()
        m.header.stamp = header.stamp
        m.header.frame_id = tracking_frame
        m.ns = "lidar_cluster_hypotheses"
        m.id = i
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(lm.map_x)
        m.pose.position.y = float(lm.map_y)
        m.pose.position.z = float(sphere_diameter_m) * 0.5
        m.pose.orientation.w = 1.0
        d = float(sphere_diameter_m)
        m.scale.x = d
        m.scale.y = d
        m.scale.z = d
        sc = min(1.0, max(0.0, float(lm.leg_candidate_score)))
        m.color.r = 0.15 + 0.45 * (1.0 - sc)
        m.color.g = 0.55 + 0.35 * sc
        m.color.b = 1.0
        m.color.a = 0.88
        m.lifetime = lt_msg
        arr.markers.append(m)
    return arr


def _triangulate_polygon(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ear clipping for a simple (possibly concave) polygon -> flat list of triangle vertices."""
    pts = list(poly)
    area2 = sum(
        pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
        for i in range(len(pts))
    )
    if area2 < 0.0:
        pts.reverse()  # make it counter-clockwise

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    tris: List[Tuple[float, float]] = []
    idx = list(range(len(pts)))
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            a, b, c = pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % n]]
            if cross(a, b, c) <= 0.0:
                continue  # reflex vertex
            if any(
                cross(a, b, pts[j]) > 0.0 and cross(b, c, pts[j]) > 0.0 and cross(c, a, pts[j]) > 0.0
                for j in idx if pts[j] not in (a, b, c)
            ):
                continue
            tris.extend((a, b, c))
            idx.pop(k)
            break
        else:
            break  # degenerate polygon: draw what we have
    if len(idx) == 3:
        tris.extend(pts[j] for j in idx)
    return tris


def build_area_zone_marker_array(
    zones: List[dict],
    area_frame: str,
    stamp,
) -> MarkerArray:
    """Flat ROI on the floor (magenta fill + border), Spencer-style. Rectangles or polygons."""
    arr = MarkerArray()
    for i, z in enumerate(zones):
        if "polygon" in z:
            outline = [(float(x), float(y)) for x, y in z["polygon"]]
        else:
            x_min, x_max = float(z["x_min"]), float(z["x_max"])
            y_min, y_max = float(z["y_min"]), float(z["y_max"])
            outline = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        is_add = bool(z.get("add", True))

        fill = Marker()
        fill.header.stamp = stamp
        fill.header.frame_id = area_frame
        fill.ns = "detection_area"
        fill.id = i
        fill.type = Marker.TRIANGLE_LIST
        fill.action = Marker.ADD
        fill.pose.orientation.w = 1.0
        fill.scale.x = fill.scale.y = fill.scale.z = 1.0
        if is_add:
            fill.color.r, fill.color.g, fill.color.b, fill.color.a = 0.82, 0.18, 0.82, 0.42
        else:
            fill.color.r, fill.color.g, fill.color.b, fill.color.a = 1.0, 0.15, 0.15, 0.35
        fill.lifetime = marker_lifetime_msg(0.0)
        z_fill = 0.025 if is_add else 0.035
        for px, py in _triangulate_polygon(outline):
            pt = Point()
            pt.x, pt.y, pt.z = px, py, z_fill
            fill.points.append(pt)
        arr.markers.append(fill)

        border = Marker()
        border.header.stamp = stamp
        border.header.frame_id = area_frame
        border.ns = "detection_area_border"
        border.id = i
        border.type = Marker.LINE_STRIP
        border.action = Marker.ADD
        border.pose.orientation.w = 1.0
        border.scale.x = 0.08
        if is_add:
            border.color.r, border.color.g, border.color.b, border.color.a = 0.95, 0.25, 0.95, 0.95
        else:
            border.color.r, border.color.g, border.color.b, border.color.a = 1.0, 0.2, 0.2, 0.95
        border.lifetime = marker_lifetime_msg(0.0)
        z_line = 0.04 if is_add else 0.05
        for px, py in outline + outline[:1]:
            pt = Point()
            pt.x, pt.y, pt.z = px, py, z_line
            border.points.append(pt)
        arr.markers.append(border)
    return arr

