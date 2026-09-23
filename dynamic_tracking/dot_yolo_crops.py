"""Build DotYoloCropArray (DOT + YOLO person crops) for pose_estimation."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Header

from .leg_pair import leg_pair_for_track
from .types import LidarMeasurement, Track


def _clip_bbox(
    bbox: Tuple[float, float, float, float],
    margin_px: float,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    m = max(0.0, float(margin_px))
    x1 = int(max(0, np.floor(bbox[0] - m)))
    y1 = int(max(0, np.floor(bbox[1] - m)))
    x2 = int(min(w, np.ceil(bbox[2] + m)))
    y2 = int(min(h, np.ceil(bbox[3] + m)))
    return x1, y1, x2, y2


def build_dot_yolo_crop_array(
    tracks: List[Track],
    cv_image: np.ndarray,
    header: Header,
    bridge: CvBridge,
    *,
    min_confirmed_hits: int,
    margin_px: float = 8.0,
    require_visual_confirmation: bool = True,
    overlay_max_age_sec: float = 0.0,
    frame_sec: float = 0.0,
    lidar_clusters: Optional[List[LidarMeasurement]] = None,
    min_leg_score: float = 0.58,
    leg_pair_max_dist_m: float = 0.9,
):
    """One crop per confirmed person track (DOT track_id + map pose + YOLO bbox)."""
    try:
        from pose_estimation_interfaces.msg import DotYoloCrop, DotYoloCropArray
    except ImportError as exc:
        raise ImportError(
            "pose_estimation_interfaces is required for DOT YOLO crops. "
            "Build the pose-estimation workspace first."
        ) from exc

    msg = DotYoloCropArray()
    msg.header = header
    h, w = cv_image.shape[:2]

    for tr in tracks:
        if tr.hits < min_confirmed_hits:
            continue
        if tr.class_name.lower() != "person":
            continue
        if require_visual_confirmation and tr.last_rgb_person_sec <= 0.0:
            continue
        if tr.last_bbox is None:
            continue
        if overlay_max_age_sec > 0.0 and tr.last_image_bbox_stamp_sec > 0.0:
            age = frame_sec - tr.last_image_bbox_stamp_sec
            if age > overlay_max_age_sec:
                continue

        x1, y1, x2, y2 = _clip_bbox(tr.last_bbox, margin_px, w, h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop_bgr = cv_image[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        entry = DotYoloCrop()
        entry.dot_track_id = int(tr.track_id)
        mx, my = tr.state()
        entry.map_x = float(mx)
        entry.map_y = float(my)
        entry.bbox_x1 = float(x1)
        entry.bbox_y1 = float(y1)
        entry.bbox_x2 = float(x2)
        entry.bbox_y2 = float(y2)
        entry.legs_valid = False
        entry.leg_left_x = 0.0
        entry.leg_left_y = 0.0
        entry.leg_right_x = 0.0
        entry.leg_right_y = 0.0
        if lidar_clusters:
            ok, llx, lly, lrx, lry = leg_pair_for_track(
                mx,
                my,
                lidar_clusters,
                min_leg_score=float(min_leg_score),
                max_dist_m=float(leg_pair_max_dist_m),
            )
            if ok:
                entry.legs_valid = True
                entry.leg_left_x = float(llx)
                entry.leg_left_y = float(lly)
                entry.leg_right_x = float(lrx)
                entry.leg_right_y = float(lry)
        entry.crop = bridge.cv2_to_imgmsg(crop_bgr, encoding="bgr8")
        entry.crop.header = header
        msg.crops.append(entry)

    return msg
