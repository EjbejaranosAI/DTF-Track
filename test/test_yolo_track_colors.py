"""Three RViz colours: green / teal coast / blue."""

from dynamic_tracking.track_visualization import track_color_bgr
from dynamic_tracking.types import Track


def _person_track(**kwargs) -> Track:
    tr = Track(1, "person", 0.0, 0.0, 0.8, 10.0)
    for k, v in kwargs.items():
        setattr(tr, k, v)
    tr.last_update_source = "lidar-only"
    return tr


def test_track_color_lidar_only_blue():
    tr = _person_track()
    b, g, r = track_color_bgr(tr, 0.12, 0.25, stamp_sec=10.0, yolo_visual_lidar_parity=True)
    assert (b, g, r) == (220, 90, 40)


def test_track_color_person_green():
    tr = _person_track(last_rgb_person_sec=10.0)
    b, g, r = track_color_bgr(tr, 0.12, 0.25, stamp_sec=10.2, yolo_visual_lidar_parity=True)
    assert (b, g, r) == (0, 255, 0)


def test_track_color_yolo_coast_teal():
    tr = _person_track(last_rgb_person_sec=9.5)
    b, g, r = track_color_bgr(
        tr, 0.12, 0.25, stamp_sec=10.0,
        yolo_visual_lidar_parity=True,
        person_rgb_coast_cl_sec=4.0,
    )
    assert (b, g, r) == (180, 200, 40)


def test_track_color_yolo_coast_teal_after_long_gap():
    """After YOLO confirmation, teal while the ID lives (does not expire after 4 s)."""
    tr = _person_track(last_rgb_person_sec=1.0)
    b, g, r = track_color_bgr(
        tr, 0.12, 0.25, stamp_sec=100.0,
        yolo_visual_lidar_parity=True,
        person_rgb_coast_cl_sec=4.0,
    )
    assert (b, g, r) == (180, 200, 40)
