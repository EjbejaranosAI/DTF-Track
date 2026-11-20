"""Area filter zones: rectangles, polygons and the config/zones/ files."""

import os

import pytest
from builtin_interfaces.msg import Time

from dynamic_tracking.map_filters import AreaFilter, load_zones_file, parse_zone
from dynamic_tracking.track_visualization import build_area_zone_marker_array

ZONES_DIR = os.path.join(os.path.dirname(__file__), "..", "config", "zones")

# L-shaped (concave) polygon: 4x4 square minus the top-right 2x2 quadrant.
L_SHAPE = {"polygon": [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]}


def test_polygon_inside_outside_concave():
    f = AreaFilter([parse_zone(L_SHAPE)], "map")
    assert f._passes(1.0, 1.0)
    assert f._passes(1.0, 3.0)
    assert not f._passes(3.0, 3.0)  # the cut-out corner
    assert not f._passes(-0.5, 1.0)


def test_polygon_margin_grows_zone():
    f = AreaFilter([parse_zone(L_SHAPE)], "map")
    assert not f._passes(-0.3, 1.0)
    assert f._passes(-0.3, 1.0, margin_m=0.5)
    assert f._passes(2.3, 2.3, margin_m=0.5)  # near the inner corner
    assert not f._passes(3.5, 3.5, margin_m=0.5)


def test_exclude_polygon_inside_rectangle():
    zones = [
        parse_zone({"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10}),
        parse_zone({"add": False, "polygon": [[4, 4], [6, 4], [5, 6]]}),
    ]
    f = AreaFilter(zones, "map")
    assert f._passes(1.0, 1.0)
    assert not f._passes(5.0, 4.5)


def test_parse_zone_drops_closing_point_and_rejects_short():
    z = parse_zone({"polygon": [[0, 0], [1, 0], [1, 1], [0, 0]]})
    assert len(z["polygon"]) == 3
    with pytest.raises(ValueError):
        parse_zone({"polygon": [[0, 0], [1, 0]]})


def test_lab_rectangle_file_matches_old_defaults():
    frame, zones = load_zones_file(os.path.join(ZONES_DIR, "lab_rectangle.yaml"))
    assert frame == "map"
    assert zones[0]["x_min"] == 1.0 and zones[0]["y_max"] == 1.3


def test_lab_perimeter_file():
    frame, zones = load_zones_file(os.path.join(ZONES_DIR, "lab_perimeter.yaml"))
    assert frame == "map" and len(zones) == 1
    f = AreaFilter(zones, frame)
    assert f._passes(0.0, 0.0)  # near the robot start
    assert f._passes(8.0, -4.0)
    assert not f._passes(2.9, 1.8)  # chairs/tables outside
    assert not f._passes(4.75, -1.0)  # notch left out of the perimeter


def test_marker_fill_triangulates_polygon():
    frame, zones = load_zones_file(os.path.join(ZONES_DIR, "lab_perimeter.yaml"))
    arr = build_area_zone_marker_array(zones, frame, Time())
    fill, border = arr.markers
    n = len(zones[0]["polygon"])
    assert len(fill.points) == 3 * (n - 2)
    assert len(border.points) == n + 1


def test_track_max_travel_from_spawn():
    from dynamic_tracking.types import Track

    tr = Track(1, "person", 0.0, 0.0, 0.9, 0.0)
    tr.append_pose_trail_sample()
    assert tr.max_travel_from_spawn_m == pytest.approx(0.0)
    for _ in range(20):
        tr.correct(1.0, 0.0)
    tr.append_pose_trail_sample()
    for _ in range(20):
        tr.correct(0.0, 0.0)
    tr.append_pose_trail_sample()
    assert tr.max_travel_from_spawn_m > 0.5  # keeps the max even after coming back
