"""Unit tests for YOLO ONNX post-processing (no model file required)."""

import numpy as np

from dynamic_tracking.yolo_backends import (
    letterbox_image,
    parse_e2e_predictions,
    scale_boxes_to_original,
)


def _is_person(name: str) -> bool:
    return name.lower() == "person"


def test_parse_e2e_filters_confidence_and_class():
    pred = np.array([[
        [10.0, 20.0, 50.0, 120.0, 0.9, 0.0],
        [5.0, 5.0, 15.0, 25.0, 0.2, 0.0],
        [1.0, 2.0, 3.0, 4.0, 0.8, 2.0],
    ]], dtype=np.float32)
    names = {0: "person", 2: "car"}
    out = parse_e2e_predictions(pred, 0.5, names, 10, _is_person)
    assert len(out) == 1
    assert out[0][4] == "person"
    assert abs(out[0][5] - 0.9) < 1e-5


def test_letterbox_and_scale_boxes_roundtrip():
    im = np.zeros((480, 640, 3), dtype=np.uint8)
    lb, ratio, pad = letterbox_image(im, (256, 256))
    assert lb.shape == (256, 256, 3)
    boxes = np.array([[50.0, 60.0, 150.0, 200.0]], dtype=np.float32)
    back = scale_boxes_to_original(boxes, ratio, pad, (480, 640))
    assert back.shape == (1, 4)
    assert 0.0 <= back[0, 0] <= 640.0
    assert 0.0 <= back[0, 1] <= 480.0
