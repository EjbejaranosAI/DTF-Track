"""Unit tests for LiDAR detector fusion."""

from dynamic_tracking.lidar_detection.fusion import fuse_segment_hypotheses
from dynamic_tracking.lidar_detection.types import (
    LidarFusionConfig,
    LidarFusionPolicy,
    LidarHypothesis,
)


def test_max_score_fusion():
    hyps = [
        LidarHypothesis(0, "spencer_rf", 0.12, False, 1.0, 2.0),
        LidarHypothesis(0, "leg_heuristic", 0.71, True, 1.0, 2.0),
    ]
    cfg = LidarFusionConfig(policy=LidarFusionPolicy.MAX_SCORE)
    assert fuse_segment_hypotheses(hyps, cfg)[0] == 0.71


def test_require_two():
    hyps = [
        LidarHypothesis(0, "spencer_rf", 0.8, True, 0.0, 0.0),
        LidarHypothesis(0, "blob", 0.85, True, 0.0, 0.0),
    ]
    cfg = LidarFusionConfig(policy=LidarFusionPolicy.REQUIRE_TWO)
    assert fuse_segment_hypotheses(hyps, cfg)[0] == 0.85

    hyps_one = [LidarHypothesis(0, "spencer_rf", 0.8, True, 0.0, 0.0)]
    assert fuse_segment_hypotheses(hyps_one, cfg) == {}
