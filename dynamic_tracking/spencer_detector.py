"""Backward-compatible wrapper around :class:`SpencerRFDetector`."""

from __future__ import annotations

from typing import List, Tuple

from .lidar_detection.detectors.spencer_learned import SpencerRFDetector
from .spencer_features import Segment


class SpencerLaserDetector:
    """Legacy API; prefer :class:`lidar_detection.pipeline.LidarDetectionPipeline`."""

    def __init__(self, model_path: str, decision_threshold: float = 0.5) -> None:
        self._inner = SpencerRFDetector(model_path, decision_threshold)

    @property
    def is_ready(self) -> bool:
        return self._inner.is_ready

    def classify(self, seg: Segment) -> Tuple[bool, float]:
        is_person, score, _ = self._inner._classify_segment(seg)
        return is_person, score

    def classify_batch(self, segments: List[Segment]) -> List[Tuple[bool, float]]:
        return [self.classify(seg) for seg in segments]
