"""Spencer blob detector rules (width + point count, no ML)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from ..types import LidarHypothesis
from .base import LidarDetector, SegmentContext


@dataclass
class BlobDetectorParams:
    min_points: float = 4.0
    max_points: float = 25.0
    min_width_m: float = 0.05
    max_width_m: float = 0.55
    max_range_m: float = 20.0
    person_score: float = 0.85


class BlobDetector(LidarDetector):
    """Port of srl_laser_detectors/naive_detectors/blob_detector.cpp criteria."""

    def __init__(self, params: BlobDetectorParams | None = None) -> None:
        self._p = params or BlobDetectorParams()

    @property
    def detector_id(self) -> str:
        return "blob"

    def detect(self, segments: List[SegmentContext]) -> List[LidarHypothesis]:
        out: List[LidarHypothesis] = []
        for ctx in segments:
            score, is_person = self._score_cluster(ctx.cluster, ctx.distance_m)
            out.append(
                LidarHypothesis(
                    segment_index=ctx.segment_index,
                    detector_id=self.detector_id,
                    score=score,
                    is_person=is_person,
                    map_x=ctx.map_x,
                    map_y=ctx.map_y,
                    raw_score=score,
                )
            )
        return out

    def _score_cluster(
        self, cluster: List[Tuple[float, float]], distance_m: float
    ) -> Tuple[float, bool]:
        n = len(cluster)
        if n < self._p.min_points or n > self._p.max_points:
            return 0.0, False
        if distance_m > self._p.max_range_m:
            return 0.0, False
        xs = [r * math.cos(a) for a, r in cluster]
        ys = [r * math.sin(a) for a, r in cluster]
        width = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        if width < self._p.min_width_m or width > self._p.max_width_m:
            return 0.0, False
        return self._p.person_score, True
