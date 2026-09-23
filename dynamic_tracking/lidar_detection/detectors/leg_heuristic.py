"""Geometric leg heuristic (Spencer-inspired multi-feature score)."""

from __future__ import annotations

from typing import List

from ...scan_clustering import LegHeuristicParams, compute_leg_candidate_score
from ..types import LidarHypothesis
from .base import LidarDetector, SegmentContext


class LegHeuristicDetector(LidarDetector):
    def __init__(self, params: LegHeuristicParams, score_threshold: float = 0.0) -> None:
        self._params = params
        self._threshold = float(score_threshold)

    @property
    def detector_id(self) -> str:
        return "leg_heuristic"

    @property
    def is_ready(self) -> bool:
        return self._params.enabled

    def detect(self, segments: List[SegmentContext]) -> List[LidarHypothesis]:
        if not self.is_ready:
            return []
        out: List[LidarHypothesis] = []
        for ctx in segments:
            score = compute_leg_candidate_score(ctx.cluster, self._params)
            is_person = score >= self._threshold
            out.append(
                LidarHypothesis(
                    segment_index=ctx.segment_index,
                    detector_id=self.detector_id,
                    score=float(score),
                    is_person=is_person,
                    map_x=ctx.map_x,
                    map_y=ctx.map_y,
                    raw_score=float(score),
                )
            )
        return out
