"""Fuse multi-detector hypotheses into one score per segment."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

from .types import LidarFusionConfig, LidarFusionPolicy, LidarHypothesis


def fuse_segment_hypotheses(
    hypotheses: List[LidarHypothesis],
    config: LidarFusionConfig,
) -> Dict[int, float]:
    """segment_index -> fused leg_candidate_score in [0, 1]."""
    by_seg: Dict[int, List[LidarHypothesis]] = defaultdict(list)
    for h in hypotheses:
        by_seg[h.segment_index].append(h)

    fused: Dict[int, float] = {}
    for seg_idx, hyps in by_seg.items():
        score = _fuse_one_segment(hyps, config)
        if score is not None:
            fused[seg_idx] = score
    return fused


def _fuse_one_segment(
    hyps: List[LidarHypothesis],
    config: LidarFusionConfig,
) -> Optional[float]:
    if not hyps:
        return None

    policy = config.policy
    scores = [h.score for h in hyps]
    persons = [h for h in hyps if h.is_person]

    if policy == LidarFusionPolicy.MAX_SCORE:
        return max(scores)

    if policy == LidarFusionPolicy.ANY_HIGH:
        if any(s >= config.any_high_threshold for s in scores):
            return max(scores)
        return max(scores) * 0.5 if scores else None

    if policy == LidarFusionPolicy.REQUIRE_TWO:
        if len(persons) >= 2:
            return max(h.score for h in persons)
        return None

    if policy == LidarFusionPolicy.UNANIMOUS:
        if len(persons) == len(hyps) and len(hyps) >= 1:
            return min(h.score for h in persons)
        return None

    return max(scores)


def passes_only_positive_gate(
    hyps: List[LidarHypothesis],
    fused_score: float,
    config: LidarFusionConfig,
) -> bool:
    """When only_positive_rf: keep segment if RF (if enabled) says person OR fused still useful."""
    if not config.only_positive_rf:
        return True
    rf_hyps = [h for h in hyps if h.detector_id == "spencer_rf"]
    if not rf_hyps:
        return True
    if config.require_rf_for_only_positive:
        return any(h.is_person for h in rf_hyps)
    return any(h.is_person for h in rf_hyps) or fused_score > 0.05


def detector_ids_present(hyps: List[LidarHypothesis]) -> Set[str]:
    return {h.detector_id for h in hyps}
