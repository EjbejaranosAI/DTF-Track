"""Types for the multi-detector LiDAR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class LidarFusionPolicy(str, Enum):
    MAX_SCORE = "max_score"
    REQUIRE_TWO = "require_two"
    UNANIMOUS = "unanimous"
    ANY_HIGH = "any_high"


@dataclass
class LidarHypothesis:
    """One detector opinion for one scan segment."""

    segment_index: int
    detector_id: str
    score: float  # normalized to [0, 1] for fusion
    is_person: bool
    map_x: float = 0.0
    map_y: float = 0.0
    raw_score: float = 0.0


@dataclass
class LidarExtractDebugStats:
    """One /scan extraction snapshot (for periodic logging)."""

    n_segments: int = 0
    n_rf_person: int = 0
    n_adaboost_person: int = 0
    n_blob_person: int = 0
    n_heuristic_person: int = 0
    n_clusters: int = 0
    n_prefilter_skip: int = 0
    n_range_skip: int = 0
    n_tf_fail: int = 0
    max_rf: float = 0.0
    max_adaboost: float = 0.0
    max_blob: float = 0.0
    max_heuristic: float = 0.0
    max_leg_score: float = 0.0
    use_spencer_rf: bool = False
    use_spencer_adaboost: bool = False
    only_positive: bool = True
    fusion_policy: str = "max_score"
    enabled_detectors: List[str] = field(default_factory=list)
    per_detector_max: Dict[str, float] = field(default_factory=dict)


@dataclass
class LidarFusionConfig:
    policy: LidarFusionPolicy = LidarFusionPolicy.MAX_SCORE
    merge_radius_m: float = 0.0
    any_high_threshold: float = 0.70
    only_positive_rf: bool = True
    require_rf_for_only_positive: bool = True
