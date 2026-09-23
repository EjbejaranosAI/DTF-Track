"""Base class for plug-in LiDAR segment classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from ..types import LidarHypothesis

if TYPE_CHECKING:
    from ...spencer_features import Segment


@dataclass
class SegmentContext:
    """Shared per-segment geometry (laser frame + map pose if available)."""

    segment_index: int
    cluster: List[tuple]
    map_x: float
    map_y: float
    distance_m: float
    spencer_segment: "Segment | None" = None


class LidarDetector(ABC):
    """Classify scan segments; one instance per detector type."""

    @property
    @abstractmethod
    def detector_id(self) -> str:
        ...

    @property
    def is_ready(self) -> bool:
        return True

    @abstractmethod
    def detect(self, segments: List[SegmentContext]) -> List[LidarHypothesis]:
        """Return one hypothesis per segment this detector scores (may be all segments)."""
