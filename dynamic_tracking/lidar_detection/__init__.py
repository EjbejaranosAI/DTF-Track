"""Multi-detector LiDAR person detection (Spencer-style, single-node pipeline)."""

from .types import LidarExtractDebugStats, LidarFusionPolicy

__all__ = [
    "LidarExtractDebugStats",
    "LidarFusionPolicy",
    "LidarDetectionPipeline",
    "build_lidar_pipeline",
]


def __getattr__(name: str):
    if name == "LidarDetectionPipeline":
        from .pipeline import LidarDetectionPipeline

        return LidarDetectionPipeline
    if name == "build_lidar_pipeline":
        from .pipeline import build_lidar_pipeline

        return build_lidar_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
