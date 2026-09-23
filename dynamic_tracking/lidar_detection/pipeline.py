"""Single-node LiDAR detection pipeline: segment once, N detectors, fuse."""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Set, Tuple

from sensor_msgs.msg import LaserScan

from ..scan_clustering import LegHeuristicParams
from ..spencer_features import build_segments_from_scan
from ..types import LidarDualExtract, LidarMeasurement
from .detectors import (
    BlobDetector,
    LegHeuristicDetector,
    LidarDetector,
    SpencerAdaBoostDetector,
    SpencerRFDetector,
)
from .detectors.blob import BlobDetectorParams
from .fusion import fuse_segment_hypotheses, passes_only_positive_gate
from .segment import (
    cluster_from_indices,
    passes_segment_prefilter,
    prepare_scan_arrays,
    segment_pose_and_distance,
    segment_scan,
)
from .types import LidarExtractDebugStats, LidarFusionConfig, LidarFusionPolicy, LidarHypothesis
from .detectors.base import SegmentContext

TransformXY = Callable[[float, float, str, object], Optional[Tuple[float, float]]]


def resolve_model_path(path_raw: str) -> str:
    if os.path.isabs(path_raw) and os.path.isfile(path_raw):
        return path_raw
    pkg_share = ""
    try:
        from ament_index_python.packages import get_package_share_directory

        pkg_share = get_package_share_directory("dynamic_tracking")
    except Exception:
        pass
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(pkg_share, path_raw) if pkg_share else "",
        os.path.join(repo_root, path_raw),
        os.path.abspath(path_raw),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return path_raw


class LidarDetectionPipeline:
    def __init__(
        self,
        detectors: List[LidarDetector],
        fusion: LidarFusionConfig,
        *,
        fusion_spawn: Optional[LidarFusionConfig] = None,
        fusion_maintenance: Optional[LidarFusionConfig] = None,
        maintenance_min_leg_score: float = 0.0,
        cluster_jump_m: float,
        min_range_m: float,
        max_range_m: float,
        blind_zone_m: float,
        pose_mode: str,
        min_points: int,
        max_points: int,
        min_segment_width_m: float,
    ) -> None:
        self._detectors = [d for d in detectors if d.is_ready]
        self._fusion = fusion
        self._fusion_spawn = fusion_spawn
        self._fusion_maint = fusion_maintenance
        self._maint_min_score = float(maintenance_min_leg_score)
        self._jump = cluster_jump_m
        self._r_lo = max(min_range_m, blind_zone_m)
        self._r_hi = max_range_m
        self._pose_mode = pose_mode
        self._min_pts = min_points
        self._max_pts = max_points
        self._min_width = min_segment_width_m

    @property
    def enabled_detector_ids(self) -> List[str]:
        return [d.detector_id for d in self._detectors]

    @property
    def fusion_policy(self) -> str:
        return self._fusion.policy.value

    @property
    def dual_tier_enabled(self) -> bool:
        return self._fusion_spawn is not None and self._fusion_maint is not None

    def _build_measurements(
        self,
        fused_scores: dict,
        hyps_by_seg: dict,
        ctx_by_seg: dict,
        fusion: LidarFusionConfig,
        min_score: float,
        debug_stats: Optional[LidarExtractDebugStats],
        spawn_tier_segments: Optional[Set[int]] = None,
    ) -> List[LidarMeasurement]:
        measurements: List[LidarMeasurement] = []
        for seg_idx, leg_score in fused_scores.items():
            if leg_score < min_score:
                continue
            seg_hyps = hyps_by_seg.get(seg_idx, [])
            if not passes_only_positive_gate(seg_hyps, leg_score, fusion):
                continue
            ctx = ctx_by_seg.get(seg_idx)
            if ctx is None:
                continue
            if debug_stats is not None:
                debug_stats.max_leg_score = max(debug_stats.max_leg_score, leg_score)
            st_pass = (
                True
                if spawn_tier_segments is None
                else int(seg_idx) in spawn_tier_segments
            )
            measurements.append(
                LidarMeasurement(
                    map_x=ctx.map_x,
                    map_y=ctx.map_y,
                    leg_candidate_score=float(leg_score),
                    spawn_tier_pass=st_pass,
                )
            )
        return measurements

    def extract(
        self,
        scan: LaserScan,
        scan_frame_id: str,
        transform_xy_to_map: TransformXY,
        debug_stats: Optional[LidarExtractDebugStats] = None,
    ) -> List[LidarMeasurement]:
        seg_indices = segment_scan(scan, self._jump)
        if debug_stats is not None:
            debug_stats.n_segments = len(seg_indices)
            debug_stats.enabled_detectors = self.enabled_detector_ids
            debug_stats.fusion_policy = self._fusion.policy.value
            debug_stats.only_positive = self._fusion.only_positive_rf
            debug_stats.use_spencer_rf = "spencer_rf" in debug_stats.enabled_detectors
            debug_stats.use_spencer_adaboost = "spencer_adaboost" in debug_stats.enabled_detectors

        if not seg_indices or not self._detectors:
            if debug_stats is not None:
                debug_stats.n_clusters = 0
            return []

        angles, ranges, n_beams = prepare_scan_arrays(scan)
        spencer_segments = build_segments_from_scan(
            angles,
            ranges,
            seg_indices,
            scan.angle_min,
            scan.angle_min + (n_beams - 1) * scan.angle_increment,
        )

        contexts: List[SegmentContext] = []
        for seg_idx, idx_list in enumerate(seg_indices):
            cluster = cluster_from_indices(scan, idx_list)
            if not passes_segment_prefilter(
                cluster,
                min_points=self._min_pts,
                max_points=self._max_pts,
                min_range_m=self._r_lo,
                max_range_m=self._r_hi,
                min_segment_width_m=self._min_width,
            ):
                if debug_stats is not None:
                    debug_stats.n_prefilter_skip += 1
                continue
            x_l, y_l, d_pose = segment_pose_and_distance(cluster, self._pose_mode)
            if d_pose < self._r_lo or d_pose > self._r_hi:
                if debug_stats is not None:
                    debug_stats.n_range_skip += 1
                continue
            transformed = transform_xy_to_map(x_l, y_l, scan_frame_id, scan.header.stamp)
            if transformed is None:
                if debug_stats is not None:
                    debug_stats.n_tf_fail += 1
                continue
            sp_seg = spencer_segments[seg_idx] if seg_idx < len(spencer_segments) else None
            contexts.append(
                SegmentContext(
                    segment_index=seg_idx,
                    cluster=cluster,
                    map_x=float(transformed[0]),
                    map_y=float(transformed[1]),
                    distance_m=float(d_pose),
                    spencer_segment=sp_seg,
                )
            )

        all_hyps: List[LidarHypothesis] = []
        for det in self._detectors:
            all_hyps.extend(det.detect(contexts))

        if debug_stats is not None:
            self._update_debug_from_hypotheses(debug_stats, all_hyps)

        hyps_by_seg: dict[int, List[LidarHypothesis]] = {}
        for h in all_hyps:
            hyps_by_seg.setdefault(h.segment_index, []).append(h)

        ctx_by_seg = {c.segment_index: c for c in contexts}
        fused_main = fuse_segment_hypotheses(all_hyps, self._fusion)
        spawn_tier_seg = set(fused_main.keys())
        min_m = self._maint_min_score if self._maint_min_score > 0.0 else 0.0
        measurements = self._build_measurements(
            fused_main,
            hyps_by_seg,
            ctx_by_seg,
            self._fusion,
            min_m,
            debug_stats,
            spawn_tier_segments=spawn_tier_seg,
        )

        if debug_stats is not None:
            debug_stats.n_clusters = len(measurements)
        return measurements

    def extract_dual(
        self,
        scan: LaserScan,
        scan_frame_id: str,
        transform_xy_to_map: TransformXY,
        debug_stats: Optional[LidarExtractDebugStats] = None,
    ) -> LidarDualExtract:
        """One segmentation; strict fusion for spawn, lax fusion for association (ByteTrack-style)."""
        if not self.dual_tier_enabled:
            m = self.extract(scan, scan_frame_id, transform_xy_to_map, debug_stats)
            return LidarDualExtract(spawn=m, maintenance=m, n_segments=0)

        seg_indices = segment_scan(scan, self._jump)
        n_seg = len(seg_indices)
        if debug_stats is not None:
            debug_stats.n_segments = n_seg
            debug_stats.enabled_detectors = self.enabled_detector_ids
            debug_stats.fusion_policy = (
                f"spawn={self._fusion_spawn.policy.value},"
                f"maint={self._fusion_maint.policy.value}"
            )
            debug_stats.only_positive = self._fusion_spawn.only_positive_rf
            debug_stats.use_spencer_rf = "spencer_rf" in debug_stats.enabled_detectors
            debug_stats.use_spencer_adaboost = (
                "spencer_adaboost" in debug_stats.enabled_detectors
            )

        if not seg_indices or not self._detectors:
            return LidarDualExtract(spawn=[], maintenance=[], n_segments=n_seg)

        angles, ranges, n_beams = prepare_scan_arrays(scan)
        spencer_segments = build_segments_from_scan(
            angles,
            ranges,
            seg_indices,
            scan.angle_min,
            scan.angle_min + (n_beams - 1) * scan.angle_increment,
        )

        contexts: List[SegmentContext] = []
        for seg_idx, idx_list in enumerate(seg_indices):
            cluster = cluster_from_indices(scan, idx_list)
            if not passes_segment_prefilter(
                cluster,
                min_points=self._min_pts,
                max_points=self._max_pts,
                min_range_m=self._r_lo,
                max_range_m=self._r_hi,
                min_segment_width_m=self._min_width,
            ):
                if debug_stats is not None:
                    debug_stats.n_prefilter_skip += 1
                continue
            x_l, y_l, d_pose = segment_pose_and_distance(cluster, self._pose_mode)
            if d_pose < self._r_lo or d_pose > self._r_hi:
                if debug_stats is not None:
                    debug_stats.n_range_skip += 1
                continue
            transformed = transform_xy_to_map(x_l, y_l, scan_frame_id, scan.header.stamp)
            if transformed is None:
                if debug_stats is not None:
                    debug_stats.n_tf_fail += 1
                continue
            sp_seg = spencer_segments[seg_idx] if seg_idx < len(spencer_segments) else None
            contexts.append(
                SegmentContext(
                    segment_index=seg_idx,
                    cluster=cluster,
                    map_x=float(transformed[0]),
                    map_y=float(transformed[1]),
                    distance_m=float(d_pose),
                    spencer_segment=sp_seg,
                )
            )

        all_hyps: List[LidarHypothesis] = []
        for det in self._detectors:
            all_hyps.extend(det.detect(contexts))

        if debug_stats is not None:
            self._update_debug_from_hypotheses(debug_stats, all_hyps)

        hyps_by_seg: dict[int, List[LidarHypothesis]] = {}
        for h in all_hyps:
            hyps_by_seg.setdefault(h.segment_index, []).append(h)
        ctx_by_seg = {c.segment_index: c for c in contexts}

        fused_spawn = fuse_segment_hypotheses(all_hyps, self._fusion_spawn)
        fused_maint = fuse_segment_hypotheses(all_hyps, self._fusion_maint)
        spawn_seg = set(fused_spawn.keys())
        spawn = self._build_measurements(
            fused_spawn,
            hyps_by_seg,
            ctx_by_seg,
            self._fusion_spawn,
            0.0,
            debug_stats,
            spawn_tier_segments=spawn_seg,
        )
        min_m = self._maint_min_score if self._maint_min_score > 0.0 else 0.0
        maintenance = self._build_measurements(
            fused_maint,
            hyps_by_seg,
            ctx_by_seg,
            self._fusion_maint,
            min_m,
            None,
            spawn_tier_segments=spawn_seg,
        )

        if debug_stats is not None:
            debug_stats.n_clusters = len(maintenance)
        return LidarDualExtract(spawn=spawn, maintenance=maintenance, n_segments=n_seg)

    @staticmethod
    def _update_debug_from_hypotheses(
        stats: LidarExtractDebugStats, hyps: List[LidarHypothesis]
    ) -> None:
        for h in hyps:
            stats.per_detector_max[h.detector_id] = max(
                stats.per_detector_max.get(h.detector_id, 0.0), h.score
            )
            if h.detector_id == "spencer_rf":
                stats.max_rf = max(stats.max_rf, h.score)
                if h.is_person:
                    stats.n_rf_person += 1
            elif h.detector_id == "spencer_adaboost":
                stats.max_adaboost = max(stats.max_adaboost, h.score)
                if h.is_person:
                    stats.n_adaboost_person += 1
            elif h.detector_id == "blob":
                stats.max_blob = max(stats.max_blob, h.score)
                if h.is_person:
                    stats.n_blob_person += 1
            elif h.detector_id == "leg_heuristic":
                stats.max_heuristic = max(stats.max_heuristic, h.score)
                if h.is_person:
                    stats.n_heuristic_person += 1


def _make_fusion_config(
    policy_name: str,
    *,
    only_positive_rf: bool,
    any_high_threshold: float,
) -> LidarFusionConfig:
    policy = LidarFusionPolicy(policy_name.strip().lower())
    fusion = LidarFusionConfig(policy=policy, only_positive_rf=only_positive_rf)
    if policy == LidarFusionPolicy.ANY_HIGH:
        fusion.any_high_threshold = any_high_threshold
    return fusion


def build_lidar_pipeline(
    *,
    enabled: List[str],
    fusion_policy: str,
    fusion_any_high_threshold: float,
    only_positive_rf: bool,
    dual_tier_enable: bool = False,
    spawn_fusion_policy: str = "require_two",
    maintenance_fusion_policy: str = "max_score",
    maintenance_min_leg_score: float = 0.0,
    cluster_jump_m: float,
    min_track_distance_m: float,
    max_track_distance_m: float,
    blind_zone_m: float,
    pose_mode: str,
    min_points: int,
    max_points: int,
    min_segment_width_m: float,
    leg_params: LegHeuristicParams,
    leg_heuristic_score_threshold: float = 0.0,
    rf_model_path: str,
    rf_threshold: float,
    adaboost_model_path: str,
    adaboost_threshold: float,
    blob_params: BlobDetectorParams,
) -> LidarDetectionPipeline:
    fusion_spawn_cfg = None
    fusion_maint_cfg = None
    if dual_tier_enable:
        fusion_spawn_cfg = _make_fusion_config(
            spawn_fusion_policy,
            only_positive_rf=only_positive_rf,
            any_high_threshold=fusion_any_high_threshold,
        )
        fusion_maint_cfg = _make_fusion_config(
            maintenance_fusion_policy,
            only_positive_rf=only_positive_rf,
            any_high_threshold=fusion_any_high_threshold,
        )
        primary = fusion_maint_cfg
    else:
        primary = _make_fusion_config(
            fusion_policy,
            only_positive_rf=only_positive_rf,
            any_high_threshold=fusion_any_high_threshold,
        )

    detectors: List[LidarDetector] = []
    enabled_set = {e.strip().lower() for e in enabled if e.strip()}

    if "spencer_rf" in enabled_set and rf_model_path:
        path = resolve_model_path(rf_model_path)
        detectors.append(SpencerRFDetector(path, rf_threshold))

    if "spencer_adaboost" in enabled_set and adaboost_model_path:
        path = resolve_model_path(adaboost_model_path)
        detectors.append(SpencerAdaBoostDetector(path, adaboost_threshold))

    _leg_thr = float(leg_heuristic_score_threshold)
    if "leg_heuristic" in enabled_set and leg_params.enabled:
        detectors.append(LegHeuristicDetector(leg_params, score_threshold=_leg_thr))

    if "blob" in enabled_set:
        detectors.append(BlobDetector(blob_params))

    if not detectors and leg_params.enabled:
        detectors.append(LegHeuristicDetector(leg_params, score_threshold=_leg_thr))

    return LidarDetectionPipeline(
        detectors,
        primary,
        fusion_spawn=fusion_spawn_cfg,
        fusion_maintenance=fusion_maint_cfg,
        maintenance_min_leg_score=maintenance_min_leg_score,
        cluster_jump_m=cluster_jump_m,
        min_range_m=min_track_distance_m,
        max_range_m=max_track_distance_m,
        blind_zone_m=blind_zone_m,
        pose_mode=pose_mode,
        min_points=min_points,
        max_points=max_points,
        min_segment_width_m=min_segment_width_m,
    )


def extract_lidar_clusters(
    scan: LaserScan,
    pipeline: LidarDetectionPipeline,
    scan_frame_id: str,
    transform_xy_to_map: TransformXY,
    debug_stats: Optional[LidarExtractDebugStats] = None,
) -> List[LidarMeasurement]:
    return pipeline.extract(scan, scan_frame_id, transform_xy_to_map, debug_stats)


def extract_lidar_dual(
    scan: LaserScan,
    pipeline: LidarDetectionPipeline,
    scan_frame_id: str,
    transform_xy_to_map: TransformXY,
    debug_stats: Optional[LidarExtractDebugStats] = None,
) -> LidarDualExtract:
    return pipeline.extract_dual(scan, scan_frame_id, transform_xy_to_map, debug_stats)
