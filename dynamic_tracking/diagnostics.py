"""Pipeline timing and LiDAR diagnostic logging (extracted from node)."""

import math
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .scan_clustering import LidarExtractDebugStats


class PipelineTiming:
    """EMA-based pipeline Hz / YOLO latency logger."""

    def __init__(self, log_interval_sec: float) -> None:
        self.interval = log_interval_sec
        self._prev_wall: Optional[float] = None
        self._ema_gap: Optional[float] = None
        self._ema_yolo: Optional[float] = None
        self._last_log: Optional[float] = None

    def update(self, wall_start: float, wall_end: float, yolo_ms: float, logger) -> None:
        if self.interval <= 0:
            return
        a = 0.15
        if self._prev_wall is not None:
            gap = (wall_start - self._prev_wall) * 1000.0
            self._ema_gap = gap if self._ema_gap is None else a * gap + (1.0 - a) * self._ema_gap
            self._ema_yolo = yolo_ms if self._ema_yolo is None else a * yolo_ms + (1.0 - a) * self._ema_yolo
        self._prev_wall = wall_start

        if self._last_log is None:
            self._last_log = wall_end
        if (wall_end - self._last_log) < self.interval:
            return
        self._last_log = wall_end

        total_ms = (wall_end - wall_start) * 1000.0
        rest_ms = max(0.0, total_ms - yolo_ms)
        if self._ema_gap is not None and self._ema_gap > 1e-3:
            hz = 1000.0 / self._ema_gap
            logger.info(
                f"Pipeline timing (EMA): ~{hz:.2f} Hz, interval ~{self._ema_gap:.0f} ms, "
                f"rest ~{rest_ms:.0f} ms"
            )
        else:
            logger.info(f"Pipeline timing: cycle ~{total_ms:.0f} ms (rest ~{rest_ms:.0f} ms)")


class StageTimer:
    """Per-stage wall time (avg/max ms + Hz), thread-safe; logs every interval."""

    def __init__(self, log_interval_sec: float) -> None:
        self.interval = log_interval_sec
        self._lock = threading.Lock()
        self._stats: Dict[str, list] = {}
        self._t0: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def record(self, stage: str, ms: float) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            s = self._stats.setdefault(stage, [0, 0.0, 0.0])
            s[0] += 1
            s[1] += ms
            s[2] = max(s[2], ms)

    def lap(self, stage: str, t_start: float) -> float:
        """Record time since t_start (perf_counter) under stage; return now for the next lap."""
        now = time.perf_counter()
        self.record(stage, (now - t_start) * 1000.0)
        return now

    def log_if_due(self, logger) -> None:
        if self.interval <= 0:
            return
        now = time.perf_counter()
        with self._lock:
            if self._t0 is None:
                self._t0 = now
                return
            dt = now - self._t0
            if dt < self.interval:
                return
            stats, self._stats, self._t0 = self._stats, {}, now
        parts = [
            f"{name}: {n / dt:.1f}Hz avg={tot / n:.0f}ms max={mx:.0f}ms"
            for name, (n, tot, mx) in sorted(stats.items())
        ]
        logger.info("Stage timing (" + f"{dt:.1f}s): " + " | ".join(parts))


def map_xy_to_robot_frame(
    mx: float, my: float, robot_tx: float, robot_ty: float, robot_yaw: float
) -> tuple[float, float, float, float]:
    """Map (mx,my) -> robot frame (x_fwd, y_left) + (dist_m, bearing_deg)."""
    dx = mx - robot_tx
    dy = my - robot_ty
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    x_robot = c * dx + s * dy
    y_robot = -s * dx + c * dy
    dist = math.hypot(x_robot, y_robot)
    bearing = math.degrees(math.atan2(y_robot, x_robot))
    return x_robot, y_robot, dist, bearing


class PoseDebugLogger:
    """Log robot + track poses in map and robot frame (for drift diagnosis)."""

    def __init__(self, interval_sec: float, robot_frame: str = "base_link") -> None:
        self.interval = max(0.0, float(interval_sec))
        self.robot_frame = str(robot_frame).strip() or "base_link"
        self._last_log: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self.interval > 0.0

    def log_if_due(
        self,
        wall_sec: float,
        logger,
        *,
        tracking_frame: str,
        robot_map_tf,  # Optional[Tuple[tx, ty, yaw]] map<-robot
        robot_odom_tf,  # Optional[Tuple[tx, ty, yaw]] odom<-robot or map<-odom
        scan_tf_mode: str,
        scan_lag_sec: float,
        scan_frame: str,
        tracks,
        clusters,
        min_confirmed_hits: int = 2,
    ) -> None:
        if not self.enabled:
            return
        if self._last_log is not None and (wall_sec - self._last_log) < self.interval:
            return
        self._last_log = wall_sec

        lines = [f"[POSE] === t={wall_sec:.3f}s scan_tf={scan_tf_mode} lag={scan_lag_sec:.3f}s frame={scan_frame} ==="]

        if robot_map_tf is not None:
            tx, ty, yaw = robot_map_tf
            lines.append(
                f"[POSE] robot@{tracking_frame}: x={tx:+.3f} y={ty:+.3f} yaw={math.degrees(yaw):+.1f}°"
            )
        else:
            lines.append(f"[POSE] robot@{tracking_frame}: TF NOT AVAILABLE")

        if robot_odom_tf is not None:
            ox, oy, oyaw = robot_odom_tf
            lines.append(
                f"[POSE] robot@odom: x={ox:+.3f} y={oy:+.3f} yaw={math.degrees(oyaw):+.1f}°"
            )

        persons = [t for t in tracks if str(t.class_name).lower() == "person"]
        if not persons:
            lines.append("[POSE] tracks: no persons")
        elif robot_map_tf is None:
            for tr in sorted(persons, key=lambda t: t.track_id):
                mx, my = tr.state()
                tag = "OK" if tr.hits >= min_confirmed_hits else f"tent {tr.hits}/{min_confirmed_hits}"
                lines.append(f"[POSE] id={tr.track_id} map=({mx:+.3f},{my:+.3f}) {tag} (no robot TF)")
        else:
            rtx, rty, ryaw = robot_map_tf
            for tr in sorted(persons, key=lambda t: t.track_id):
                mx, my = tr.state()
                vx, vy = tr.velocity()
                xr, yr, dist, bear = map_xy_to_robot_frame(mx, my, rtx, rty, ryaw)
                tag = "OK" if tr.hits >= min_confirmed_hits else f"tent {tr.hits}/{min_confirmed_hits}"
                lines.append(
                    f"[POSE] id={tr.track_id} map=({mx:+.3f},{my:+.3f}) "
                    f"robot=({xr:+.3f},{yr:+.3f}) dist={dist:.2f}m bear={bear:+.1f}° "
                    f"v={math.hypot(vx, vy):.2f} {tag} m={tr.missed_frames}"
                )

        if clusters:
            lines.append(f"[POSE] clusters LiDAR ({len(clusters)}):")
            for i, cl in enumerate(clusters[:4]):
                mx, my = float(cl.map_x), float(cl.map_y)
                if robot_map_tf is not None:
                    rtx, rty, ryaw = robot_map_tf
                    xr, yr, dist, bear = map_xy_to_robot_frame(mx, my, rtx, rty, ryaw)
                    lines.append(
                        f"[POSE]   cl#{i} map=({mx:+.3f},{my:+.3f}) "
                        f"robot=({xr:+.3f},{yr:+.3f}) dist={dist:.2f}m leg={cl.leg_candidate_score:.2f}"
                    )
                else:
                    lines.append(f"[POSE]   cl#{i} map=({mx:+.3f},{my:+.3f}) leg={cl.leg_candidate_score:.2f}")
            if len(clusters) > 4:
                lines.append(f"[POSE]   ... +{len(clusters) - 4} more")

        for line in lines:
            logger.info(line)


class PersonTrackLogger:
    """Compact log of tracked persons (id + position)."""

    def __init__(self, log_interval_sec: float) -> None:
        self.interval = log_interval_sec
        self._last_log: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self.interval > 0.0

    def log_if_due(
        self,
        wall_sec: float,
        logger,
        tracks,
        *,
        min_confirmed_hits: int = 2,
        n_clusters: int = 0,
        yolo_detections: int = -1,
        yolo_ms: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        if self._last_log is not None and (wall_sec - self._last_log) < self.interval:
            return
        self._last_log = wall_sec
        all_person = [t for t in tracks if str(t.class_name).lower() == "person"]
        yolo_suffix = ""
        if yolo_detections >= 0:
            yolo_suffix = f" | YOLO: {yolo_detections} det ({yolo_ms:.0f}ms)"
        if not all_person:
            logger.info(f"Persons: none ({n_clusters} LiDAR clusters){yolo_suffix}")
            return
        parts = []
        for tr in sorted(all_person, key=lambda t: t.track_id):
            x, y = tr.state()
            vx, vy = tr.velocity()
            spd = math.hypot(vx, vy)
            tag = "OK" if tr.hits >= min_confirmed_hits else f"tent {tr.hits}/{min_confirmed_hits}"
            parts.append(
                f"id={tr.track_id} ({x:+.2f},{y:+.2f}) {tag} v={spd:.2f} m={tr.missed_frames}"
            )
        suffix = f"  [{n_clusters} clusters]" if n_clusters >= 0 else ""
        logger.info("Persons: " + " | ".join(parts) + suffix + yolo_suffix)


class LidarDiagnostics:
    """Accumulates and periodically logs LiDAR fusion statistics."""

    def __init__(self, log_interval_sec: float) -> None:
        self.interval = log_interval_sec
        self._last_log: Optional[float] = None
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.dets = 0
        self.ray_ok = 0
        self.ray_miss: Dict[str, int] = defaultdict(int)
        self.map_lidar = 0
        self.tf_map_fail = 0
        self.rgb_fallback = 0
        self.no_map = 0
        self.cl_rejected_wall = 0
        self.clusters_raw = 0
        self.clusters_kept = 0
        self.stamp_skew_sum = 0.0
        self.stamp_skew_n = 0
        self.track_sum = 0
        self.track_confirmed_sum = 0
        self.yolo_target_sum = 0
        self.max_leg_score_sum = 0.0
        self.max_leg_raw_sum = 0.0
        self.spawn_eligible_sum = 0
        self.spawn_eligible_raw_sum = 0

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def log_if_due(
        self,
        wall_now: float,
        enable_lidar_fusion: bool,
        map_filter_enable: bool,
        neighbor_radius_cells: int,
        logger,
        *,
        enable_yolo: bool = False,
        spencer_rf_ready: bool = False,
        spencer_only_positive: bool = False,
        pub_lidar_clusters: bool = False,
        scan_ok: bool = True,
        marker_topic: str = "",
        tracking_frame: str = "",
        viz_frame: str = "",
        spawn_min_leg_score: float = 0.0,
    ) -> None:
        if self.interval <= 0:
            return
        if self._last_log is None:
            self._last_log = wall_now
        if (wall_now - self._last_log) < self.interval:
            return
        self._last_log = wall_now
        if self.frames <= 0:
            return
        nf = float(self.frames)
        dropped = self.clusters_raw - self.clusters_kept
        parts = [
            f"Lidar diag ({int(nf)} cycles):",
            f"clusters raw/filtered={self.clusters_raw / nf:.1f}/"
            f"{self.clusters_kept / nf:.1f}/cycle (~{dropped / max(nf, 1.0):.1f} filtered)",
        ]
        if self.stamp_skew_n > 0:
            parts.append(f"| |img-scan stamp| avg={self.stamp_skew_sum / float(self.stamp_skew_n):.3f}s")
        parts.append(
            f"| tracks avg={self.track_sum / nf:.2f} (confirmed≥min avg={self.track_confirmed_sum / nf:.2f})"
        )
        if enable_yolo:
            parts.append(f"| YOLO target boxes/cycle={self.yolo_target_sum / nf:.2f}")
        if not scan_ok and enable_lidar_fusion:
            parts.append("| WARNING: no LaserScan yet — LiDAR clusters/tracks idle")
        avg_raw = self.clusters_raw / nf
        kept_avg = self.clusters_kept / nf
        if avg_raw < 0.25 and scan_ok and spencer_rf_ready and spencer_only_positive:
            parts.append(
                "| hint: ~no LiDAR clusters — Spencer may drop all segments "
                "(try spencer_rf_only_positive_classifications:=false)"
            )
        elif kept_avg >= 0.5 and (self.track_sum / nf) < 0.05:
            max_leg = self.max_leg_score_sum / nf
            max_raw = self.max_leg_raw_sum / nf
            spawn_ok = self.spawn_eligible_sum / nf
            spawn_raw = self.spawn_eligible_raw_sum / nf
            parts.append(
                f"| leg_score max raw≈{max_raw:.2f} kept≈{max_leg:.2f} "
                f"spawn_eligible raw/kept≈{spawn_raw:.1f}/{spawn_ok:.1f} per cycle (≥{spawn_min_leg_score:.2f})"
            )
            if spawn_min_leg_score > 1e-6 and max_leg < spawn_min_leg_score - 1e-6:
                if max_raw >= spawn_min_leg_score - 1e-6 and spawn_raw >= 0.25:
                    parts.append(
                        "| hint: person-like clusters exist BEFORE map/costmap filters but disappear after — "
                        "try static_map_filter_enable:=false and/or map_filter_enable:=false, "
                        "or lower static_map_disk_min_free_ratio"
                    )
                else:
                    parts.append(
                        f"| hint: no track spawn — max_score<{spawn_min_leg_score:.2f} "
                        f"(RF Gazebo ~0.05–0.15; with only_positive=false uses max(rf,heuristic)); "
                        f"lower lidar_spawn_person_min_leg_score or check the LiDAR extract log"
                    )
            else:
                parts.append(
                    "| hint: clusters seen but no tracks — check lidar_spawn_confirm_frames, map_filter, association"
                )
        if enable_lidar_fusion and not pub_lidar_clusters and kept_avg >= 0.25:
            parts.append("| RViz: cyan cluster spheres OFF (publish_lidar_cluster_markers:=true)")
        if marker_topic:
            rv = f"RViz track MarkerArray={marker_topic} frame_id={tracking_frame or '?'}"
            if viz_frame:
                rv += f" remap→{viz_frame}"
            parts.append(f"| {rv}")
        if enable_lidar_fusion and self.dets > 0:
            nd = float(self.dets)
            parts.append(
                f"| fusion/det: ray_ok {100.0 * self.ray_ok / nd:.0f}% "
                f"-> map_lidar {100.0 * self.map_lidar / nd:.0f}% "
                f"CL_wall {100.0 * self.cl_rejected_wall / nd:.0f}% "
                f"TF_fail {100.0 * self.tf_map_fail / nd:.0f}% "
                f"cam_fb {100.0 * self.rgb_fallback / nd:.0f}% "
                f"no_map {100.0 * self.no_map / nd:.0f}%"
            )
            if self.ray_miss:
                items = sorted(self.ray_miss.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
                parts.append(f"| miss: [{','.join(f'{k}={v}' for k, v in items)}]")
        elif enable_lidar_fusion:
            parts.append("| fusion: 0 target detections in interval")
        raw_tot, kept_tot = int(self.clusters_raw), int(self.clusters_kept)
        if map_filter_enable and neighbor_radius_cells > 0 and raw_tot > 80 and kept_tot < max(5, int(0.08 * raw_tot)):
            logger.warning(
                "Almost no LiDAR clusters pass costmap filter. "
                "Try map_filter_neighbor_radius_cells:=0 or different threshold."
            )
        elif map_filter_enable and neighbor_radius_cells == 0 and raw_tot > 80 and kept_tot < max(3, int(0.05 * raw_tot)):
            logger.warning(
                "Almost no clusters pass costmap (neighbor=0). "
                "Raise occupancy_threshold to ~253 or disable map_filter_enable."
            )
        logger.info(" ".join(parts))
        self.reset()


class SpencerExtractDiagnostics:
    """Periodic log of one /scan -> clusters pipeline (Spencer RF + heuristica)."""

    def __init__(self, log_interval_sec: float) -> None:
        self.interval = log_interval_sec
        self._last_log: Optional[float] = None
        self._acc_frames = 0
        self._acc_segments = 0
        self._acc_clusters = 0
        self._acc_rf_person = 0
        self._acc_prefilter = 0
        self._acc_range = 0
        self._acc_tf_fail = 0
        self._max_rf = 0.0
        self._max_heur = 0.0
        self._max_leg = 0.0
        self._spawn_eligible_sum = 0
        self._last_fusion_policy = "max_score"
        self._last_enabled_detectors: list = []

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def accumulate(
        self,
        stats: "LidarExtractDebugStats",
        spawn_min_leg_score: float,
    ) -> None:
        self._last_fusion_policy = getattr(stats, "fusion_policy", "max_score")
        self._last_enabled_detectors = list(getattr(stats, "enabled_detectors", []))
        self._acc_frames += 1
        self._acc_segments += stats.n_segments
        self._acc_clusters += stats.n_clusters
        self._acc_rf_person += stats.n_rf_person
        self._acc_prefilter += stats.n_prefilter_skip
        self._acc_range += stats.n_range_skip
        self._acc_tf_fail += stats.n_tf_fail
        self._max_rf = max(self._max_rf, stats.max_rf)
        self._max_heur = max(self._max_heur, stats.max_heuristic)
        self._max_leg = max(self._max_leg, stats.max_leg_score)
        if spawn_min_leg_score > 1e-9 and stats.max_leg_score >= spawn_min_leg_score - 1e-9:
            self._spawn_eligible_sum += 1

    def log_if_due(
        self,
        wall_now: float,
        logger,
        *,
        spawn_min_leg_score: float,
        rf_threshold: float,
        only_positive: bool,
        scan_frame: str,
    ) -> None:
        if self.interval <= 0 or self._acc_frames <= 0:
            return
        if self._last_log is None:
            self._last_log = wall_now
        if (wall_now - self._last_log) < self.interval:
            return
        self._last_log = wall_now
        nf = float(self._acc_frames)
        parts = [
            f"LiDAR extract ({int(nf)} scans, frame={scan_frame or '?'}):",
            f"segments/scan={self._acc_segments / nf:.1f}",
            f"clusters/scan={self._acc_clusters / nf:.1f}",
            f"| RF person@{rf_threshold:.2f}/scan={self._acc_rf_person / nf:.1f}",
            f"only_positive={only_positive}",
            f"fusion={self._last_fusion_policy}",
            f"detectors={self._last_enabled_detectors}",
            f"| skip prefilter={self._acc_prefilter / nf:.1f}",
            f"range={self._acc_range / nf:.1f}",
            f"TF_fail={self._acc_tf_fail / nf:.1f}",
            f"| max RF={self._max_rf:.2f}",
            f"max_heur={self._max_heur:.2f}",
            f"max_score={self._max_leg:.2f}",
            f"spawn_ok_scans={self._spawn_eligible_sum}/{int(nf)} (≥{spawn_min_leg_score:.2f})",
        ]
        if (
            self._acc_clusters / nf >= 1.0
            and self._max_leg < spawn_min_leg_score - 1e-6
            and self._max_heur >= spawn_min_leg_score - 1e-6
        ):
            parts.append(
                "| hint: heuristic OK but final score low — restart after rebuild "
                "(max(rf,heur) with only_positive=false)"
            )
        elif self._max_rf < 0.2 and self._max_heur < 0.35:
            parts.append(
                "| hint: RF and heuristic both low — check lidar_cluster_jump_m, legs in range, or disable RF"
            )
        logger.info(" ".join(parts))
        self._acc_frames = 0
        self._acc_segments = 0
        self._acc_clusters = 0
        self._acc_rf_person = 0
        self._acc_prefilter = 0
        self._acc_range = 0
        self._acc_tf_fail = 0
        self._max_rf = 0.0
        self._max_heur = 0.0
        self._max_leg = 0.0
        self._spawn_eligible_sum = 0
