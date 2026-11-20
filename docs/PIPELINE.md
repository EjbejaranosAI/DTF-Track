# `dynamic_tracking` pipeline — technical reference

## Data flow (one cycle with `sync_image_with_lidar`)

LiDAR-only detail (detectors and fusion): [LIDAR_DETECTORS.md](LIDAR_DETECTORS.md).  
Tracking (Kalman, verify, Spencer): [TRACKING.md](TRACKING.md).

### Pipeline diagram

![dynamic_tracking flow: sync, multi-detector LiDAR, auxiliary YOLO, ROS outputs](../imgs/pipeline_flow.png)

| Diagram colour | Block |
|----------------------|--------|
| Blue | Input (`Image`, `LaserScan`) |
| Green | Laser pipeline (segment → detectors → fusion → filters → Kalman) |
| Yellow | Auxiliary YOLO (metadata; does **not** move the Kalman position) |
| Purple | Output topics |

**Legend:** the 3D position of a track is defined **by the laser only**. YOLO confirms class/colour (`Person` / `DynObct`) and the image bbox.

### Laser branch only (detail)

![Parallel LiDAR detectors and fusion](../imgs/lidar_detectors_flow.png)

### Editing or regenerating the image

- Mermaid source: [`docs/diagrams/pipeline_flow.mmd`](diagrams/pipeline_flow.mmd) and [`lidar_detectors_flow.mmd`](diagrams/lidar_detectors_flow.mmd)
- Preview at [mermaid.live](https://mermaid.live)
- Regenerate the PNGs with the Mermaid CLI (`npm install -g @mermaid-js/mermaid-cli`):

```bash
mmdc -i docs/diagrams/pipeline_flow.mmd -o imgs/pipeline_flow.png -b white -s 2
mmdc -i docs/diagrams/lidar_detectors_flow.mmd -o imgs/lidar_detectors_flow.png -b white -s 2
```

## LiDAR first (default behaviour)

**Design:** the track **position** (Kalman `correct()`) is defined **by the LiDAR only** in the current node (`update_tracks(..., yolo_meas=[])`). **YOLO** adds **image metadata** (class, score, `last_bbox`, `last_rgb_person_sec`, "green" colour in RViz) through `apply_yolo_auxiliary_person_confirmation` **without** moving the state. With **`yolo_auxiliary_project_tracks_to_image: true`**, **phase 0** is filled: the track is projected to `(u,v)` at several typical Z heights (`yolo_auxiliary_projection_height_m`, floor), using `CameraInfo` + the optical←`tracking_frame` TF, and it matches if the point falls inside the YOLO `person` bbox (optional margin in px). **`yolo_debug_laser_polar_markers`** publishes `/dynamic_tracking/yolo_laser_polar_markers` (MarkerArray, `frame_id` = latest `/scan`): **green** segments = ray to the YOLO hit in the laser plane, **cyan** = `(r,θ)` of each track (to compare alignment with the polar phase).

**Recommended (robust) LiDAR cascade:** `/scan` segmentation (once) → **N detectors** (`lidar_detectors_enabled`: RF, AdaBoost, heuristic, blob) → **fusion** (`lidar_fusion_policy`) → costmap/static filter → association + Kalman. See [LIDAR_DETECTORS.md](LIDAR_DETECTORS.md).

**Dual-tier phase 2 (`lidar_dual_tier_enable: true`):** a single `/scan` produces two lists: **spawn** (`require_two`), only to open IDs; **maintenance** (`max_score`), for association and `correct()`. **Audit** (`person_track_dual_audit_*`): every N cycles, an unmatched `person` track must have a nearby cluster with `spawn_tier_pass` (≥2 detectors) or a recent YOLO; otherwise `scan_verify` does not keep it alive.

After **phase 1** (`map_xy`), **`yolo_auxiliary_polar_match_enable`** enables **phase 1b**: each detection stores the **(range, angle)** of the `/scan` hit in the bbox direction; it is compared with the **(r, θ)** of the track projected into the laser frame (useful when the TF to the map fails). **`yolo_auxiliary_bearing_match_enable`**: **phase 3** compares the angle `atan2(py,px)` in the laser frame between the track's floor point `(x,y,0)` and the ray to the centre/bottom of the bbox (same geometry as the fusion ray), without relying on the pinhole projection to a pixel.

| Config | Tracking + 3D position in RViz |
|--------|----------------------------------|
| `enable_yolo_detections: false` | 100 % laser; no YOLO inference. |
| `enable_yolo_detections: true` and **`yolo_auxiliary_person_only: true`** | **Same as the row above**: `update_tracks(..., yolo_meas=[])` — YOLO does **not** enter the Kalman or the Hungarian. Only `apply_yolo_auxiliary_person_confirmation` marks `person`/green and the image bbox on the existing laser track. Markers use **`tr.state()`** (laser correction only). |
| `yolo_auxiliary_person_only: false` | Legacy: YOLO in map association. |

## Frames and transforms

| Data | Typical frame | Notes |
|------|----------------|-------|
| Laser points | `scan.header.frame_id` (e.g. `base_laser_link`) | Range + angle → local laser `(x,y)`. **`lidar_blind_zone_radius_m`**: a cluster is rejected if its centre is closer to the laser origin than `max(min_track_distance_m, blind_zone)` (removes chassis reflections, which on the map seemed to "move with the robot") |
| Projection to the map | `tracking_frame` (e.g. `map`) | `lookup_transform(map, laser, stamp)`; on failure, retry with "latest" |
| Ray from the camera | `image.header.frame_id` (optical) | Optical→laser TF for the bearing; scan from the **same** sync callback |
| `/map` filter on clusters | `tracking_frame` → TF → `map` | By default only the cluster **centre** on `/map` (`static_map_neighbor_radius_cells`). Optional: `static_map_corner_probe_enable` → **cross** probes (5 points) if the centre is free but ≥2 cardinal points are occupied. **Spencer disk** (`static_map_disk_filter_enable`): counts free cells in a circle of radius `static_map_disk_radius_m`; if the free fraction ≤ `static_map_disk_min_free_ratio`, it is discarded (corners, things against a wall). No cells inside the map: `static_map_disk_permit_outside_map` |
| Laser `person` association | Phase 1: score ≥ `max(0.58, leg_min×0.88)`. **Phase 1b** (only `person` without RGB): score ≥ `max(0.38, min(leg_min×0.52, hi−ε), hi−0.08)` with `hi` = phase 1 threshold — prevents walls at ~0.45–0.60 from correcting the track when working LiDAR-only |
| Duplicates on the same point | Map IoU **NMS** (`person_track_map_merge_min_iou`) + optional Spencer **temporal** check (`duplicate_track_*`): last N Kalman states closer than `duplicate_track_max_dist_m` in M of N cycles → the worse track is deleted | `pose_trail_ring` feeds the duplicate check and, if `publish_track_trajectory_markers`, the RViz **LINE_STRIP** (`dynamic_tracks_trail`) |
| Nav2 costmap filter | `costmap` in `map` or `odom` | `occupancy_threshold` (253 = inscribed/lethal). `map_filter_inflation_threshold`: almost always **0** because the global costmap includes **dynamic** obstacles (it inflates people). Margin on walls only: **`/map` + `static_map_neighbor_radius_cells`** |
| Camera→map fallback | optical | `transform_point_to_map` with fixed depth or pixel depth |
| RViz markers | By default **empty `marker_visualization_frame`** = markers in **`tracking_frame`** (`odom`); RViz with Fixed Frame `map` applies the **`odom→map`** TF (recommended). With **`marker_visualization_frame: map`**, the node remaps the vertices (order: fusion stamp, optionally image, `latest` only if `marker_remap_allow_latest_fallback`). With **`yolo_visual_lidar_parity`**: `last_update_source` **`lidar-only`**; **green** = YOLO saw `person` ≤ **`person_yolo_seen_green_sec`** ago. 3D position; `TEXT_VIEW_FACING` text uses translation only. |

**Planar approximation:** `_transform_xy_to_map` uses only the TF **yaw** (2D floor). Valid for mobile bases; wrong if the laser is not in the horizontal plane of the map.

**Temporal consistency:** clusters use `scan.header.stamp`; with `lidar_fusion_use_scan_stamp_for_tf` and `/scan` available, `fusion_tf_stamp` is **always** the scan stamp (also in async image+laser mode), so it matches `_transform_xy_to_map` and the `odom→map` remap. Without LiDAR (`enable_lidar_fusion: false`), the image stamp is used. The **Kalman Δt** (`predict`) uses the interval between **`fusion_tf_stamp` stamps** (not between image callback arrivals), so it is not corrupted when the callback runs long (e.g. waiting for YOLO: keep **`yolo_sync_wait_max_sec: 0`**). **`/detections` and `/detections/tracked`** use `header.frame_id = tracking_frame` and that same stamp (like `trackedPersons.header.frame_id = world_frame` in Spencer's `ros_interface.cpp`). An average image–scan offset is normal; `sensor_sync_slop` must cover it. If RViz is still misaligned with `marker_visualization_frame: map`, try **Fixed Frame = `odom`** and leave `marker_visualization_frame` empty (no remap), like the default `world_frame` of the Spencer tracker.

## Kalman (OpenCV `cv2.KalmanFilter`)

- **State:** `[x, y, vx, vy]^T` in the **map**.
- **Model:** constant velocity; `F` applies `x += vx*dt`, `y += vy*dt` with the cycle `dt` (capped by `kalman_predict_dt_cap_sec` in the node).
- **Measurement:** the Kalman position `(x, y)` is defined **by the laser only** (or by the prediction if there is no `correct()` that cycle). With `person_yolo_defer_position_to_lidar: true` and `enable_lidar_fusion: true`, YOLO **never** calls `correct()` for the `person` class: it only confirms identity (class, image bbox, `last_rgb_person_sec`, fusion colour in RViz). Even if a cycle has **no** `lidar_meas`, the YOLO map projection is not used for the state (before, without clusters, it fell back to YOLO). With `person_yolo_defer_position_to_lidar: false`, a person can be corrected with YOLO on the map as before. With `person_lidar_refinement_after_yolo`, the track stays in laser association even after a camera match. **`person_yolo_match_by_iou_first`** (default `true` with defer+laser): before the map Hungarian, **person↔YOLO person** are paired by **image IoU** (threshold `person_yolo_assoc_iou_min`), so identity does not depend on a `map_xy` that is unstable while turning; other classes and unmatched detections still use map distance. **`yolo_lidar_fusion_label_only: true`**: YOLO does not `correct()` on the map for **any** class and does not create tracks from the camera; only label/bbox/RGB; position = laser + prediction. Tracks that match YOLO (any class) are fed back into the laser Hungarian (`all_yolo_matched_ti`), not only `person` ones, so they are not excluded and do not drift on `predict` alone. `person_rgb_coast_cl_sec` is **visual only**. Optional exclusion from laser matching is `person_hold_skip_lidar_association_sec` (**0** recommended). With defer disabled and laser in the same cycle, the refinement can weaken the YOLO correction (`person_yolo_weak_scale_with_lidar_refinement`).
- **Laser gate:** `person_lidar_innovation_gate_m` only applies to a `person` with a recent `last_rgb_person_sec` within `person_lidar_innovation_gate_stale_rgb_sec`; outside that window (or with no RGB ever), the gate is not applied — association up to `tracking_person_lidar_assoc_cost_threshold`. **With `enable_yolo_detections=false`, `last_rgb_person_sec` stays at 0 and the gate never acts** (that is why laser-only looks stable). **With YOLO on**, the same gate can **veto** track↔cluster pairs if the innovation exceeds the threshold (e.g. while turning); if `person_yolo_defer_position_to_lidar` or `yolo_lidar_fusion_label_only` is also on, the pipeline **disables** that gate for laser DA, because the position must no longer depend on "recent RGB".
- **Laser phase 1b (loose floor):** by default only `person` tracks with `last_rgb_person_sec<=0` enter. If YOLO is on and a track already has RGB but there is **no** high-score cluster that frame, it used to go neither to 1b nor to phase 2 (only `unknown`) → cycles **without** a laser `correct()` → drift that could affect many boxes at once. With defer/label-only, 1b also includes those RGB `person` tracks to re-anchor them to the laser.
- **Annotated image:** each track carries `last_image_bbox_stamp_sec` when the bbox comes from YOLO in that cycle. With `image_track_overlay_max_age_sec` > 0, the second layer (ID/label above the bbox) is not drawn if there was no recent detection — avoids "stuck" boxes when only the laser updates the map. YOLO's own boxes in the same callback are still only the current frame's detections.
- **Duplicate IDs:** if map-distance association leaves a YOLO measurement without a track but its IoU with an unmatched track exceeds `yolo_merge_unmatched_iou`, the same update as a camera match is applied instead of opening a new track.
- **Noise:** base `processNoiseCov` / `measurementNoiseCov` in `Track` (OpenCV); **every** laser correction uses the same `correct()` without scaling `R` by "recent RGB" (YOLO does not change the cluster weight).

**Possible improvements (not implemented):** different `R` matrices for camera/laser; a full EKF if explicit orientation is included.

## Association

- **Algorithm:** global **Hungarian** assignment (`scipy.optimize.linear_sum_assignment`; greedy fallback without SciPy).
- **Cost:** **Euclidean** distance on the map (prediction `association_position` ↔ measurement), or **squared Mahalanobis** + **chi²** (2 DOF, 99% ≈ 9.21) if `use_mahalanobis_lidar: true` — aligned with the gating of Spencer's `global_nearest_neighbor`.
- **Class:** `assoc_class_key(name) = hash(lower(strip(name)))`; `unknown` tracks can associate with any YOLO class.
- **LiDAR:** per-track threshold (`person` may use `tracking_person_lidar_assoc_cost_threshold`).

## Parity with `srl_nearest_neighbor_tracker` (Spencer)

We do not run the ROS1 C++ binary, but the pipeline already reflects several ideas of the Spencer tracker:

| Spencer idea | In this repo |
|--------------|----------------|
| GNN + innovation / Mahalanobis | `use_mahalanobis_lidar` + `Track.mahalanobis_sq` + `CHI2_99_2DOF` in `association.py` |
| MISSED vs OCCLUSION (outside the laser FOV) | `is_track_in_lidar_fov` + `missed_frames` vs `consecutive_occlusions` in `track_pipeline.py` |
| Different limits for mature / immature tracks | `max_occlusions_before_deletion`, `max_occlusions_mature_track`, `hits` and `_MATURE_HIT_THRESHOLD` |
| Occluded update (prediction without measurement) | `Track.occluded_update()` |
| Presence check in the scan | `verify_track_in_scan` |
| Initiator / spawn logic | Clusters + laser spawn + `/map` filters; not a line-by-line port of `logic_initiator.cpp` |
| `filter_detections_by_static_map` (free/occupied disk) | `static_map_disk_*` in `dynamic_tracking_node.py` after the centre/cross probes |
| Debounce of LiDAR-only `person` spawn | `lidar_spawn_confirm_frames` / `lidar_spawn_confirm_radius_m` + `PendingLidarSpawn` in `track_pipeline.py` (0 = no debounce) |
| Temporal duplicates (NNH) | `duplicate_track_temporal_enable` + `pose_trail_ring` | Aligned with `deleteDuplicateTracks` of the C++ Spencer tracker |
| Single stamp for the fused message + TF at that stamp | `fusion_tf_stamp` for the YOLO ray and for the markers/remap `header` | Avoids mixing geometry from one instant with `map`↔`odom` from another |
| Laser → camera cascade (no double tracking) | **`yolo_auxiliary_person_only`** (default) | `update_tracks` receives an **empty list** of YOLO measurements: prediction/correction/spawn = LiDAR only. `apply_yolo_auxiliary_person_confirmation`: distance match only with the **laser-ray `map_xy`** (`lidar_rgb`) if **`yolo_auxiliary_distance_match_lidar_ray_only`**; tracks below the hit threshold (**`yolo_auxiliary_min_track_hits`**, 0 = `tracking_min_confirmed_hits`) are not confirmed; `rgb_only` by image IoU only. NMS without `merge_lidar_only` with RGB. `false` = legacy (YOLO in the Hungarian) |

**Not ported** (would need new work): the full `LaserShadeOcclusionManager` (shadow sampling between the laser and synchronized segmentation), `polygon_occlusion_manager`, Spencer's extended 4D EKF, a separate per-modality `max_gating_distance` parameter. For 2D LiDAR-only tracking the above is usually enough; if you need shadows between two lasers, the next step would be additional logic or keep using the Spencer node in ROS1/bridge.

## Typical bottlenecks

1. **Image+laser synchronizer:** if one topic is slow, the callback rate drops and the log shows "rest ~270 ms" (waiting for the pair).
2. **YOLO:** usually ~5–15 ms on GPU; rarely dominant.
3. **Scan clustering:** O(n beams); normally cheap compared with sync.
4. **TF:** a cache by `(frame, stamp)` reduces repeated lookups.
5. **Costmap filter:** one query per cluster; adds up with a large costmap and many clusters.

## Readability and suggested refactor (technical debt)

| Area | Status | Suggestion |
|------|--------|------------|
| `dynamic_tracking_node.py` (~2200 lines) | Monolith | Extract modules: `tf_planar.py`, `yolo_map_fusion.py`, `costmap_filter.py`, `pipeline_tick.py` with a `process_frame(ctx)` function. |
| `update_tracks(...)` | Many arguments | A `TrackingParams` `dataclass` or an immutable configuration object. |
| Diagnostics | Counters in the node | A small `LidarFusionDiag` class with `reset_interval()`. |
| Tests | Few | ROS-free unit tests of `assign_measurements`, `assoc_class_key`, `scan_window_best_range`. |

## Common position errors (not Kalman bugs as such)

- **AMCL / pose:** jumps in `map`↔`base` make map measurements "jump"; the tracker only fuses measurements, it does not correct localization.
- **Association threshold:** too strict → `missed_frames`; too loose → ID swaps.
- **Camera without detections:** without a YOLO `map_xy`, only the LiDAR remains; see the `fusion: 0 target detections` logs.
