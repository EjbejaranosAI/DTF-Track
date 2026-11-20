# Tracking (Kalman + association) — `dynamic_tracking`

Reference: **Spencer** `srl_nearest_neighbor_tracker` in `old_spencer/`, launches in `example/iri_spencer_tracker_launch/`.

## 360° design (laser first)

1. **Detection** (`lidar_detection/`): segments → fusion (`require_two`, etc.) → clusters in `map`.
2. **Tracking** (`track_pipeline.update_tracks`): 4D Kalman + Hungarian + verification.
3. **Confirmation** (`apply_yolo_auxiliary_person_confirmation`): YOLO adds metadata only (colour, bbox), **no** `correct()` on persons.

## Per-frame cycle (order)

| Step | What it does |
|------|----------|
| 1 | `predict(dt)` — constant velocity; `dt` from the `/scan` stamp (cap `kalman_predict_dt_cap_sec`) |
| 2 | LiDAR association (Hungarian): phase 1 high-score → 1b loose floor → phase 2 the rest |
| 3 | `correct(x,y)` only with a matched laser cluster |
| 4 | Spawn + `lidar_spawn_confirm_frames` (simplified Logic Initiator) |
| 5 | **Scan verify** (Spencer): no cluster — is there a `/scan` return at the expected (r,θ)? → keeps the track, does **not** move the Kalman |
| 6 | FOV: inside the FOV with no cluster or verify → `missed_frames` (with debounce); outside the FOV → `consecutive_occlusions` |
| 7 | `occluded_update()` if there was no measurement |
| 8 | Pruning by `missed` / occlusions; map dedup; auxiliary YOLO |

## Verification in the tracker (avoid FPs and disappearances)

| Mechanism | Anti-FP | Anti-disappearance |
|-----------|---------|-------------------|
| **`require_two` fusion** | Fewer spurious clusters | One frame may be missing → debounce |
| **Map/costmap filters** | Before the tracker | — |
| **`lidar_spawn_confirm_frames`** | Does not open a track on the first blip | ~2 frame spawn delay |
| **`use_mahalanobis_lidar: true`** | χ² gate; odd jumps are not matched | Re-associates after occlusion (covariance grows) |
| **`scan_verify_*`** | Can keep a FP on a wall (ray hits the wall) | Still person without a cluster that frame |
| **`person_missed_debounce_frames`** | — | 1–2 frames without detection do not delete |
| **`max_occlusions_*`** | — | Person behind the robot (360°) |
| **`person_lidar_only_max_missed_frames`** | — | A track without YOLO survives ~88 cycles |
| **Auxiliary YOLO** | Does not create positions | Confirms class / green |

## Spencer vs us

| Spencer NNT | Us |
|-------------|----------|
| Separate node, `DetectedPerson` observations | Same node: segmentation + tracking |
| `world_frame` typically `odom` | `tracking_frame` typically `map` |
| Occlusion manager **before** the Hungarian | Binary FOV + `consecutive_occlusions` |
| Logic Initiator (N scans, velocity) | `PendingLidarSpawn` + `confirm_frames` |
| EKF + Mahalanobis in association | OpenCV Kalman + optional Mahalanobis |
| `updateOccludedTrack` | `occluded_update()` |
| Low-confidence laser recovery | No (equivalent: phase 1b + verify) |
| `filter_tracks.launch` post-filter | RViz / `require_visual_confirmation` |

## Key parameters (`dynamic_tracking_params.yaml`)

```yaml
use_mahalanobis_lidar: true
person_missed_debounce_frames: 2
scan_verify_range_tolerance_m: 0.65
scan_verify_window_deg: 22.0
tracking_person_lidar_assoc_cost_threshold: 1.0
tracking_person_max_missed_frames: 90
person_lidar_only_max_missed_frames: 88
max_occlusions_before_deletion: 50
lidar_spawn_confirm_frames: 2
tracking_min_confirmed_hits: 2
```

## Symptoms and tuning

| Symptom | Try |
|---------|--------|
| Tracks disappear while the person is visible | Raise `person_missed_debounce_frames` (3), `scan_verify_range_tolerance_m`, `tracking_person_lidar_assoc_cost_threshold` |
| Box moves (robot still) | Cluster jitter + AMCL in `map`; try `tracking_frame: odom` |
| FPs on walls persist | `require_two`, map, raise `lidar_spawn_person_min_leg_score`; lower the `scan_verify` tolerance |
| ID flicker / new ID on re-detection | `person_missed_debounce_frames: 2`, `person_lidar_only_max_missed_frames: 22–28`, slightly looser `scan_verify_*`, `tracking_person_lidar_assoc_cost_threshold` ~1.25–1.35 |
| Two boxes on the same person (e.g. ID 9+12) | `person_track_map_merge_lidar_only: false`, IoU ~0.18 |
| Box jumps when the robot turns | `person_lidar_max_correction_jump_m: 0.55` |
| Box drifts with the robot still | `person_static_zero_velocity_speed_m: 0.08` |

## References in `example/`

- **`iri_spencer_tracker_launch`**: sensor relay + `freiburg_people_tracking` (fused detection + NNT + filters).
- **`smart_track`**: 3D Kalman with `N_meas` / covariance to confirm a track (drones); similar idea to `tracking_min_confirmed_hits` + initiator.
