# LiDAR multi-detector pipeline

A single ROS 2 node (`dynamic_tracking_node`): one `/scan` → segment once → several
classifiers → fuse → map filters → Kalman. Inspired by
[Spencer `srl_laser_detectors`](https://gitlab.iri.upc.edu/labrobotica/ros/perception/people_tracking/spencer_people_tracking),
but with **several detectors in parallel** (classic Spencer used **one** per node).

## Flow (diagram)

![LiDAR: segment_scan, parallel detectors, fusion, filters, Kalman](../imgs/lidar_detectors_flow.png)

Source: [`docs/diagrams/lidar_detectors_flow.mmd`](diagrams/lidar_detectors_flow.mmd)

## Models in the repo

| Folder | Files | Use |
|---------|----------|-----|
| `models/spencer_adaboost/` | `tiago_v3_2026-10-14.adaboost` | **Retrained on TIAGo Pro** (laser at 21 cm), `robot` profile default; see `training_detectors/` |
| `models/spencer_rf/` | `*_noetic.random_forest.gz` | Original Spencer RF (GitLab **master** branch) |
| `models/spencer_adaboost/` | `lms500_0.25deg_height70cm_rathausgasse.adaboost` | Original Spencer AdaBoost (GitLab **icra16** branch, FileStorage YAML) |

The `.adaboost.txt` only holds training metrics (F1≈0.87 on Freiburg data); it is not needed at run time.

## Old Spencer vs our node

| | **old_spencer (ROS)** | **dynamic-tracking** |
|--|----------------------|----------------------|
| Detectors | **1** per launch (`type:=random_forest` **or** `adaboost` **or** `blob`) | **Several** in `lidar_detectors_enabled` |
| ML fusion | Did not mix RF+AdaBoost at the same time | `lidar_fusion_policy: max_score` → `max(rf, ada, heur)` |
| Typical launch | `learned_detector_with_segmentation.launch` → RF by default | RF + AdaBoost + heuristic |

In icra16 they had **both** models on disk, but at run time you picked **one** with the `type` parameter.

## Key YAML parameters (sim profile)

```yaml
lidar_detectors_enabled: ["spencer_rf", "spencer_adaboost", "leg_heuristic"]
lidar_fusion_policy: "max_score"
spencer_rf_model_path: "models/spencer_rf/lms500_0.25deg_height70cm_fr_rathausgasse_noetic.random_forest.gz"
spencer_adaboost_model_path: "models/spencer_adaboost/lms500_0.25deg_height70cm_rathausgasse.adaboost"
spencer_adaboost_decision_threshold: -10.0
spencer_rf_only_positive_classifications: false
```

### Fusion policies

- **max_score** (AdaBoost test): raises the score if AdaBoost or the heuristic sees a leg even when RF is low (Gazebo).
- **require_two** (anti-FP): only segments where **≥2** detectors say `person` (e.g. RF + heuristic, or AdaBoost + heuristic).

## How to test and choose the default

1. `colcon build --packages-select dynamic_tracking --symlink-install`
2. Start in Gazebo; the log shows `LiDAR detectors: ['spencer_rf', 'spencer_adaboost', 'leg_heuristic']`
3. If recall is **better** without more FPs → keep the list as it is.
4. If there are **more FPs** on walls → remove `spencer_adaboost` or use `lidar_fusion_policy: "require_two"`.
5. If it is **worse** than RF+heuristic only → go back to `["spencer_rf", "leg_heuristic"]`.

## Adding another detector

1. Class in `dynamic_tracking/lidar_detection/detectors/`.
2. Register it in `build_lidar_pipeline()` (`pipeline.py`).
3. Add its id to `lidar_detectors_enabled`.
