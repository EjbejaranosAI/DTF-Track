# AdaBoost models (Spencer features)

Compatible with Spencer `srl_laser_detectors` (same features as the Random Forest).

| File | What it is |
|------|------------|
| `tiago_v3_2026-10-14.adaboost` | **Retrained on TIAGo Pro** (laser at 21 cm); default of the `robot` profile. How to retrain: `training_detectors/README.md` |
| `tiago_height21cm_v2.adaboost` | Previous TIAGo model (dataset v2 only) |
| `lms500_0.25deg_height70cm_rathausgasse.adaboost` | Original Spencer model (laser at ~70 cm); used by the `sim` profile |

## Where the original Spencer models come from (IRI GitLab)

| Branch | URL | Contents |
|--------|-----|----------|
| **master** (current) | [models on master](https://gitlab.iri.upc.edu/labrobotica/ros/perception/people_tracking/spencer_people_tracking/-/tree/master/detection/laser_detectors/srl_laser_detectors/models) | Only `*.random_forest.gz` (noetic) |
| **icra16** (historical) | [models on icra16](https://gitlab.iri.upc.edu/labrobotica/ros/perception/people_tracking/spencer_people_tracking/-/tree/icra16/detection/laser_detectors/srl_laser_detectors/models) | `*.adaboost`, `*.svm`, `*.random_forest` **without .gz** |

`lms500_0.25deg_height70cm_rathausgasse.adaboost` comes from **icra16**; the optional
`lms500_0.25deg_height70cm_rathausgasse.adaboost.txt` only holds its training metrics.

## Format

- **master / noetic:** `*_noetic.random_forest.gz` → `cv2.ml.RTrees_load`
- **icra16 and TIAGo models:** `*.adaboost` → OpenCV **FileStorage**; the node loads it automatically

Do not use the `boostFile.txt` of `iri_laser_people_detection` (a different algorithm, not Spencer features).

## Enable in the YAML

```yaml
lidar_detectors_enabled: ["spencer_adaboost", "leg_heuristic"]
spencer_adaboost_model_path: "models/spencer_adaboost/tiago_v3_2026-10-14.adaboost"
spencer_adaboost_decision_threshold: -1.01
lidar_fusion_policy: "require_two"   # or "max_score"
```

After `colcon build`, the log must show `spencer_adaboost` in `LiDAR detectors: [...]`.
