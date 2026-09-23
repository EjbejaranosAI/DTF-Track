# Spencer AdaBoost models

Compatible with Spencer `srl_laser_detectors` (54 features, same as Random Forest).

## Where to download (GitLab IRI)

| Branch | URL | Contents |
|--------|-----|----------|
| **master** (actual) | [models on master](https://gitlab.iri.upc.edu/labrobotica/ros/perception/people_tracking/spencer_people_tracking/-/tree/master/detection/laser_detectors/srl_laser_detectors/models) | Solo `*.random_forest.gz` (noetic) |
| **icra16** (histórico) | [models on icra16](https://gitlab.iri.upc.edu/labrobotica/ros/perception/people_tracking/spencer_people_tracking/-/tree/icra16/detection/laser_detectors/srl_laser_detectors/models) | `*.adaboost`, `*.svm`, `*.random_forest` **sin .gz** |

Para TIAGO/Gazebo (mismo entorno que vuestro RF `rathausgasse`), descargad de **icra16**:

```text
lms500_0.25deg_height70cm_rathausgasse.adaboost
```

Opcional (metadatos): `lms500_0.25deg_height70cm_rathausgasse.adaboost.txt`

### Descarga rápida (con acceso GitLab)

```bash
cd /tmp
git clone -b icra16 --depth 1 \
  ssh://git@gitlab.iri.upc.edu:2202/labrobotica/ros/perception/people_tracking/spencer_people_tracking.git
cp spencer_people_tracking/detection/laser_detectors/srl_laser_detectors/models/\
lms500_0.25deg_height70cm_rathausgasse.adaboost \
  ~/Desktop/PhD/dynamic-tracking/models/spencer_adaboost/
```

## Formato

- **master / noetic:** `*_noetic.random_forest.gz` → `cv2.ml.RTrees_load`
- **icra16:** `*.adaboost` → OpenCV **FileStorage** (XML); el nodo lo carga automáticamente

No uses `iri_laser_people_detection` `boostFile.txt` (otro algoritmo, no Spencer 54-D).

## Activar en YAML

```yaml
lidar_detectors_enabled: ["spencer_rf", "spencer_adaboost", "leg_heuristic"]
spencer_adaboost_model_path: "models/spencer_adaboost/lms500_0.25deg_height70cm_rathausgasse.adaboost"
spencer_adaboost_decision_threshold: -10.0
lidar_fusion_policy: "max_score"   # o "require_two" para anti-FP
```

Tras `colcon build`, el log debe mostrar `spencer_adaboost` en `LiDAR detectors: [...]`.
