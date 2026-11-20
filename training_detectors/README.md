# Retraining the LiDAR leg detectors

The tracker finds legs in the laser scan with an **AdaBoost** classifier (and optionally a **Random Forest**) on SPENCER's 50 geometric features. These models depend on the **laser height and scanner**, so a robot other than TIAGo Pro should retrain them. The tools here record data from your robot, label it automatically, train both models, and compare them with the shipped ones.

```
training_detectors/
├── scripts/
│   ├── record_session.py   record one session from the live robot
│   ├── train.py            train + evaluate AdaBoost, RF or both
│   └── common.py           feature extraction (same code as the node) and labelling
├── data/
│   ├── sessions.yaml           your sessions (created by the first recording)
│   ├── sessions_tiago_v3.yaml  the TIAGo Pro dataset we used (reference only)
│   └── <dataset>/<session>.pkl recordings (not in git, ~10-20 MB each)
└── detectors/
    ├── adaboost/ random_forest/   results per run: runs/<run>/ (model, metrics.json, ROC curves)
    └── comparison_<run>.md        all methods side by side
```

## How the labelling works (no manual annotation)

You choose a **labelling zone**: a rectangle in the `map` frame. Then you record two kinds of sessions from the same robot pose:

- **Empty**: nobody inside the zone. Everything the laser sees inside the zone, plus walls from `/map`, is labelled *not person*.
- **Walk**: one or more people walk inside the zone the whole time. Everything inside the zone is labelled *person*; the rest is ignored.

That is why the zone must contain **only floor and people**: move chairs out, or draw the zone around them.

## 0. Setup

The robot must be localized (AMCL / Nav2 publishing `/map` and the `map` TF), and `/scan` must be running. The tracker node does not need to run.

```bash
unset PYTHONPATH
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/dtf_ws/install/setup.bash        # dynamic_tracking must be built
pip install matplotlib pyyaml             # if missing
cd ~/dtf_ws/src/DTF-Track/training_detectors/scripts
```

Create `../data/sessions.yaml` with **your** zone (map frame, metres). Otherwise the default is our lab's zone:

```yaml
zone: {x_min: 1.0, x_max: 6.0, y_min: -2.0, y_max: 2.0}
sessions: []
```

## 1. Record

For each robot pose, record one empty session and one walk session, both with the **same `--group`**. Use a new group name for every pose:

```bash
# a) Empty zone (you stand outside it)
python3 record_session.py --name v1/s1_empty_pos1 --label zone_empty --group pos1 --seconds 30

# b) Walking inside the zone (stop, turn, come close to the robot, pass behind it)
python3 record_session.py --name v1/s2_walk_pos1  --label zone       --group pos1 --seconds 60
```

Every 5 s the recorder prints two checks:

- `on map obstacles`: share of laser points that match the map. Below ~40 % means bad localization; fix it with *2D Pose Estimate* in RViz.
- `inside zone`: must be **~0.0 seg/scan** during an empty session. If it is not, something is inside the zone; move it or shrink the zone.

Tips:
- Record **at least 3 poses**, including one looking at the zone from the far side. The model generalises across poses, not from one pose.
- Keep the robot **still** or moving slowly. Fast turns break localization and put walls inside the zone (`train.py` drops those scans, but you lose data).
- Unplug the charging cable: a cable on the floor looks like a leg.

All labels:

| `--label` | Situation | Person | Not person |
|---|---|---|---|
| `zone` | people walk inside the zone | segments inside the zone | — |
| `zone_empty` | nobody inside the zone; operator may be outside | — | segments inside the zone + map walls |
| `empty` | nobody within laser range | — | everything |
| `background` | people anywhere, robot **not moving**; needs `--background <empty sessions of the same pose>` | segments in front of the empty background | the rest |

## 2. Train and evaluate

```bash
python3 train.py                        # AdaBoost and RF
python3 train.py --detector adaboost    # only one: adaboost | rf
python3 train.py --recompute            # recompute features from raw scans (after changing feature code)
```

What it does:

1. Labels all sessions in `data/sessions.yaml`. For `zone` and `zone_empty` it keeps only well-localized scans (`--min-map-agreement`, default 0.5).
2. **Leave-one-group-out evaluation**: each group (pose) is the test set once, training on the others. The new models are compared with the leg heuristic, the original SPENCER models and the shipped TIAGo model.
3. Trains the **final model on all data** and prints suggested thresholds.

How to read the results (`detectors/comparison_<run>.md`): **AUC** (1 = perfect, 0.5 = chance) and **TP@FP x%** = share of legs detected while accepting x % false positives on the background. Our TIAGo Pro results are in `detectors/comparison_v3_2026-10-14.md`.

## 3. Use the new model

```bash
cp ../detectors/adaboost/runs/<run>/tiago_<run>.adaboost ../../models/spencer_adaboost/
```

In `config/dynamic_tracking_params.yaml` (or your own params file):

```yaml
spencer_adaboost_model_path: "models/spencer_adaboost/tiago_<run>.adaboost"
spencer_adaboost_decision_threshold: <fp5 threshold printed by train.py>
```

Rebuild (`colcon build --packages-select dynamic_tracking --symlink-install`) so the model is installed, and relaunch the tracker.

For the RF: copy `runs/<run>/tiago_<run>.random_forest.gz` to `models/spencer_rf/`, set `spencer_rf_model_path` and `spencer_rf_decision_threshold` (fraction of trees, 0–1), and add `"spencer_rf"` to `lidar_detectors_enabled`. On TIAGo Pro the RF was not better than AdaBoost, so we do not enable it (see `detectors/random_forest/README.md`).

## Notes

- Features come from the node's own code and the parameters in `config/dynamic_tracking_params.yaml`, so they match what the tracker sees at run time. If you change the segmentation parameters there, retrain with `--recompute`.
- The `.pkl` files store raw scans and the robot pose per scan, so they can always be re-featurized. They are git-ignored; share them separately (e.g. Git LFS or a download link).
- To leave a session out of training without deleting it, add `excluded: true` to it in `sessions.yaml`.
