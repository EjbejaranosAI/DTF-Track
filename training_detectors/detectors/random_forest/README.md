# Random Forest (Spencer features, OpenCV RTrees, 100 trees, depth 10)

## Status in the node

**Not deployed.** The old Spencer RF
(`models/spencer_rf/lms500_0.25deg_height70cm_fr_rathausgasse_noetic.random_forest.gz`)
is not in `lidar_detectors_enabled` for the robot profile: with the TIAGo laser (21 cm
height) it never voted "person".

The RF score is the **fraction of trees that vote person** (0–1). The node's loader reads
the number of trees (`ntrees`) from the file.

## Run `v3_2026-10-14` (datasets v2 + v3, 6 groups)

Final model: `runs/v3_2026-10-14/tiago_v3_2026-10-14.random_forest.gz`. Suggested thresholds
(training set): FP 5 % → `0.08`, FP 2 % → `0.24`, FP 1 % → `0.46`.

Leave-one-group-out evaluation (AUC):

| Test group | Old RF (Spencer) | New AdaBoost | **New RF** | New AdaBoost **and** RF |
|---|---|---|---|---|
| moving | 0.900 | 0.995 | **0.996** | 0.996 |
| pos1 | 0.655 | 0.962 | **0.964** | 0.963 |
| pos2 | 0.869 | 0.996 | **0.998** | 0.999 |
| pos3 | 0.851 | 0.997 | **0.999** | 0.998 |
| pos4 | 0.817 | 0.999 | **1.000** | 1.000 |
| pos5 | 0.690 | 0.914 | **0.910** | 0.920 |

Conclusions:
- **The new RF performs like the new AdaBoost**, not consistently better or worse.
- **Using both (both must agree) gives no clear gain**: some numbers go up and others down,
  depending on group and threshold (see `../comparison_v3_2026-10-14.md`). Both use the same
  50 features, so they fail on the same cases.
- Enabling it next to AdaBoost would cost time (features for two models) with no proven
  gain. **Not recommended for deployment for now.**
- What would help both most is more data from **far poses like pos5**, where both fail.
