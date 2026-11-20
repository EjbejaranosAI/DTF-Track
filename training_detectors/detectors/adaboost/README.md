# AdaBoost (Spencer features, Gentle AdaBoost, 100 stumps)

## Deployed model

`models/spencer_adaboost/tiago_v3_2026-10-14.adaboost`, threshold `-1.01` (run `v3_2026-10-14`).
Previous: `tiago_height21cm_v2.adaboost`, threshold `-1.65` (dataset v2 only).

Live test (3 min, one person walking over the whole zone with stops; robot moved during the
test). **Not an isolated model comparison:** the v2 test was done before leg-pair fusion and
odometry-based rotation correction were added, which also affect tracking.

| | v2 (no leg fusion, no odometry) | v3 (current configuration) |
|---|---|---|
| IDs for one person | 2 | **1** (the full 180 s) |
| Messages with > 1 ID (ghosts/duplicates) | 0 % | 0 % |
| Messages with no ID | 1 % | **0 %** |
| Rate / latency | 15.0 Hz / 97 ms | 14.9 Hz / 97 ms |

## Run `v3_2026-10-14` (datasets v2 + v3, 6 groups)

Final model: `runs/v3_2026-10-14/tiago_v3_2026-10-14.adaboost`, trained on 13,585 leg
segments and 126,084 background segments. Suggested thresholds (training set):
FP 5 % → `-1.01`, FP 2 % → `-0.27`, FP 1 % → `0.44`.

Leave-one-group-out evaluation: AUC with each group as the test set, training on the rest.

| Test group | Leg heuristic | Old AdaBoost (Spencer) | Deployed AdaBoost (v2) | **New AdaBoost** |
|---|---|---|---|---|
| moving (robot moving) | 0.729 | 0.713 | 0.996 | **0.995** |
| pos1 | 0.814 | 0.514 | (in its training set) | **0.962** |
| pos2 | 0.795 | 0.671 | (in its training set) | **0.996** |
| pos3 | 0.683 | 0.650 | 0.999 | **0.997** |
| pos4 | 0.673 | 0.581 | 1.000 | **0.999** |
| pos5 (other side of the zone) | 0.707 | 0.553 | 0.921 | **0.914** |

- On the groups v2 never saw (moving, pos3–pos5), **v2 and the new model perform the same**:
  more data did not improve these numbers, which were already close to the maximum.
- **pos5 is the hard case for every method** (AUC ~0.91, ~50 % of legs at FP 5 %): looking at
  the scene from the other side of the zone changes the background a lot. The final model
  includes pos5, so it should do better there than v2, but there is no held-out data to measure it.
- More detail (TP@FP and ROC curves): `../comparison_v3_2026-10-14.md` and `runs/v3_2026-10-14/`.
