# Detector comparison — run `v3_2026-10-14`

Leave-one-group-out: each group (robot pose/condition) is the test set once, training on the others. TP@FP x% = legs detected while accepting x% false positives on the background.

## Test on `moving` (343 legs, 6516 background)

| Method | AUC | TP@FP5% | TP@FP2% | TP@FP1% |
|---|---|---|---|---|
| leg_heuristic | 0.729 | 0.08 | 0.03 | 0.00 |
| old adaboost (Spencer) | 0.713 | 0.27 | 0.13 | 0.08 |
| old rf (Spencer) | 0.900 | 0.69 | 0.59 | 0.44 |
| deployed adaboost (v2) | 0.996 | 0.99 | 0.98 | 0.98 |
| NEW adaboost | 0.995 | 0.99 | 0.99 | 0.98 |
| NEW rf | 0.996 | 0.99 | 0.99 | 0.96 |
| NEW adaboost AND rf | 0.996 | 0.99 | 0.99 | 0.98 |

## Test on `pos1` (5133 legs, 75177 background)

| Method | AUC | TP@FP5% | TP@FP2% | TP@FP1% |
|---|---|---|---|---|
| leg_heuristic | 0.814 | 0.47 | 0.41 | 0.33 |
| old adaboost (Spencer) | 0.514 | 0.16 | 0.11 | 0.07 |
| old rf (Spencer) | 0.655 | 0.25 | 0.25 | 0.21 |
| deployed adaboost (v2) | 1.000 | 1.00 | 1.00 | 0.99 |
| NEW adaboost | 0.962 | 0.64 | 0.32 | 0.30 |
| NEW rf | 0.964 | 0.80 | 0.00 | 0.00 |
| NEW adaboost AND rf | 0.963 | 0.70 | 0.35 | 0.22 |

## Test on `pos2` (5802 legs, 23908 background)

| Method | AUC | TP@FP5% | TP@FP2% | TP@FP1% |
|---|---|---|---|---|
| leg_heuristic | 0.795 | 0.34 | 0.03 | 0.03 |
| old adaboost (Spencer) | 0.671 | 0.10 | 0.07 | 0.06 |
| old rf (Spencer) | 0.869 | 0.47 | 0.40 | 0.40 |
| deployed adaboost (v2) | 1.000 | 1.00 | 1.00 | 1.00 |
| NEW adaboost | 0.996 | 0.99 | 0.94 | 0.91 |
| NEW rf | 0.998 | 0.99 | 0.97 | 0.96 |
| NEW adaboost AND rf | 0.999 | 0.99 | 0.98 | 0.98 |

## Test on `pos3` (410 legs, 5371 background)

| Method | AUC | TP@FP5% | TP@FP2% | TP@FP1% |
|---|---|---|---|---|
| leg_heuristic | 0.683 | 0.06 | 0.02 | 0.02 |
| old adaboost (Spencer) | 0.650 | 0.25 | 0.16 | 0.12 |
| old rf (Spencer) | 0.851 | 0.51 | 0.44 | 0.35 |
| deployed adaboost (v2) | 0.999 | 0.99 | 0.99 | 0.99 |
| NEW adaboost | 0.997 | 0.99 | 0.99 | 0.99 |
| NEW rf | 0.999 | 0.99 | 0.99 | 0.99 |
| NEW adaboost AND rf | 0.998 | 0.99 | 0.99 | 0.99 |

## Test on `pos4` (1083 legs, 5157 background)

| Method | AUC | TP@FP5% | TP@FP2% | TP@FP1% |
|---|---|---|---|---|
| leg_heuristic | 0.673 | 0.14 | 0.03 | 0.02 |
| old adaboost (Spencer) | 0.581 | 0.08 | 0.02 | 0.01 |
| old rf (Spencer) | 0.817 | 0.50 | 0.50 | 0.43 |
| deployed adaboost (v2) | 1.000 | 1.00 | 1.00 | 1.00 |
| NEW adaboost | 0.999 | 0.99 | 0.99 | 0.99 |
| NEW rf | 1.000 | 1.00 | 1.00 | 1.00 |
| NEW adaboost AND rf | 1.000 | 1.00 | 0.99 | 0.99 |

## Test on `pos5` (814 legs, 9955 background)

| Method | AUC | TP@FP5% | TP@FP2% | TP@FP1% |
|---|---|---|---|---|
| leg_heuristic | 0.707 | 0.06 | 0.02 | 0.02 |
| old adaboost (Spencer) | 0.553 | 0.05 | 0.02 | 0.01 |
| old rf (Spencer) | 0.690 | 0.30 | 0.24 | 0.24 |
| deployed adaboost (v2) | 0.921 | 0.54 | 0.43 | 0.39 |
| NEW adaboost | 0.914 | 0.51 | 0.33 | 0.32 |
| NEW rf | 0.910 | 0.46 | 0.40 | 0.00 |
| NEW adaboost AND rf | 0.920 | 0.52 | 0.42 | 0.35 |

## Final models (all data)

- **adaboost**: `training_detectors/detectors/adaboost/runs/v3_2026-10-14/tiago_v3_2026-10-14.adaboost` — thresholds (training set): {'fp5': -1.0122, 'fp2': -0.2715, 'fp1': 0.4375}
- **rf**: `training_detectors/detectors/random_forest/runs/v3_2026-10-14/tiago_v3_2026-10-14.random_forest.gz` — thresholds (training set): {'fp5': 0.08, 'fp2': 0.24, 'fp1': 0.46}
