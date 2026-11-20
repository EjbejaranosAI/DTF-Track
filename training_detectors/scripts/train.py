#!/usr/bin/env python3
"""Train and evaluate the LiDAR leg detectors (Spencer features) on the recorded sessions.

Reads every session in data/sessions.yaml (except `excluded: true`), labels the
segments (see record_session.py), and for each detector:

  1. Leave-one-group-out evaluation: each `group` (robot pose/condition) is the
     test set once, training on all the others. Compared on the same segments
     with the leg heuristic, the old Spencer models, the deployed model and,
     with --detector both, AdaBoost AND RF together.
  2. Final model trained on all sessions, saved in Spencer FileStorage format
     (loads unchanged in the node) with suggested thresholds.

Outputs:
  detectors/<adaboost|random_forest>/runs/<run>/   model, metrics.json, ROC figures, summary.md
  detectors/comparison_<run>.md                    all methods side by side

Examples:
  python3 train.py                         # both detectors
  python3 train.py --detector adaboost
  python3 train.py --recompute             # rebuild features from raw scans (after code changes)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (  # noqa: E402
    DETECTORS_DIR, REPO_DIR, build_pipeline, extract_segments, feature_names, label_session,
    load_manifest, load_params, make_scan, session_path,
)

DETECTOR_DIRS = {"adaboost": "adaboost", "rf": "random_forest"}
FP_TARGETS = (0.05, 0.02, 0.01)
DEPLOYED = {
    "adaboost": "models/spencer_adaboost/tiago_v3_2026-10-14.adaboost",  # model shipped for TIAGo Pro
    "rf": None,
}


# ---------------------------------------------------------------------------- data

def load_session(name: str, recompute: bool, pipe=None, names=None) -> dict:
    d = pickle.load(open(session_path(name), "rb"))
    if recompute:
        m = d["meta"]
        rows, feats = [], []
        for i, s in enumerate(d["scans"]):
            scan = make_scan(s["ranges"], m["angle_min"], m["angle_increment"], m["frame_id"])
            r, f = extract_segments(pipe, names, scan, s["pose"])
            for row in r:
                row.update(scan=i, t=s["t"], rx=s["pose"][0], ry=s["pose"][1], ryaw=s["pose"][2])
            rows += r
            feats += f
        d["rows"], d["X"] = rows, np.vstack(feats)
    return d


def build_dataset(args) -> tuple[dict, list]:
    manifest = load_manifest()
    entries = [e for e in manifest["sessions"] if not e.get("excluded")]
    pipe = build_pipeline() if args.recompute else None
    names = feature_names(pipe) if args.recompute else None
    needed = {e["name"] for e in entries} | {b for e in entries for b in e.get("background", [])}
    cache = {n: load_session(n, args.recompute, pipe, names) for n in sorted(needed)}
    p = load_params()
    parts = []
    print(f"{'session':28s} {'label':11s} {'group':8s} {'person':>7s} {'not':>7s} {'ignored':>8s}")
    for e in entries:
        d = cache[e["name"]]
        lab = label_session(e, d, manifest, cache, args.min_map_agreement, p)
        keep = lab >= 0
        col = lambda k: np.array([r[k] for r in d["rows"]], float)[keep]  # noqa: E731
        parts.append(dict(X=d["X"][keep], y=lab[keep], group=np.array([e["group"]] * int(keep.sum())),
                          leg=col("leg"), ada_old=col("ada_old"), rf_old=col("rf")))
        print(f"{e['name']:28s} {e['label']:11s} {e['group']:8s} {int((lab == 1).sum()):7d} "
              f"{int((lab == 0).sum()):7d} {int((lab < 0).sum()):8d}")
    data = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    return data, cache[entries[0]["name"]]["feature_names"]


def subset(data: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in data.items()}


# ---------------------------------------------------------------------------- models

def train_adaboost(X, y):
    m = cv2.ml.Boost_create()
    m.setBoostType(cv2.ml.BOOST_GENTLE)
    m.setWeakCount(100)  # Spencer icra16: 100 stumps
    m.setMaxDepth(1)
    m.setWeightTrimRate(0.95)
    m.setPriors(np.array([1.0, (y == 0).sum() / max((y == 1).sum(), 1)], np.float32))
    m.train(cv2.ml.TrainData_create(X.astype(np.float32), cv2.ml.ROW_SAMPLE, y.astype(np.int32)))
    return m


RF_TREES = 100


def train_rf(X, y):
    m = cv2.ml.RTrees_create()
    m.setMaxDepth(10)
    m.setMinSampleCount(5)
    m.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, RF_TREES, 0))
    m.setPriors(np.array([1.0, (y == 0).sum() / max((y == 1).sum(), 1)], np.float32))
    m.train(cv2.ml.TrainData_create(X.astype(np.float32), cv2.ml.ROW_SAMPLE, y.astype(np.int32)))
    return m


TRAIN = {"adaboost": train_adaboost, "rf": train_rf}


def score(det: str, model, X) -> np.ndarray:
    """AdaBoost: weighted vote sum (node threshold scale). RF: fraction of trees voting person."""
    s = model.predict(X.astype(np.float32), flags=cv2.ml.DTREES_PREDICT_SUM)[1].ravel()
    return s if det == "adaboost" else s / RF_TREES


def load_spencer_model(rel_path: str):
    path = os.path.join(REPO_DIR, rel_path)
    if not os.path.isfile(path):
        return None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    m = cv2.ml.Boost_create() if path.endswith(".adaboost") else cv2.ml.RTrees_create()
    m.read(fs.getNode("classifier"))
    fs.release()
    return m


def save_spencer_format(model, names, path):
    """OpenCV model + Spencer header (detector_type, used_feature_dimensions)."""
    tmp = path + ".tmp.yml"
    model.save(tmp)
    lines = open(tmp).read().split("\n")
    os.remove(tmp)
    root = lines[1] if lines[1].strip() != "---" else lines[2]
    body = "\n".join(ln for ln in lines[1:] if ln.strip() != "---")
    kind = "adaboost" if root.startswith("opencv_ml_boost") else "random_forest"
    with open(path, "w") as f:
        f.write(f"%YAML:1.0\ndetector_type: {kind}\nused_feature_dimensions:\n")
        for n in names:
            f.write(f'   - "{n}"\n')
        f.write(body.replace(root.split(":")[0] + ":", "classifier:", 1))


# ---------------------------------------------------------------------------- metrics

def auc(s, y):
    s = np.nan_to_num(s, nan=-1e9)
    rank = s.argsort().argsort() + 1.0
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tp_at_fp(s, y, fp):
    s = np.nan_to_num(s, nan=-1e9)
    thr = float(np.quantile(s[y == 0], 1.0 - fp))
    return float((s[y == 1] > thr).mean()), thr


def metrics(s, y):
    out = dict(auc=round(auc(s, y), 4))
    for fp in FP_TARGETS:
        out[f"tp_at_fp{fp * 100:g}"] = round(tp_at_fp(s, y, fp)[0], 4)
    return out


def rank01(s):
    return np.nan_to_num(s, nan=-1e9).argsort().argsort() / max(len(s) - 1, 1)


def roc_plot(scores: dict, y, title, path):
    plt.figure(figsize=(5.5, 5))
    for label, s in scores.items():
        s = np.nan_to_num(s, nan=-1e9)
        thr = np.unique(np.quantile(s, np.linspace(0, 1, 300)))
        plt.plot([(s[y == 0] > t).mean() for t in thr], [(s[y == 1] > t).mean() for t in thr],
                 label=f"{label} ({auc(s, y):.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.xlim(0, 0.3)
    plt.xlabel("false positives (background)")
    plt.ylabel("legs detected")
    plt.title(title, fontsize=9)
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=90, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=["adaboost", "rf", "both"], default="both")
    ap.add_argument("--recompute", action="store_true", help="rebuild features from the saved raw scans")
    ap.add_argument("--min-map-agreement", type=float, default=0.5,
                    help="zone labels only from scans with >= this share of segments on /map obstacles")
    ap.add_argument("--run", default=time.strftime("%Y-%m-%d_%H%M"), help="run name (output subfolder)")
    args = ap.parse_args()
    dets = ["adaboost", "rf"] if args.detector == "both" else [args.detector]

    data, names = build_dataset(args)
    groups = [g for g in sorted(set(data["group"]))
              if (data["y"][data["group"] == g] == 1).any() and (data["y"][data["group"] == g] == 0).any()]
    print(f"\ntotal: person={int((data['y'] == 1).sum())} not={int((data['y'] == 0).sum())} | test groups: {groups}")

    deployed = {d: load_spencer_model(p) for d, p in DEPLOYED.items() if p}
    run_dirs = {d: os.path.join(DETECTORS_DIR, DETECTOR_DIRS[d], "runs", args.run) for d in dets}
    for rd in run_dirs.values():
        os.makedirs(rd, exist_ok=True)

    results = {}  # group -> method -> metrics
    for g in groups:
        tr, te = subset(data, data["group"] != g), subset(data, data["group"] == g)
        scores = {"leg_heuristic": te["leg"], "old adaboost (Spencer)": te["ada_old"],
                  "old rf (Spencer)": te["rf_old"]}
        for d, m in deployed.items():
            if m is not None:
                scores[f"deployed {d} (tiago_v3)"] = score(d, m, te["X"])
        for d in dets:
            scores[f"NEW {d}"] = score(d, TRAIN[d](tr["X"], tr["y"]), te["X"])
        if len(dets) == 2:
            scores["NEW adaboost AND rf"] = np.minimum(rank01(scores["NEW adaboost"]),
                                                        rank01(scores["NEW rf"]))
        results[g] = {k: metrics(s, te["y"]) for k, s in scores.items()}
        results[g]["_n"] = dict(person=int((te["y"] == 1).sum()), not_person=int((te["y"] == 0).sum()))
        for d in dets:
            roc_plot(scores, te["y"], f"test on group '{g}' (trained on the rest)",
                     os.path.join(run_dirs[d], f"roc_test_{g}.png"))
        print(f"\n== test {g} (person={results[g]['_n']['person']}, not={results[g]['_n']['not_person']})")
        for k, v in results[g].items():
            if not k.startswith("_"):
                print(f"  {k:28s} AUC={v['auc']:.3f}  " + "  ".join(
                    f"TP@FP{fp * 100:g}%={v[f'tp_at_fp{fp * 100:g}']:.2f}" for fp in FP_TARGETS))

    finals = {}
    for d in dets:
        model = TRAIN[d](data["X"], data["y"])
        ext = "adaboost" if d == "adaboost" else "random_forest"
        path = os.path.join(run_dirs[d], f"tiago_{args.run}.{ext}")
        save_spencer_format(model, names, path)
        if d == "rf":  # ~9 MB as text; OpenCV FileStorage reads .gz directly
            import gzip
            import shutil
            with open(path, "rb") as fi, gzip.open(path + ".gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(path)
            path += ".gz"
        s = score(d, model, data["X"])
        thr = {f"fp{fp * 100:g}": round(tp_at_fp(s, data["y"], fp)[1], 4) for fp in FP_TARGETS}
        finals[d] = dict(model=os.path.relpath(path, REPO_DIR), thresholds_train_set=thr)
        json.dump(dict(run=args.run, detector=d, min_map_agreement=args.min_map_agreement,
                       totals=dict(person=int((data["y"] == 1).sum()), not_person=int((data["y"] == 0).sum())),
                       leave_one_group_out=results, final_model=finals[d]),
                  open(os.path.join(run_dirs[d], "metrics.json"), "w"), indent=2)
        print(f"\nfinal {d}: {path}\n  thresholds (train set): {thr}")

    write_comparison(args.run, results, finals, dets)


def write_comparison(run, results, finals, dets):
    methods = [k for k in next(iter(results.values())) if not k.startswith("_")]
    lines = [f"# Detector comparison — run `{run}`", "",
             "Leave-one-group-out: each group (robot pose/condition) is the test set once, "
             "training on the others. TP@FP x% = legs detected while accepting x% false "
             "positives on the background.", ""]
    for g, res in results.items():
        n = res["_n"]
        lines += [f"## Test on `{g}` ({n['person']} legs, {n['not_person']} background)", "",
                  "| Method | AUC | TP@FP5% | TP@FP2% | TP@FP1% |", "|---|---|---|---|---|"]
        for k in methods:
            v = res[k]
            lines.append(f"| {k} | {v['auc']:.3f} | {v['tp_at_fp5']:.2f} | {v['tp_at_fp2']:.2f} | {v['tp_at_fp1']:.2f} |")
        lines.append("")
    lines += ["## Final models (all data)", ""]
    for d, f in finals.items():
        lines.append(f"- **{d}**: `{f['model']}` — thresholds (training set): {f['thresholds_train_set']}")
    path = os.path.join(DETECTORS_DIR, f"comparison_{run}.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"\ncomparison -> {path}")


if __name__ == "__main__":
    main()
