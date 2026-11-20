"""Shared code for recording and training the LiDAR leg detectors (AdaBoost, RF).

Uses the same code path as dynamic_tracking_node (build_lidar_pipeline +
spencer_features), so features match what the node feeds the detectors.
Both the recorder and the trainer (--recompute) go through extract_segments(),
so a saved raw scan always reproduces its features.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import yaml
from sensor_msgs.msg import LaserScan

from dynamic_tracking.lidar_detection.detectors.blob import BlobDetectorParams
from dynamic_tracking.lidar_detection.pipeline import build_lidar_pipeline
from dynamic_tracking.scan_clustering import LegHeuristicParams
from dynamic_tracking.spencer_features import feature_values_by_name

TD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # training_detectors/
REPO_DIR = os.path.dirname(TD_DIR)
DATA_DIR = os.path.join(TD_DIR, "data")
MANIFEST = os.path.join(DATA_DIR, "sessions.yaml")
DETECTORS_DIR = os.path.join(TD_DIR, "detectors")
PARAMS_YAML = os.path.join(REPO_DIR, "config", "dynamic_tracking_params.yaml")

# Default labelling zone (map frame): during "zone" sessions only people are inside
# it (checked with an "empty" session: 0 segments inside). Overridable in the manifest.
ZONE = dict(x_min=1.0, x_max=8.5, y_min=-3.3, y_max=1.3)

# Old Spencer models, kept for the old-vs-new comparison.
OLD_ADABOOST = "models/spencer_adaboost/lms500_0.25deg_height70cm_rathausgasse.adaboost"
OLD_RF = "models/spencer_rf/lms500_0.25deg_height70cm_fr_rathausgasse_noetic.random_forest.gz"


def load_params() -> dict:
    return yaml.safe_load(open(PARAMS_YAML))["dynamic_tracking_node"]["ros__parameters"]


def build_pipeline(p: Optional[dict] = None):
    """All detectors on, no fusion filtering: every segment gets every score."""
    p = p or load_params()
    return build_lidar_pipeline(
        enabled=["spencer_rf", "spencer_adaboost", "leg_heuristic"],
        fusion_policy="max_score",
        fusion_any_high_threshold=0.7,
        only_positive_rf=False,
        dual_tier_enable=True,
        spawn_fusion_policy="max_score",
        maintenance_fusion_policy="max_score",
        maintenance_min_leg_score=0.0,
        cluster_jump_m=p["lidar_cluster_jump_m"],
        min_track_distance_m=p["min_track_distance_m"],
        max_track_distance_m=p["max_track_distance_m"],
        blind_zone_m=0.0,
        pose_mode="median",
        min_points=p["lidar_min_points_per_segment"],
        max_points=p["lidar_max_points_per_segment"],
        min_segment_width_m=0.0,
        leg_params=LegHeuristicParams(
            True, p["leg_min_span_deg"], p["leg_max_span_deg"],
            p["leg_max_range_spread_ratio"], p["leg_min_chord_m"], p["leg_max_chord_m"],
        ),
        leg_heuristic_score_threshold=p["leg_heuristic_detector_min_score"],
        rf_model_path=OLD_RF,
        rf_threshold=p["spencer_rf_decision_threshold"],
        adaboost_model_path=OLD_ADABOOST,
        adaboost_threshold=p["spencer_adaboost_decision_threshold"],
        blob_params=BlobDetectorParams(),
    )


def feature_names(pipe) -> List[str]:
    return next(d for d in pipe._detectors if d.detector_id == "spencer_adaboost")._feature_names


def make_scan(ranges: np.ndarray, angle_min: float, angle_inc: float, frame_id: str) -> LaserScan:
    m = LaserScan()
    m.header.frame_id = frame_id
    m.angle_min = float(angle_min)
    m.angle_increment = float(angle_inc)
    m.angle_max = float(angle_min + (len(ranges) - 1) * angle_inc)
    m.range_min, m.range_max = 0.05, 25.0
    m.ranges = [float(r) for r in ranges]
    return m


def extract_segments(pipe, names: List[str], scan: LaserScan,
                     pose: Tuple[float, float, float]) -> Tuple[List[dict], List[np.ndarray]]:
    """Per-segment rows (map position, old detector scores) + 50-D feature vectors."""
    tx, ty, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)

    def to_map(x, y, _frame, _stamp):
        return (tx + x * c - y * s, ty + x * s + y * c)

    hyps, ctxs = [], {}
    originals = []
    for d in pipe._detectors:
        f = type(d).detect.__get__(d)
        originals.append((d, f))

        def wrapped(cs, f=f):
            for ctx in cs:
                ctxs[ctx.segment_index] = ctx
            r = f(cs)
            hyps.extend(r)
            return r

        d.detect = wrapped
    try:
        pipe.extract_dual(scan, scan.header.frame_id, to_map)
    finally:
        for d, f in originals:
            d.detect = f

    by = {}
    for h in hyps:
        by.setdefault(h.segment_index, {})[h.detector_id] = h
    rows, feats = [], []
    for seg, dd in sorted(by.items()):
        ctx = ctxs.get(seg)
        if ctx is None or ctx.spencer_segment is None:
            continue
        lut = feature_values_by_name(ctx.spencer_segment)
        feats.append(np.array([lut.get(n, -1.0) for n in names], np.float32))

        def raw(k):
            return float(dd[k].raw_score) if k in dd else float("nan")

        rows.append(dict(seg=seg, x=ctx.map_x, y=ctx.map_y, dist=ctx.distance_m,
                         npts=len(ctx.spencer_segment.ranges),
                         leg=float(dd["leg_heuristic"].score) if "leg_heuristic" in dd else float("nan"),
                         ada_old=raw("spencer_adaboost"), rf=raw("spencer_rf")))
    return rows, feats


def in_zone(x: np.ndarray, y: np.ndarray, zone: Optional[dict] = None) -> np.ndarray:
    z = zone or ZONE
    return (x >= z["x_min"]) & (x <= z["x_max"]) & (y >= z["y_min"]) & (y <= z["y_max"])


def load_manifest() -> dict:
    if not os.path.isfile(MANIFEST):
        return {"zone": dict(ZONE), "sessions": []}
    m = yaml.safe_load(open(MANIFEST)) or {}
    m.setdefault("zone", dict(ZONE))
    m.setdefault("sessions", [])
    return m


def save_manifest(m: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MANIFEST, "w") as f:
        yaml.safe_dump(m, f, sort_keys=False, allow_unicode=True)


def session_path(name: str) -> str:
    """name = '<dataset>/<session>', e.g. 'v3/s1_walk_moving'."""
    return os.path.join(DATA_DIR, name + ".pkl")


def background_ranges(empty_sessions: List[dict]) -> np.ndarray:
    """Per-beam background range: nearest obstacle across empty sessions (same robot pose)."""
    meds = []
    for d in empty_sessions:
        R = np.vstack([s["ranges"] for s in d["scans"]]).astype(float)
        R[~np.isfinite(R)] = np.nan
        meds.append(np.nanmedian(R, axis=0))
    bg = np.nanmin(np.vstack(meds), axis=0)
    return np.where(np.isfinite(bg), bg, np.inf)


def label_by_background(session: dict, bg: np.ndarray, p: Optional[dict] = None,
                        margin_m: float = 0.30, min_frac: float = 0.8) -> np.ndarray:
    """1 = segment whose beams are >= margin_m in front of the background on at least
    min_frac of its beams (a person); 0 = matches the background (walls, furniture,
    pieces of a wall split by an occluding leg). Robot must not move between sessions."""
    from dynamic_tracking.lidar_detection.segment import segment_scan

    p = p or load_params()
    meta = session["meta"]
    by_scan = {}
    for i, row in enumerate(session["rows"]):
        by_scan.setdefault(row["scan"], []).append((i, row["seg"]))
    labels = np.zeros(len(session["rows"]), np.int32)
    for si, items in by_scan.items():
        s = session["scans"][si]
        scan = make_scan(s["ranges"], meta["angle_min"], meta["angle_increment"], meta["frame_id"])
        idx = segment_scan(scan, p["lidar_cluster_jump_m"])
        r = s["ranges"].astype(float)
        for row_i, seg in items:
            beams = np.asarray(idx[seg], np.intp)
            closer = r[beams] < bg[beams] - margin_m
            labels[row_i] = int(closer.mean() >= min_frac)
    return labels


def load_map(session: dict) -> Optional[dict]:
    """/map snapshot stored with the session, else data/map_default.npz."""
    m = session.get("map")
    if m is None:
        path = os.path.join(DATA_DIR, "map_default.npz")
        if not os.path.isfile(path):
            return None
        z = np.load(path)
        m = dict(occupancy=z["occupancy"], resolution=float(z["resolution"]),
                 origin_x=float(z["origin_x"]), origin_y=float(z["origin_y"]))
    return m


def on_map_obstacles(session: dict, map_: dict, margin_m: float = 0.3) -> np.ndarray:
    """Per segment: True if it lies on a /map obstacle (+- margin_m)."""
    from scipy.ndimage import binary_dilation

    r = float(map_["resolution"])
    occ = binary_dilation(np.asarray(map_["occupancy"]) > 50, iterations=max(1, int(margin_m / r)))
    x = np.array([q["x"] for q in session["rows"]])
    y = np.array([q["y"] for q in session["rows"]])
    i = ((y - map_["origin_y"]) / r).astype(int)
    j = ((x - map_["origin_x"]) / r).astype(int)
    ok = (i >= 0) & (i < occ.shape[0]) & (j >= 0) & (j < occ.shape[1])
    on = np.zeros(len(x), bool)
    on[ok] = occ[i[ok], j[ok]]
    return on


def map_agreement_per_scan(session: dict, map_: dict, margin_m: float = 0.3) -> np.ndarray:
    """Per scan: share of segments lying on /map obstacles (+- margin_m).

    A low value means the robot pose in `map` is wrong (walls projected elsewhere),
    so zone labels of that scan cannot be trusted.
    """
    on = on_map_obstacles(session, map_, margin_m)
    sc = np.array([q["scan"] for q in session["rows"]], int)
    n = len(session["scans"])
    hits = np.bincount(sc, weights=on.astype(float), minlength=n)
    tot = np.bincount(sc, minlength=n)
    return np.divide(hits, tot, out=np.zeros(n), where=tot > 0)


def label_session(entry: dict, session: dict, manifest: dict, sessions_cache: dict,
                  min_map_agreement: float, p: Optional[dict] = None) -> np.ndarray:
    """Per-segment labels: 1 person, 0 not person, -1 ignored.

    zone / zone_empty use only scans whose map agreement >= min_map_agreement.
    zone_empty negatives = segments inside the (empty) zone + segments on /map
    obstacles; free space outside the zone is ignored (the operator may be there).
    """
    rows = session["rows"]
    x = np.array([q["x"] for q in rows])
    y = np.array([q["y"] for q in rows])
    sc = np.array([q["scan"] for q in rows], int)
    lab = np.full(len(rows), -1, np.int32)
    kind = entry["label"]
    if kind == "empty":
        lab[:] = 0
    elif kind in ("zone", "zone_empty"):
        inside = in_zone(x, y, manifest["zone"])
        good = np.ones(len(rows), bool)
        map_ = load_map(session)
        if min_map_agreement > 0.0 and map_ is not None:
            good = map_agreement_per_scan(session, map_)[sc] >= min_map_agreement
        if kind == "zone":
            lab[inside & good] = 1
        else:
            lab[inside & good] = 0
            if map_ is not None:
                lab[on_map_obstacles(session, map_) & good] = 0
    elif kind == "background":
        bg = background_ranges([sessions_cache[b] for b in entry["background"]])
        lab = label_by_background(session, bg, p).astype(np.int32)
    else:
        raise ValueError(f"unknown label {kind!r} for {entry['name']}")
    return lab
