#!/usr/bin/env python3
"""Record a LiDAR session from the live robot for detector training (read-only).

Subscribes to /scan + TF (map <- laser); never touches the tracking node.
Saves raw scans + robot pose per scan (features can be recomputed later) and the
per-segment Spencer features, and registers the session in data/sessions.yaml.

Labels (how train.py turns segments into person / not-person):
  zone        people walk inside the labelling zone -> segments inside = person,
              everything else is ignored. Check first with an "empty" session
              that nothing else is inside the zone. Works with the robot moving.
  empty       nobody in LiDAR range -> every segment = not-person.
  zone_empty  nobody inside the zone -> segments inside = not-person, outside ignored
              (e.g. the operator drives the robot from outside the zone).
  background  people walk anywhere; person = segment clearly in front of the
              background recorded by --background sessions. Robot must NOT move.

Examples:
  python3 record_session.py --name v3/s1_walk_moving --label zone  --group moving --seconds 120
  python3 record_session.py --name v3/s2_empty_moving --label zone_empty --group moving --seconds 60
"""

import argparse
import math
import os
import pickle
import time

import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

from common import (
    build_pipeline, extract_segments, feature_names, in_zone, load_manifest, save_manifest,
    session_path,
)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--name", required=True, help="<dataset>/<session>, e.g. v3/s1_walk_moving")
ap.add_argument("--label", required=True, choices=["zone", "empty", "zone_empty", "background"])
ap.add_argument("--group", required=True,
                help="robot pose/condition; train.py tests on each group with the others as training")
ap.add_argument("--seconds", type=float, default=120.0)
ap.add_argument("--background", nargs="*", default=[],
                help="for --label background: empty sessions recorded from the same robot pose")
ap.add_argument("--notes", default="", help="free text stored in the manifest")
args = ap.parse_args()

out = session_path(args.name)
if os.path.exists(out):
    raise SystemExit(f"{out} already exists; choose another --name")
os.makedirs(os.path.dirname(out), exist_ok=True)

pipe = build_pipeline()
names = feature_names(pipe)

rclpy.init()
node = Node("record_session")
buf = tf2_ros.Buffer()
tf2_ros.TransformListener(buf, node)
scans, rows, feats = [], [], []
t0 = [None]
zone = load_manifest()["zone"]

# Live localization check: share of segments lying on /map obstacles (+-0.3 m) and
# segments inside the labelling zone, printed every 5 s. Objects missing from the map
# lower the first number; segments inside an empty zone mean bad localization.
grid = {}


def map_cb(msg: OccupancyGrid):
    from scipy.ndimage import binary_dilation
    g = np.asarray(msg.data, np.int16).reshape(msg.info.height, msg.info.width)
    r = msg.info.resolution
    grid.update(occ=binary_dilation(g > 50, iterations=max(1, int(0.3 / r))), res=r,
                ox=msg.info.origin.position.x, oy=msg.info.origin.position.y,
                raw=dict(occupancy=g.astype(np.int8), resolution=r,
                         origin_x=msg.info.origin.position.x, origin_y=msg.info.origin.position.y))


def on_map_obstacle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    occ = grid["occ"]
    i = ((y - grid["oy"]) / grid["res"]).astype(int)
    j = ((x - grid["ox"]) / grid["res"]).astype(int)
    ok = (i >= 0) & (i < occ.shape[0]) & (j >= 0) & (j < occ.shape[1])
    out = np.zeros(len(x), bool)
    out[ok] = occ[i[ok], j[ok]]
    return out


live = dict(t=0.0, n=0, wall=0, zone=0, scans=0)


def cb(scan: LaserScan):
    now = time.time()
    if t0[0] is None:
        t0[0] = now
    if now - t0[0] > args.seconds:
        return
    try:
        t = buf.lookup_transform("map", scan.header.frame_id, Time(), timeout=Duration(seconds=0.0))
    except Exception:
        return
    q = t.transform.rotation
    pose = (t.transform.translation.x, t.transform.translation.y,
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
    i = len(scans)
    scans.append(dict(t=now - t0[0], pose=pose, ranges=np.asarray(scan.ranges, np.float32)))
    if i == 0:
        scans[0].update(angle_min=scan.angle_min, angle_increment=scan.angle_increment,
                        frame_id=scan.header.frame_id)
    r, f = extract_segments(pipe, names, scan, pose)
    for row in r:
        row.update(scan=i, t=now - t0[0], rx=pose[0], ry=pose[1], ryaw=pose[2])
    rows.extend(r)
    feats.extend(f)
    if r and grid:
        x = np.array([row["x"] for row in r])
        y = np.array([row["y"] for row in r])
        live["n"] += len(r)
        live["wall"] += int(on_map_obstacle(x, y).sum())
        live["zone"] += int(in_zone(x, y, zone).sum())
        live["scans"] += 1
    if now - t0[0] - live["t"] >= 5.0 and live["n"]:
        live["t"] = now - t0[0]
        print(f"  t={live['t']:5.1f}s  on map obstacles {100 * live['wall'] / live['n']:3.0f}%  "
              f"| inside zone {live['zone'] / max(live['scans'], 1):4.1f} seg/scan", flush=True)
        live.update(n=0, wall=0, zone=0, scans=0)


node.create_subscription(LaserScan, "/scan", cb, qos_profile_sensor_data)
node.create_subscription(OccupancyGrid, "/map", map_cb, QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
print(f"[{args.name}] recording {args.seconds:.0f} s ...", flush=True)
while t0[0] is None or time.time() - t0[0] <= args.seconds:
    rclpy.spin_once(node, timeout_sec=0.5)
if not scans:
    raise SystemExit("no scans recorded (is /scan publishing and TF map->laser available?)")

meta = dict(angle_min=scans[0]["angle_min"], angle_increment=scans[0]["angle_increment"],
            frame_id=scans[0]["frame_id"])
# The /map snapshot lets train.py check localization per scan (see --min-map-agreement).
pickle.dump(dict(name=args.name, feature_names=names, meta=meta, scans=scans, rows=rows,
                 X=np.vstack(feats), map=grid.get("raw")), open(out, "wb"))

poses = np.array([s["pose"] for s in scans])
moved = float(np.sum(np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))))
turned = float(np.degrees(np.sum(np.abs(np.diff(np.unwrap(poses[:, 2]))))))
m = load_manifest()
m["sessions"] = [s for s in m["sessions"] if s["name"] != args.name]
entry = dict(name=args.name, label=args.label, group=args.group, scans=len(scans),
             segments=len(rows), robot_moved_m=round(moved, 2), robot_turned_deg=round(turned, 1),
             recorded=time.strftime("%Y-%m-%d %H:%M"))
if args.background:
    entry["background"] = args.background
if args.notes:
    entry["notes"] = args.notes
m["sessions"].append(entry)
save_manifest(m)
print(f"[{args.name}] scans={len(scans)} segments={len(rows)} robot moved {moved:.1f} m, "
      f"turned {turned:.0f} deg -> {out}", flush=True)
