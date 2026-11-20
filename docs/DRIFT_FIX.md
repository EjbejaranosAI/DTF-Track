# Bounding-box drift while the robot turns

Reference document: symptom, root cause, failed attempts and the applied fix (2026).

---

## Symptom

With the person **standing still on the map** and the robot **turning in place**, the 3D boxes and the LiDAR clusters projected into `map` **moved** by several decimetres or metres. In RViz the blue box was misaligned with the green laser points.

The `[POSE]` logs showed:
- the track's `robot=(x,y)` **changing** in the robot frame although the person did not move
- impossible velocities (`v > 2 m/s`) for a static person
- lost tracks and respawns with new IDs

**The Kalman filter was not the primary cause**: it was following LiDAR measurements that were already badly projected.

---

## Root cause

**LiDAR → `map`** projection with a temporal mismatch between:

1. the **`header.stamp` of `/scan`** (when the measurement was taken)
2. the **TF used to transform** the laser points into `map`

### Mechanism

```
scan captured at T  +  TF from T+Δt  →  wrong point on the map
```

If the robot **turns** between `T` and `T+Δt`, the map error is proportional to `Δt × angular velocity × distance to the object`.

### Why it happened in our pipeline

| Factor | Effect |
|--------|--------|
| `sync_image_with_lidar: true` | The laser was only processed when a paired image arrived (`ApproximateTimeSynchronizer` queue + YOLO ~100 ms) → **200–400 ms lag** |
| `scan_tf=latest` (fallback) | TF of the current instant, not of the scan stamp |
| Single-threaded executor | YOLO blocked the laser callback → more lag |
| QoS queue `depth: 10` | Up to ~1 s of old scans queued (with `stamped` it does not break the projection, but it lowers the update rate) |

With **low lag** (~2 ms) and a still person, the relative `robot=(x,y)` stayed stable. With **high lag + turning**, clusters and tracks drifted.

---

## Attempts that did not fix it

| Attempt | Why it was not enough |
|---------|------------------|
| Tuning the Kalman (gates, maximum jumps) during rotation | Patches the symptom; the measurements are still wrong |
| Rejecting scans with `lag > 50 ms` (option A) | Avoids bad updates but leaves the track "blind" to the LiDAR during high lag |
| Forcing `TF@stamp` only, with long timeouts | Slow pipeline (~0.75 Hz), flicker |
| TF cache with `Time()` on scan arrival | Helps when lag is low; not enough with chronic lag |

---

## Applied fix

### 1. Immediate LiDAR (decoupled from the image sync)

- Direct subscription to `/scan` → `_scan_cb` → `_apply_scan` as soon as the message arrives.
- The image+laser sync **only** pairs the camera for YOLO; it no longer blocks cluster extraction.

### 2. TF captured in `_apply_scan`

- Try `TF@scan.header.stamp` first (`lookup_planar_stamped_only`).
- Fall back to `latest` only if the stamped TF is not in the buffer.
- `_transform_xy` uses the TF captured for that scan (not a later lookup).

### 3. `MultiThreadedExecutor` + callback group for `/scan`

- YOLO and image processing do not block the laser callback.

### 4. `lidar_qos_depth: 1` (direct subscription)

- Only the latest scan; avoids processing ~1 s queues.
- The image sync keeps its own queue (`max(lidar_qos_depth, sensor_sync_queue_size)`).

### 5. Larger TF buffer

- `Buffer(cache_time=60 s)` to allow lookups at past stamps (simulation with `use_sim_time`).

---

## Checking with the `[POSE]` logs

Enable with `debug:=true` in the launch (see README).

Useful fields:

| Field | Meaning |
|-------|------------|
| `scan_tf=stamped` | Projection consistent with the scan instant |
| `scan_tf=latest` | Fallback; risk of drift with lag and rotation |
| `lag` | Delay between `now` and `scan.header.stamp` |
| `id=N map=(x,y) robot=(x,y)` | Track on the map and in the robot frame |
| `clusters LiDAR` | Cluster positions after projection |

**Success criterion** (person still, robot turning): the track's `map=(x,y)` and the clusters are **stable** (~15–25 cm variation); `robot=` and `bear` change (normal while turning); `dist` is ~constant.

Stable example:

```
id=2 map=(-1.94,+0.71) robot=(-2.44,+0.57) dist=2.51m  yaw robot -74°
id=2 map=(-1.94,+0.71) robot=(-1.94,+1.97) dist=2.61m  yaw robot -0°
```

The **map** position stays put; in the **robot** frame it rotates with the vehicle.

---

## Relevant parameters (YAML)

| Parameter | Role |
|-----------|-----|
| `sync_image_with_lidar` | `false` on the robot (async YOLO); `true` in sim if Gazebo pairs well |
| `sensor_sync_queue_size` | Image sync queue (4 in sim/robot) |
| `lidar_qos_depth` | `1` on the direct `/scan` subscription |
| `lidar_fusion_use_scan_stamp_for_tf` | The Kalman uses the scan stamp for `fusion_tf_stamp` |
| `debug_pose_log_enable` | `[POSE]` logs (diagnostics only) |

Pipeline detail: [`PIPELINE.md`](PIPELINE.md). Tracking: [`TRACKING.md`](TRACKING.md).
