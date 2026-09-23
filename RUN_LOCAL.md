# Run on your own laptop

No robot needed — TIAGO Gazebo simulation, LiDAR + YOLO tracking, RViz. Use branch `production`.

## 0. Requirements

- ROS 2 Jazzy (or Humble+)
- A TIAGO Gazebo simulation package already working on your machine (this repo only consumes its topics, it doesn't spawn the robot)
- `dynamic_tracking` in your workspace `src/`, on branch `production`

## 1. Build

```bash
cd ~/ros2_ws                       # your workspace
git -C src/dynamic_tracking checkout production
git -C src/dynamic_tracking pull
colcon build --packages-select dynamic_tracking --symlink-install
source install/setup.bash
```

## 2. Launch

**Terminal A** — Gazebo + TIAGO (your own sim launch, e.g.):

```bash
ros2 launch tiago_gazebo tiago_gazebo.launch.py world:=<your_world> is_public_sim:=True
```

**Terminal B** — dynamic_tracking:

```bash
source install/setup.bash
ros2 launch dynamic_tracking dynamic_tracking.launch.py \
  profile:=sim use_rviz:=true use_sim_time:=true
```

## 3. Check it's working

```bash
ros2 topic list | grep detections
ros2 topic hz /detections/tracked
ros2 topic echo /detections/tracked
```

In RViz: set **Fixed Frame = `map`**. Displays: `/dynamic_tracks_markers` (3D boxes), `/detection_area_markers` (ROI), `/image_annotated` (YOLO overlay).

## Variants

| Want | Add to the launch command |
|---|---|
| No GPU / LiDAR-only | `enable_yolo_detections:=false` |
| Camera overlay window | `show_image_view:=true` |
| Restrict to a zone (sim ROI `x∈[-3,1] y∈[-2,2]`) | `area_filter_enable:=true area_filter_x_min:=-3.0 area_filter_x_max:=1.0 area_filter_y_min:=-2.0 area_filter_y_max:=2.0` |

## Stop

`Ctrl+C` in both terminals.

---

For the real TIAGO robot (SSH, ONNX on CPU) and topic details, see [`README.md`](README.md). For pipeline internals, see [`docs/`](docs/).
