# YOLO ONNX models (robot / CPU)

The TIAGO robot profile uses **ONNX Runtime** instead of PyTorch for YOLO inference.

## Generate weights

From the repo root (with `ultralytics` installed):

```bash
python3 scripts/export_yolo_onnx.py --model yolo26n.pt --imgsz 256
```

This creates `yolo26n_256.onnx` in this folder. After `colcon build`, it is installed to
`share/dynamic_tracking/models/yolo/`.

## Robot parameters

```yaml
yolo_backend: onnx
yolo_onnx_model: "yolo26n_256.onnx"
yolo_imgsz: 256
yolo_onnx_providers: ["CPUExecutionProvider"]
```

`yolo_imgsz` must match the size used at export time.

## Simulation

Gazebo/sim keeps `yolo_backend: ultralytics` and `device: auto` (CUDA when available).
