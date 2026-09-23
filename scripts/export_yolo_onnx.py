#!/usr/bin/env python3
"""Export YOLO26 to ONNX for dynamic_tracking robot (CPU ONNX Runtime).

Example:
  python3 scripts/export_yolo_onnx.py --model yolo26n.pt --imgsz 256
  # → models/yolo/yolo26n_256.onnx
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX for TIAGO robot")
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics .pt weights")
    parser.add_argument("--imgsz", type=int, default=256, help="Square input size (match yolo_imgsz)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo"),
        help="Output directory",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.model))[0]
    out_name = f"{stem}_{args.imgsz}.onnx"
    out_path = os.path.join(args.out_dir, out_name)

    print(f"Loading {args.model} …")
    model = YOLO(args.model)
    print(f"Exporting ONNX imgsz={args.imgsz} opset={args.opset} …")
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True)
    exported = str(exported)
    if os.path.abspath(exported) != os.path.abspath(out_path):
        shutil.copy2(exported, out_path)
    print(f"Saved: {out_path}")
    print("Robot YAML: yolo_backend:=onnx  yolo_onnx_model:=" + out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
