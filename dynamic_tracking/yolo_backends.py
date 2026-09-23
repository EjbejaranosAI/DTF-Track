"""YOLO inference backends: Ultralytics (GPU/sim) and ONNX Runtime (CPU/robot)."""

from __future__ import annotations

import ast
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# (x1, y1, x2, y2, class_name, confidence)
RawYoloDet = Tuple[float, float, float, float, str, float]


def resolve_yolo_onnx_path(model_name: str, pkg_share_dir: str) -> str:
    """Resolve ONNX weights: share/models/yolo/, share/models/, cwd, or absolute path."""
    candidates = []
    if os.path.isabs(model_name):
        candidates.append(model_name)
    else:
        candidates.extend([
            os.path.join(pkg_share_dir, "models", "yolo", model_name),
            os.path.join(pkg_share_dir, "models", model_name),
            os.path.join("models", "yolo", model_name),
            model_name,
        ])
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"ONNX model not found: {model_name!r}. "
        "Run: python3 scripts/export_yolo_onnx.py --imgsz 256"
    )


def _parse_class_names(metadata: dict) -> Dict[int, str]:
    raw = metadata.get("names", "")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    try:
        parsed = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        return {}
    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {i: str(n) for i, n in enumerate(parsed)}
    return {}


def letterbox_image(
    im: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize with aspect ratio + pad (Ultralytics LetterBox, inference-only)."""
    h0, w0 = im.shape[:2]
    nh, nw = int(new_shape[0]), int(new_shape[1])
    r = min(nh / h0, nw / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = (nw - new_unpad[0]) / 2.0
    dh = (nh - new_unpad[1]) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def scale_boxes_to_original(
    boxes: np.ndarray,
    ratio: float,
    pad_xy: Tuple[float, float],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """Map xyxy boxes from letterboxed coords back to original image."""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    pad_x, pad_y = pad_xy
    out = boxes.copy()
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out[:, :4] /= max(ratio, 1e-9)
    h0, w0 = orig_shape
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w0)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h0)
    return out


def parse_e2e_predictions(
    pred: np.ndarray,
    conf: float,
    class_map: Dict[int, str],
    max_det: int,
    is_target,
) -> List[RawYoloDet]:
    """Parse YOLO26 end-to-end output (N, 6) with columns xyxy, score, class."""
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim == 2 and pred.shape[0] == 6 and pred.shape[1] != 6:
        pred = pred.T
    if pred.ndim != 2 or pred.shape[-1] < 6:
        return []

    out: List[RawYoloDet] = []
    for row in pred:
        score = float(row[4])
        if score < conf:
            continue
        cls_id = int(round(float(row[5])))
        cn = class_map.get(cls_id, str(cls_id))
        if not is_target(cn):
            continue
        out.append((
            float(row[0]), float(row[1]), float(row[2]), float(row[3]),
            cn, score,
        ))
        if len(out) >= max_det:
            break
    out.sort(key=lambda d: d[5], reverse=True)
    return out[:max_det]


class YoloInferenceBackend(ABC):
    """Common interface for background YOLO worker."""

    class_map: Dict[int, str]

    @abstractmethod
    def infer(self, cv_image_bgr: np.ndarray) -> List[RawYoloDet]:
        ...

    @abstractmethod
    def warmup(self) -> None:
        ...

    @property
    @abstractmethod
    def backend_label(self) -> str:
        ...


class UltralyticsYoloBackend(YoloInferenceBackend):
    """PyTorch / Ultralytics path (sim with GPU or fallback)."""

    def __init__(
        self,
        model,
        class_map: Dict[int, str],
        conf: float,
        imgsz: int,
        max_det: int,
        device: str,
        is_target,
    ) -> None:
        self._model = model
        self.class_map = class_map
        self._conf = conf
        self._imgsz = imgsz
        self._max_det = max(1, max_det)
        self._device = device
        self._is_target = is_target

    @property
    def backend_label(self) -> str:
        return f"ultralytics/{self._device}"

    def infer(self, cv_image_bgr: np.ndarray) -> List[RawYoloDet]:
        results = self._model(
            cv_image_bgr,
            verbose=False,
            conf=self._conf,
            device=self._device,
            imgsz=self._imgsz,
            max_det=self._max_det,
        )
        boxes: List[RawYoloDet] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                lid = int(box.cls[0])
                cn = self.class_map.get(lid, str(lid))
                if not self._is_target(cn):
                    continue
                boxes.append((
                    float(x1), float(y1), float(x2), float(y2),
                    cn, float(box.conf[0]),
                ))
        return boxes

    def warmup(self) -> None:
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self.infer(dummy)


class OnnxYoloBackend(YoloInferenceBackend):
    """ONNX Runtime path for CPU robot (YOLO26 end-to-end, no NMS)."""

    def __init__(
        self,
        onnx_path: str,
        class_map: Optional[Dict[int, str]],
        conf: float,
        imgsz: int,
        max_det: int,
        providers: Sequence[str],
        intra_op_threads: int,
        is_target,
        inter_op_threads: int = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "yolo_backend:=onnx requiere onnxruntime. En el TIAGO:\n"
                "  source ~/iri-lab/virtualenv/perception_env/bin/activate\n"
                "  pip install onnxruntime\n"
                "  export PYTHONPATH=.../perception_env/lib/python3.10/site-packages:$PYTHONPATH"
            ) from exc

        opts = ort.SessionOptions()
        if intra_op_threads > 0:
            opts.intra_op_num_threads = int(intra_op_threads)
        if inter_op_threads > 0:
            opts.inter_op_num_threads = int(inter_op_threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        avail = set(ort.get_available_providers())
        resolved: List[str] = []
        for p in providers:
            if p in avail:
                resolved.append(p)
        if not resolved:
            resolved = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(onnx_path, sess_options=opts, providers=resolved)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._provider = resolved[0]

        meta = dict(self._session.get_modelmeta().custom_metadata_map)
        parsed_names = _parse_class_names(meta)
        self.class_map = class_map if class_map else parsed_names
        if not self.class_map:
            self.class_map = {0: "person"}

        args_raw = meta.get("args", "")
        self._end2end = True
        if args_raw:
            try:
                args = ast.literal_eval(str(args_raw))
                if isinstance(args, dict):
                    self._end2end = bool(args.get("nms", True))
            except (SyntaxError, ValueError):
                pass
        if meta.get("end2end", "").lower() in ("true", "1"):
            self._end2end = True

        self._conf = conf
        self._imgsz = int(imgsz)
        self._max_det = max(1, max_det)
        self._is_target = is_target
        self._onnx_path = onnx_path

    @property
    def backend_label(self) -> str:
        return f"onnx/{self._provider}"

    def _preprocess(self, cv_image_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int]]:
        orig_shape = cv_image_bgr.shape[:2]
        lb, ratio, pad = letterbox_image(cv_image_bgr, (self._imgsz, self._imgsz))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, ratio, pad, orig_shape

    def infer(self, cv_image_bgr: np.ndarray) -> List[RawYoloDet]:
        tensor, ratio, pad, orig_shape = self._preprocess(cv_image_bgr)
        outputs = self._session.run(self._output_names, {self._input_name: tensor})
        pred = outputs[0]

        if self._end2end or (pred.ndim >= 2 and pred.shape[-1] == 6):
            dets = parse_e2e_predictions(
                pred, self._conf, self.class_map, self._max_det, self._is_target,
            )
        else:
            return self._parse_grid_output(pred, ratio, pad, orig_shape)

        if not dets:
            return []
        arr = np.array([[d[0], d[1], d[2], d[3]] for d in dets], dtype=np.float32)
        arr = scale_boxes_to_original(arr, ratio, pad, orig_shape)
        return [
            (float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]), float(arr[i, 3]), d[4], d[5])
            for i, d in enumerate(dets)
        ]

    def _parse_grid_output(
        self,
        pred: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> List[RawYoloDet]:
        """Fallback for non-e2e ONNX (rare with YOLO26 default export)."""
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[-1]:
            pred = pred.T
        if pred.shape[-1] < 6:
            return []
        nc = pred.shape[-1] - 4
        out: List[RawYoloDet] = []
        for row in pred:
            scores = row[4:4 + nc]
            cls_id = int(np.argmax(scores))
            score = float(scores[cls_id])
            if score < self._conf:
                continue
            cn = self.class_map.get(cls_id, str(cls_id))
            if not self._is_target(cn):
                continue
            xywh = row[:4].copy()
            xyxy = np.array([
                xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2,
                xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2,
            ], dtype=np.float32)
            xyxy = scale_boxes_to_original(xyxy.reshape(1, 4), ratio, pad, orig_shape)[0]
            out.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), cn, score))
            if len(out) >= self._max_det:
                break
        out.sort(key=lambda d: d[5], reverse=True)
        return out[:self._max_det]

    def warmup(self) -> None:
        dummy = np.zeros((max(64, self._imgsz // 2), max(64, self._imgsz // 2), 3), dtype=np.uint8)
        self.infer(dummy)
