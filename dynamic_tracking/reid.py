"""Re-ID por apariencia (OSNet) para recuperar el ID de una persona al volver a verla.

OSNet: Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification" (ICCV 2019).
Arquitectura copiada de deep-person-reid / torchreid (MIT License, Kaiyang Zhou), solo la
parte de inferencia (sin InstanceNorm ni clasificador). Pesos: kaiyangzhou/osnet en Hugging Face.

`torch` se importa de forma perezosa: con `enable_reid:=false` este modulo no carga nada.
"""

import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]

# Configuraciones OSNet (channels por etapa); todas con layers [2, 2, 2].
OSNET_CHANNELS = {
    "osnet_x0_25": (16, 64, 96, 128),
    "osnet_x0_5": (32, 128, 192, 256),
    "osnet_x0_75": (48, 192, 288, 384),
    "osnet_x1_0": (64, 256, 384, 512),
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_osnet(variant: str = "osnet_x0_25", feature_dim: int = 512):
    """Devuelve un nn.Module OSNet (modo inferencia: salida = embedding de `feature_dim`)."""
    import torch.nn as nn

    class ConvLayer(nn.Module):
        def __init__(self, cin, cout, k, stride=1, padding=0):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1Linear(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return self.bn(self.conv(x))

    class LightConv3x3(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
            self.bn = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv2(self.conv1(x))))

    class ChannelGate(nn.Module):
        def __init__(self, cin, reduction=16):
            super().__init__()
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(cin, cin // reduction, 1, bias=True)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(cin // reduction, cin, 1, bias=True)
            self.gate_activation = nn.Sigmoid()

        def forward(self, x):
            g = self.global_avgpool(x)
            g = self.gate_activation(self.fc2(self.relu(self.fc1(g))))
            return x * g

    class OSBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            mid = cout // 4
            self.conv1 = Conv1x1(cin, mid)
            self.conv2a = LightConv3x3(mid, mid)
            self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
            self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
            self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
            self.gate = ChannelGate(mid)
            self.conv3 = Conv1x1Linear(mid, cout)
            self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x
            x1 = self.conv1(x)
            x2 = (
                self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
                + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
            )
            x3 = self.conv3(x2)
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.relu(x3 + identity)

    def make_layer(cin, cout, n_blocks, reduce_spatial):
        layers = [OSBlock(cin, cout)] + [OSBlock(cout, cout) for _ in range(1, n_blocks)]
        if reduce_spatial:
            layers.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    class OSNet(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = ConvLayer(3, ch[0], 7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = make_layer(ch[0], ch[1], 2, True)
            self.conv3 = make_layer(ch[1], ch[2], 2, True)
            self.conv4 = make_layer(ch[2], ch[3], 2, False)
            self.conv5 = Conv1x1(ch[3], ch[3])
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(ch[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            x = self.maxpool(self.conv1(x))
            x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
            v = self.global_avgpool(x).flatten(1)
            return self.fc(v)

    if variant not in OSNET_CHANNELS:
        raise ValueError(f"OSNet desconocido '{variant}' (opciones: {sorted(OSNET_CHANNELS)})")
    return OSNet(OSNET_CHANNELS[variant])


def load_osnet_state_dict(path: str):
    """Carga pesos (solo tensores: weights_only=True) y descarta el clasificador de entrenamiento."""
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    out = OrderedDict()
    for k, v in sd.items():
        k = k[7:] if k.startswith("module.") else k
        if k.startswith("classifier."):
            continue
        out[k] = v
    return out


class ReIdEmbedder:
    """Recortes de persona (BGR) -> embeddings L2-normalizados (N, D) en la GPU del YOLO."""

    def __init__(
        self,
        weights_path: str,
        variant: str = "osnet_x0_25",
        device: str = "cuda:0",
        half: bool = True,
        input_hw: Tuple[int, int] = (256, 128),
        min_crop_px: int = 24,
    ) -> None:
        import torch

        self._torch = torch
        self.device = torch.device(device if (not str(device).startswith("cuda") or torch.cuda.is_available()) else "cpu")
        self.half = bool(half) and self.device.type == "cuda"
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        self.min_crop_px = int(min_crop_px)
        model = build_osnet(variant)
        model.load_state_dict(load_osnet_state_dict(weights_path), strict=True)
        model.eval().to(self.device)
        if self.half:
            model.half()
        self._model = model
        self.feature_dim = int(model.fc[0].out_features)

    def warmup(self) -> None:
        h, w = self.input_hw
        self.embed(np.zeros((h * 2, w * 2, 3), dtype=np.uint8), [(0.0, 0.0, float(w), float(h * 2))])

    def _preprocess(self, image_bgr: np.ndarray, boxes: Sequence[BBox]) -> Tuple[np.ndarray, List[int]]:
        import cv2

        ih, iw = image_bgr.shape[:2]
        h, w = self.input_hw
        crops: List[np.ndarray] = []
        keep: List[int] = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            xa, ya = max(0, int(round(x1))), max(0, int(round(y1)))
            xb, yb = min(iw, int(round(x2))), min(ih, int(round(y2)))
            if xb - xa < self.min_crop_px // 2 or yb - ya < self.min_crop_px:
                continue
            crop = cv2.resize(image_bgr[ya:yb, xa:xb], (w, h), interpolation=cv2.INTER_LINEAR)
            crops.append(crop[:, :, ::-1])  # BGR -> RGB
            keep.append(i)
        if not crops:
            return np.zeros((0, 3, h, w), dtype=np.float32), keep
        batch = np.stack(crops).astype(np.float32) / 255.0
        batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2)), keep

    def embed(self, image_bgr: np.ndarray, boxes: Sequence[BBox]) -> Dict[int, np.ndarray]:
        """Devuelve {indice de caja: embedding}; omite cajas demasiado pequenas."""
        if not boxes:
            return {}
        batch, keep = self._preprocess(image_bgr, boxes)
        if not keep:
            return {}
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device)
            if self.half:
                x = x.half()
            feats = self._model(x).float()
            feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy()
        return {bi: feats[k] for k, bi in enumerate(keep)}


class EmbeddingCache:
    """Embeddings por bbox exacta (tupla de floats del detector), compartido entre hilos.

    El worker YOLO calcula embeddings y los guarda aqui; el pipeline de medidas los busca
    con la misma tupla bbox que viaja en RawYoloDet (sin cambiar el formato de las cajas).
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._lock = threading.Lock()
        self._data: "OrderedDict[BBox, np.ndarray]" = OrderedDict()
        self._max = int(max_entries)

    def put(self, bbox: BBox, emb: np.ndarray) -> None:
        key = tuple(float(v) for v in bbox)
        with self._lock:
            self._data[key] = emb
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def get(self, bbox: BBox) -> Optional[np.ndarray]:
        key = tuple(float(v) for v in bbox)
        with self._lock:
            return self._data.get(key)
