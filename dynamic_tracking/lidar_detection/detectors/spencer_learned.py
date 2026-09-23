"""Spencer OpenCV learned models (Random Forest + AdaBoost, 54 features)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np

from ...spencer_features import (
    Segment,
    compute_feature_vector,
    compute_feature_vector_for_names,
    read_model_feature_dimension_names,
)
from ..types import LidarHypothesis
from .base import LidarDetector, SegmentContext

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2_ML = hasattr(cv2, "ml")
except ImportError:
    _HAS_CV2_ML = False

ModelKind = Literal["random_forest", "adaboost"]


def _infer_model_kind(path: Path) -> ModelKind:
    name = path.name.lower()
    if "adaboost" in name:
        return "adaboost"
    return "random_forest"


def _filestorage_detector_type(path: Path) -> Optional[str]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    try:
        node = fs.getNode("detector_type")
        if node.empty():
            return None
        return str(node.string())
    finally:
        fs.release()


def _load_from_filestorage(path: Path, kind: ModelKind) -> Optional[object]:
    """Spencer icra16-era models: OpenCV FileStorage XML (no .gz)."""
    expected = "adaboost" if kind == "adaboost" else "random_forest"
    det_type = _filestorage_detector_type(path)
    if det_type and det_type != expected:
        logger.warning(
            "Model %s has detector_type=%s, expected %s",
            path.name,
            det_type,
            expected,
        )

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    # icra16 Spencer YAML uses "classifier"; newer exports may use boost/trees.
    node_names = (
        ("classifier", "boost", "Boost", "opencv_ml_boost")
        if kind == "adaboost"
        else ("classifier", "trees", "RTrees", "opencv_ml_rtrees", "random_forest")
    )
    try:
        for name in node_names:
            node = fs.getNode(name)
            if node.empty():
                continue
            if kind == "adaboost":
                model = cv2.ml.Boost_create()
            else:
                model = cv2.ml.RTrees_create()
            model.read(node)
            if not model.empty():
                logger.info("Loaded Spencer %s from FileStorage node '%s'", kind, name)
                return model
    finally:
        fs.release()
    return None


def _load_opencv_model(path: Path, kind: ModelKind) -> Tuple[Optional[object], int]:
    """Try native .gz load (noetic) then legacy FileStorage (icra16). Returns (model, num_trees)."""
    path_str = str(path)
    if kind == "adaboost" and hasattr(cv2.ml, "Boost_load"):
        try:
            model = cv2.ml.Boost_load(path_str)
            if model is not None and not model.empty():
                return model, 1
        except Exception:
            pass
    if kind == "random_forest" and hasattr(cv2.ml, "RTrees_load"):
        try:
            model = cv2.ml.RTrees_load(path_str)
            if model is not None and not model.empty():
                tc = model.getTermCriteria()
                return model, max(1, tc[1])
        except Exception:
            pass
    legacy = _load_from_filestorage(path, kind)
    if legacy is not None:
        n_trees = 1
        if kind == "random_forest":
            try:
                n_trees = max(1, legacy.getTermCriteria()[1])
            except Exception:
                n_trees = 50
        return legacy, n_trees
    return None, 1


def _normalize_adaboost_score(raw: float, threshold: float) -> Tuple[bool, float]:
    """Map Spencer AdaBoost margin to [0, 1] for fusion with RF/heuristic."""
    margin = float(raw) - float(threshold)
    # Soft step: ~0 at threshold, ~1 when margin >= 5
    score = min(1.0, max(0.0, 0.5 + 0.1 * margin))
    return margin > 0.0, score


class SpencerLearnedDetector(LidarDetector):
    """Shared loader for Spencer .random_forest.gz and .adaboost.gz models."""

    def __init__(
        self,
        model_path: str,
        decision_threshold: float,
        detector_id: str,
        model_kind: Optional[ModelKind] = None,
    ) -> None:
        self._path = model_path
        self._threshold = decision_threshold
        self._detector_id = detector_id
        self._kind: ModelKind = model_kind or _infer_model_kind(Path(model_path))
        self._model: Optional[object] = None
        self._num_trees: int = 1
        self._feature_names: Optional[List[str]] = None

        if not _HAS_CV2_ML:
            logger.warning("cv2.ml not available – %s disabled", detector_id)
            return

        path = Path(model_path)
        if not path.is_file():
            logger.warning("Spencer model not found: %s", model_path)
            return

        try:
            self._model, self._num_trees = _load_opencv_model(path, self._kind)
            if self._model is None:
                logger.warning("Spencer model not loaded: %s", model_path)
                return
            names = read_model_feature_dimension_names(str(path))
            if names:
                self._feature_names = names
                logger.info(
                    "Spencer %s: %d feature dimensions from model file",
                    self._kind,
                    len(names),
                )
            logger.info(
                "Spencer %s loaded (%s): threshold=%.3f",
                self._kind,
                path.name,
                self._threshold,
            )
        except Exception as exc:
            logger.error("Failed to load Spencer %s: %s", self._kind, exc)
            self._model = None

    @property
    def detector_id(self) -> str:
        return self._detector_id

    @property
    def model_kind(self) -> ModelKind:
        return self._kind

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _classify_segment(self, seg: Segment) -> Tuple[bool, float, float]:
        if self._model is None:
            return False, 0.0, 0.0
        if self._feature_names:
            fv = compute_feature_vector_for_names(seg, self._feature_names).reshape(
                1, -1
            )
        else:
            fv = compute_feature_vector(seg).reshape(1, -1)
        if self._kind == "adaboost":
            raw = float(self._model.predict(fv)[1][0][0])
            is_person, score = _normalize_adaboost_score(raw, self._threshold)
            return is_person, score, raw
        _, results = self._model.predict(fv, flags=cv2.ml.DTREES_PREDICT_SUM)
        conf = float(results[0][0]) / self._num_trees
        return conf > self._threshold, conf, conf

    def detect(self, segments: List[SegmentContext]) -> List[LidarHypothesis]:
        if not self.is_ready:
            return []
        out: List[LidarHypothesis] = []
        for ctx in segments:
            seg = ctx.spencer_segment
            if seg is None:
                continue
            is_person, score, raw = self._classify_segment(seg)
            out.append(
                LidarHypothesis(
                    segment_index=ctx.segment_index,
                    detector_id=self._detector_id,
                    score=float(score),
                    is_person=bool(is_person),
                    map_x=ctx.map_x,
                    map_y=ctx.map_y,
                    raw_score=float(raw),
                )
            )
        return out


class SpencerRFDetector(SpencerLearnedDetector):
    def __init__(self, model_path: str, decision_threshold: float = 0.5) -> None:
        super().__init__(
            model_path,
            decision_threshold,
            detector_id="spencer_rf",
            model_kind="random_forest",
        )


class SpencerAdaBoostDetector(SpencerLearnedDetector):
    def __init__(self, model_path: str, decision_threshold: float = -10.0) -> None:
        super().__init__(
            model_path,
            decision_threshold,
            detector_id="spencer_adaboost",
            model_kind="adaboost",
        )
