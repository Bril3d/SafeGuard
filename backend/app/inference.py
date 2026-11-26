"""ONNX Runtime inference wrapper for SafeGuard (strictly decoupled from training)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

DEFAULT_CANONICAL_NAMES: dict[int, str] = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "gloves",
    4: "no_helmet",
}


@dataclass(frozen=True)
class PreprocessMeta:
    """Letterbox geometry metadata required to unscale boxes back to source size."""

    orig_width: int
    orig_height: int
    target_size: int
    new_width: int
    new_height: int
    scale: float
    pad_left: int
    pad_top: int


@dataclass(frozen=True)
class DetectionResult:
    """Single decoded object detection in original pixel xyxy coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass(frozen=True)
class InferenceOutput:
    """Structured output and stage-level timing breakdown from ONNXDetector."""

    detections: list[DetectionResult]
    image_width: int
    image_height: int
    timings_ms: dict[str, float]


def letterbox_image(
    image: Image.Image,
    target_size: int = 640,
    fill_value: int = 114,
) -> tuple[Image.Image, PreprocessMeta]:
    """Resize and center-pad an RGB image to a square canvas preserving aspect ratio."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    orig_w, orig_h = rgb.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    dw = (target_size - new_w) / 2.0
    dh = (target_size - new_h) / 2.0
    pad_left = int(round(dw - 0.1))
    pad_top = int(round(dh - 0.1))

    canvas = Image.new(
        "RGB", (target_size, target_size), (fill_value, fill_value, fill_value)
    )
    canvas.paste(resized, (pad_left, pad_top))

    meta = PreprocessMeta(
        orig_width=orig_w,
        orig_height=orig_h,
        target_size=target_size,
        new_width=new_w,
        new_height=new_h,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return canvas, meta


def preprocess_image(
    image: Image.Image,
    target_size: int = 640,
) -> tuple[np.ndarray, PreprocessMeta]:
    """Convert PIL image to contiguous FP32 NCHW tensor scaled to [0, 1]."""
    canvas, meta = letterbox_image(image, target_size=target_size)
    arr = np.asarray(canvas, dtype=np.float32) * (1.0 / 255.0)
    chw = np.transpose(arr, (2, 0, 1))
    nchw = np.ascontiguousarray(np.expand_dims(chw, axis=0))
    return nchw, meta


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Perform greedy Non-Maximum Suppression on xyxy boxes sorted by score."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[idx] + areas[rest] - inter
        ious = np.where(union > 0.0, inter / union, 0.0)
        order = rest[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def postprocess_yolo_output(
    raw_output: np.ndarray,
    meta: PreprocessMeta,
    class_names: dict[int, str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    max_det: int = 300,
) -> list[DetectionResult]:
    """Decode Ultralytics YOLO ONNX output into clipped source-space detections."""
    preds = np.squeeze(raw_output, axis=0) if raw_output.ndim == 3 else raw_output
    if preds.ndim != 2:
        raise ValueError(
            f"Expected 2D/3D YOLO output tensor, got shape {raw_output.shape}"
        )

    # Ultralytics Detect ONNX exports (4 + nc, N); transpose if channels precede boxes.
    num_expected = 4 + len(class_names)
    if preds.shape[0] == num_expected or preds.shape[0] < preds.shape[1]:
        preds = preds.T

    if preds.shape[1] <= 4:
        return []

    boxes_cxcywh = preds[:, :4]
    cls_scores = preds[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    confidences = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    mask = confidences >= conf_threshold
    if not np.any(mask):
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    cx = boxes_cxcywh[:, 0]
    cy = boxes_cxcywh[:, 1]
    w = boxes_cxcywh[:, 2]
    h = boxes_cxcywh[:, 3]
    boxes_xyxy = np.stack(
        [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], axis=1
    )

    # Class-aware offset NMS mirrors torchvision.ops.batched_nms used by Ultralytics.
    max_wh = 7680.0
    offsets = class_ids.astype(np.float32)[:, None] * max_wh
    keep = nms_boxes(boxes_xyxy + offsets, confidences, iou_threshold=iou_threshold)
    if keep.size > max_det:
        keep = keep[:max_det]

    results: list[DetectionResult] = []
    for idx in keep.tolist():
        x1, y1, x2, y2 = boxes_xyxy[idx]
        ux1 = float(np.clip((x1 - meta.pad_left) / meta.scale, 0.0, meta.orig_width))
        uy1 = float(np.clip((y1 - meta.pad_top) / meta.scale, 0.0, meta.orig_height))
        ux2 = float(np.clip((x2 - meta.pad_left) / meta.scale, 0.0, meta.orig_width))
        uy2 = float(np.clip((y2 - meta.pad_top) / meta.scale, 0.0, meta.orig_height))
        if ux2 <= ux1 or uy2 <= uy1:
            continue
        cid = int(class_ids[idx])
        cname = class_names.get(cid, f"class_{cid}")
        results.append(
            DetectionResult(
                class_id=cid,
                class_name=cname,
                confidence=round(float(confidences[idx]), 6),
                bbox_xyxy=[round(ux1, 2), round(uy1, 2), round(ux2, 2), round(uy2, 2)],
            )
        )
    return results


def _load_detector_metadata(
    onnx_path: Path,
    metadata_path: Path | None,
) -> dict[str, Any]:
    resolved_meta = metadata_path or onnx_path.with_suffix(".metadata.json")
    if resolved_meta.exists():
        return json.loads(resolved_meta.read_text(encoding="utf-8"))
    return {}


class ONNXDetector:
    """Reusable ONNX Runtime inference session for SafeGuard PPE detection."""

    def __init__(
        self,
        onnx_path: Path | str,
        metadata_path: Path | str | None = None,
        providers: list[str] | None = None,
        default_conf: float = 0.25,
        default_iou: float = 0.50,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX artifact not found: {self.onnx_path}")

        meta_path = Path(metadata_path) if metadata_path else None
        self.metadata = _load_detector_metadata(self.onnx_path, meta_path)

        requested_providers = providers or ["CPUExecutionProvider"]
        t_load0 = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=requested_providers,
        )
        self.load_time_ms = round((time.perf_counter() - t_load0) * 1000.0, 3)

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name
        self.input_shape = list(inputs[0].shape)
        self.output_shape = list(outputs[0].shape)
        self.providers = list(self.session.get_providers())
        self.active_provider = (
            self.providers[0] if self.providers else "CPUExecutionProvider"
        )
        self.gpu_available = any(
            ep in self.active_provider for ep in ("CUDA", "Tensorrt", "Dml")
        )

        raw_names = self.metadata.get("class_names")
        if isinstance(raw_names, dict):
            self.class_names = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, list):
            self.class_names = {idx: str(v) for idx, v in enumerate(raw_names)}
        else:
            self.class_names = dict(DEFAULT_CANONICAL_NAMES)

        self.imgsz = int(self.metadata.get("imgsz", self.input_shape[-1] or 640))
        self.default_conf = float(self.metadata.get("default_conf", default_conf))
        self.default_iou = float(self.metadata.get("default_iou", default_iou))
        self.model_version = str(
            self.metadata.get("model_version", self.onnx_path.stem)
        )

    def run_raw_tensor(self, input_tensor: np.ndarray) -> np.ndarray:
        """Execute ONNX Runtime forward pass on a pre-built NCHW FP32 tensor."""
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return np.asarray(outputs[0])

    def predict_image(
        self,
        image: Image.Image,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> InferenceOutput:
        """Run timed preprocess -> ONNX inference -> NMS postprocess on an image."""
        conf = self.default_conf if conf_threshold is None else conf_threshold
        iou = self.default_iou if iou_threshold is None else iou_threshold

        t0 = time.perf_counter()
        tensor, meta = preprocess_image(image, target_size=self.imgsz)
        t1 = time.perf_counter()
        raw_output = self.run_raw_tensor(tensor)
        t2 = time.perf_counter()
        detections = postprocess_yolo_output(
            raw_output=raw_output,
            meta=meta,
            class_names=self.class_names,
            conf_threshold=conf,
            iou_threshold=iou,
        )
        t3 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0
        return InferenceOutput(
            detections=detections,
            image_width=meta.orig_width,
            image_height=meta.orig_height,
            timings_ms={
                "preprocess_ms": round(pre_ms, 3),
                "inference_ms": round(inf_ms, 3),
                "postprocess_ms": round(post_ms, 3),
                "end_to_end_ms": round(total_ms, 3),
            },
        )
