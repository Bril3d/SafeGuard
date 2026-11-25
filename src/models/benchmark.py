"""Runtime latency, FPS, memory, and artifact size benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Compute mean, median, P95, min, max latency (ms) and FPS."""
    if not latencies_ms:
        raise ValueError("latencies_ms must be non-empty")
    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "count": float(len(arr)),
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "fps": round(1000.0 / mean_ms if mean_ms > 0 else 0.0, 2),
    }


def _preprocess_image(image_path: Path, imgsz: int) -> np.ndarray:
    with Image.open(image_path) as img:
        resized = img.convert("RGB").resize((imgsz, imgsz), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def benchmark_onnx_model(
    onnx_path: Path,
    image_paths: list[Path],
    imgsz: int = 640,
    warmup_runs: int = 5,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Benchmark ONNX Runtime model load time, inference latency, and size."""
    if not image_paths:
        raise ValueError("image_paths must contain at least one image")

    selected_providers = providers or ["CPUExecutionProvider"]
    load_t0 = time.perf_counter()
    session = ort.InferenceSession(str(onnx_path), providers=selected_providers)
    load_ms = (time.perf_counter() - load_t0) * 1000.0

    input_name = session.get_inputs()[0].name
    tensors = [_preprocess_image(p, imgsz) for p in image_paths]

    for idx in range(warmup_runs):
        session.run(None, {input_name: tensors[idx % len(tensors)]})

    infer_latencies: list[float] = []
    for tensor in tensors:
        t0 = time.perf_counter()
        session.run(None, {input_name: tensor})
        infer_latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "runtime": "onnxruntime",
        "providers": session.get_providers(),
        "artifact_path": str(onnx_path),
        "artifact_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 3),
        "load_time_ms": round(load_ms, 3),
        "imgsz": imgsz,
        "warmup_runs": warmup_runs,
        "inference_latency": summarize_latencies(infer_latencies),
    }


def main() -> None:
    """CLI entrypoint for benchmarking an ONNX model over an image folder."""
    parser = argparse.ArgumentParser(description="Benchmark SafeGuard ONNX model.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    imgs = sorted(
        p
        for p in args.images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.limit]
    report = benchmark_onnx_model(args.onnx, imgs, imgsz=args.imgsz)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
