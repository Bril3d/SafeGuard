"""Runtime latency, FPS, memory, and artifact size benchmarking (PyTorch vs. ONNX)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

from backend.app.inference import ONNXDetector, preprocess_image
from src.data.inspect_dataset import IMAGE_EXTENSIONS


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


def _get_process_rss_mb() -> float:
    import psutil

    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)


def benchmark_onnx_model(
    onnx_path: Path,
    image_paths: list[Path],
    imgsz: int = 640,
    warmup_runs: int = 10,
    conf: float = 0.25,
    iou: float = 0.50,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Benchmark ONNX Runtime load time, preprocess, inference, postprocess, and RSS."""
    if not image_paths:
        raise ValueError("image_paths must contain at least one image")

    selected_providers = providers or ["CPUExecutionProvider"]
    rss_before = _get_process_rss_mb()
    detector = ONNXDetector(
        onnx_path=onnx_path,
        providers=selected_providers,
        default_conf=conf,
        default_iou=iou,
    )
    rss_after_load = _get_process_rss_mb()

    pil_images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as im:
            pil_images.append(im.convert("RGB"))

    for idx in range(warmup_runs):
        detector.predict_image(pil_images[idx % len(pil_images)])

    pre_lat: list[float] = []
    inf_lat: list[float] = []
    post_lat: list[float] = []
    e2e_lat: list[float] = []
    peak_rss = rss_after_load

    for pil_img in pil_images:
        out = detector.predict_image(pil_img, conf_threshold=conf, iou_threshold=iou)
        pre_lat.append(out.timings_ms["preprocess_ms"])
        inf_lat.append(out.timings_ms["inference_ms"])
        post_lat.append(out.timings_ms["postprocess_ms"])
        e2e_lat.append(out.timings_ms["end_to_end_ms"])
        peak_rss = max(peak_rss, _get_process_rss_mb())

    return {
        "runtime": f"onnxruntime ({detector.active_provider})",
        "device": "CPU" if not detector.gpu_available else "GPU",
        "providers": detector.providers,
        "artifact_path": str(onnx_path),
        "artifact_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 3),
        "load_time_ms": detector.load_time_ms,
        "imgsz": imgsz,
        "warmup_runs": warmup_runs,
        "rss_before_mb": rss_before,
        "rss_after_load_mb": rss_after_load,
        "rss_peak_mb": peak_rss,
        "rss_delta_mb": round(peak_rss - rss_before, 2),
        "cuda_peak_allocated_mb": 0.0,
        "cuda_peak_reserved_mb": 0.0,
        "preprocess_latency": summarize_latencies(pre_lat),
        "inference_latency": summarize_latencies(inf_lat),
        "postprocess_latency": summarize_latencies(post_lat),
        "end_to_end_latency": summarize_latencies(e2e_lat),
    }


def benchmark_pytorch_model(
    weights_path: Path,
    image_paths: list[Path],
    device: str = "cpu",
    imgsz: int = 640,
    warmup_runs: int = 10,
    conf: float = 0.25,
    iou: float = 0.50,
) -> dict[str, Any]:
    """Benchmark PyTorch load time, preprocess, inference, postprocess, and memory."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.ops import scale_boxes
    from ultralytics.utils.nms import non_max_suppression

    if not image_paths:
        raise ValueError("image_paths must contain at least one image")

    use_cuda = device != "cpu" and torch.cuda.is_available()
    torch_device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)

    rss_before = _get_process_rss_mb()
    t_load0 = time.perf_counter()
    yolo_wrapper = YOLO(str(weights_path))
    inner_model = yolo_wrapper.model.to(torch_device).eval()
    if use_cuda:
        torch.cuda.synchronize()
    load_ms = round((time.perf_counter() - t_load0) * 1000.0, 3)
    rss_after_load = _get_process_rss_mb()

    pil_images: list[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as im:
            pil_images.append(im.convert("RGB"))

    def _run_one(pil_img: Image.Image) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        nchw, _ = preprocess_image(pil_img, target_size=imgsz)
        tensor = torch.from_numpy(nchw).to(torch_device, non_blocking=False)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.inference_mode():
            raw = inner_model(tensor)
            preds = raw[0] if isinstance(raw, (tuple, list)) else raw
        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        dets = non_max_suppression(preds, conf_thres=conf, iou_thres=iou, max_det=300)
        if dets and dets[0] is not None and len(dets[0]) > 0:
            scale_boxes(
                (imgsz, imgsz),
                dets[0][:, :4],
                (pil_img.height, pil_img.width),
            )
        if use_cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0

    for idx in range(warmup_runs):
        _run_one(pil_images[idx % len(pil_images)])

    pre_lat: list[float] = []
    inf_lat: list[float] = []
    post_lat: list[float] = []
    e2e_lat: list[float] = []
    peak_rss = rss_after_load

    for pil_img in pil_images:
        p_ms, i_ms, post_ms = _run_one(pil_img)
        pre_lat.append(p_ms)
        inf_lat.append(i_ms)
        post_lat.append(post_ms)
        e2e_lat.append(p_ms + i_ms + post_ms)
        peak_rss = max(peak_rss, _get_process_rss_mb())

    cuda_alloc = (
        round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
        if use_cuda
        else 0.0
    )
    cuda_res = (
        round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 2) if use_cuda else 0.0
    )
    hw_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    return {
        "runtime": f"pytorch ({hw_name})",
        "device": "GPU" if use_cuda else "CPU",
        "artifact_path": str(weights_path),
        "artifact_size_mb": round(weights_path.stat().st_size / (1024 * 1024), 3),
        "load_time_ms": load_ms,
        "imgsz": imgsz,
        "warmup_runs": warmup_runs,
        "rss_before_mb": rss_before,
        "rss_after_load_mb": rss_after_load,
        "rss_peak_mb": peak_rss,
        "rss_delta_mb": round(peak_rss - rss_before, 2),
        "cuda_peak_allocated_mb": cuda_alloc,
        "cuda_peak_reserved_mb": cuda_res,
        "preprocess_latency": summarize_latencies(pre_lat),
        "inference_latency": summarize_latencies(inf_lat),
        "postprocess_latency": summarize_latencies(post_lat),
        "end_to_end_latency": summarize_latencies(e2e_lat),
    }


def _run_isolated_benchmark_worker(args_dict: dict[str, Any]) -> dict[str, Any]:
    """Run one runtime benchmark in a clean subprocess for isolated RSS and load timings."""
    cmd = [
        sys.executable,
        "-m",
        "src.models.benchmark",
        "--worker-payload",
        json.dumps(args_dict),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    out = subprocess.check_output(cmd, text=True, env=env)
    return json.loads(out)


def _plot_latency_comparison(results: list[dict[str, Any]], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["runtime"] for r in results]
    pre_means = [r["preprocess_latency"]["mean_ms"] for r in results]
    inf_means = [r["inference_latency"]["mean_ms"] for r in results]
    post_means = [r["postprocess_latency"]["mean_ms"] for r in results]
    e2e_p95 = [r["end_to_end_latency"]["p95_ms"] for r in results]
    fps_vals = [r["end_to_end_latency"]["fps"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels))
    width = 0.45

    ax1.bar(x, pre_means, width, label="Preprocess", color="#64748b")
    ax1.bar(x, inf_means, width, bottom=pre_means, label="Inference", color="#2563eb")
    bottom2 = [p + i for p, i in zip(pre_means, inf_means)]
    ax1.bar(x, post_means, width, bottom=bottom2, label="Postprocess", color="#f59e0b")
    ax1.scatter(x, e2e_p95, color="#dc2626", zorder=5, label="E2E P95")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Stage-Wise Mean Latency & E2E P95")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    bars = ax2.bar(x, fps_vals, width, color="#10b981")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Throughput (FPS)")
    ax2.set_title("End-to-End Throughput (FPS)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, fps_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _write_markdown_report(
    report: dict[str, Any],
    verification: dict[str, Any] | None,
    md_path: Path,
) -> None:
    rows = []
    for r in report["runtimes"]:
        rows.append(
            f"| {r['runtime']} | {r['artifact_size_mb']} | {r['load_time_ms']} | "
            f"{r['preprocess_latency']['mean_ms']} | {r['inference_latency']['mean_ms']} | "
            f"{r['postprocess_latency']['mean_ms']} | {r['end_to_end_latency']['mean_ms']} | "
            f"{r['end_to_end_latency']['p95_ms']} | {r['end_to_end_latency']['fps']} | "
            f"{r['rss_peak_mb']} (Δ+{r['rss_delta_mb']}) | {r['cuda_peak_allocated_mb']} |"
        )
    table = "\n".join(rows)

    ver_block = ""
    if verification:
        rp = verification["raw_tensor_parity"]
        stp = verification.get("same_tensor_prediction_parity", {})
        pp = verification["prediction_parity"]
        st_line = (
            f"- Same-tensor NMS parity (`N={stp.get('images_checked', 0)}`): "
            f"`PyTorch={stp.get('total_pytorch_detections')}`, `ONNX={stp.get('total_onnx_detections')}`, "
            f"`matched={stp.get('matched_detections')}` (`match_rate={stp.get('match_rate_vs_pytorch')}`), "
            f"`mean_iou={stp.get('mean_matched_iou')}`, `max_conf_diff={stp.get('max_conf_diff')}`, "
            f"`max_box_diff_px={stp.get('max_box_coord_diff_px')}`\n"
            if stp
            else ""
        )
        ver_block = (
            "\n## 2. PyTorch vs. ONNX Parity Verification\n\n"
            f"- Images verified (validation-only): `{verification['images_verified']}`\n"
            f"- Input tensor shape: `{verification['tensor_shapes']['input_shape']}`, "
            f"Output tensor shape: `{verification['tensor_shapes']['output_shape']}`\n"
            f"- Class mapping consistent: `{verification['class_mapping_verification']['consistent']}` "
            f"(mismatches: `{verification['class_mapping_verification']['class_mismatches']}`)\n"
            f"- Raw forward tensor parity (`N={rp['images_checked']}`): "
            f"`max_abs_diff={rp['max_abs_diff']}`, `mean_abs_diff={rp['mean_abs_diff']}`, "
            f"`min_cosine_similarity={rp['min_cosine_similarity']}`\n"
            f"{st_line}"
            f"- End-to-end prediction parity (`conf={verification['conf_threshold']}`, `iou={verification['iou_threshold']}`): "
            f"`PyTorch={pp['total_pytorch_detections']}`, `ONNX={pp['total_onnx_detections']}`, "
            f"`matched={pp['matched_detections']}` (`match_rate={pp['match_rate_vs_pytorch']}`), "
            f"`mean_iou={pp['mean_matched_iou']}`, `max_conf_diff={pp['max_conf_diff']}`, "
            f"`max_box_diff_px={pp['max_box_coord_diff_px']}`, `bounds_violations={pp['bounds_violations']}`\n"
            f"- Threshold sweep monotonic: PyTorch `{verification['threshold_behavior']['pytorch_monotonic']}`, "
            f"ONNX `{verification['threshold_behavior']['onnx_monotonic']}`\n"
        )

    by_mode = {r["runtime"]: r for r in report["runtimes"]}
    pt_cpu = by_mode.get("pytorch (CPU)")
    ort_cpu = by_mode.get("onnxruntime (CPUExecutionProvider)")
    pt_gpu = next((r for r in report["runtimes"] if r["device"] == "GPU"), None)
    conclusions = []
    if pt_cpu and ort_cpu:
        inf_speedup = (
            pt_cpu["inference_latency"]["mean_ms"]
            / ort_cpu["inference_latency"]["mean_ms"]
        )
        e2e_speedup = (
            pt_cpu["end_to_end_latency"]["mean_ms"]
            / ort_cpu["end_to_end_latency"]["mean_ms"]
        )
        conclusions.append(
            f"- **CPU vs. CPU**: ONNX Runtime (`CPUExecutionProvider`) is **`{inf_speedup:.2f}x` faster** on inference "
            f"(`{ort_cpu['inference_latency']['mean_ms']} ms` vs. `{pt_cpu['inference_latency']['mean_ms']} ms`) and "
            f"**`{e2e_speedup:.2f}x` faster** end-to-end (`{ort_cpu['end_to_end_latency']['mean_ms']} ms` / "
            f"`{ort_cpu['end_to_end_latency']['fps']} FPS` vs. `{pt_cpu['end_to_end_latency']['mean_ms']} ms` / "
            f"`{pt_cpu['end_to_end_latency']['fps']} FPS`), while reducing total process peak RSS from "
            f"`{pt_cpu['rss_peak_mb']} MB` to `{ort_cpu['rss_peak_mb']} MB`."
        )
    if pt_gpu and ort_cpu:
        conclusions.append(
            f"- **GPU vs. CPU caveat**: On this host, PyTorch CUDA (`{pt_gpu['runtime']}`) remains faster than "
            f"ONNX Runtime `CPUExecutionProvider` (`{pt_gpu['end_to_end_latency']['mean_ms']} ms` / "
            f"`{pt_gpu['end_to_end_latency']['fps']} FPS` vs. `{ort_cpu['end_to_end_latency']['mean_ms']} ms` / "
            f"`{ort_cpu['end_to_end_latency']['fps']} FPS`) because only CPU ONNX Runtime is installed in the "
            f"lightweight runtime environment. FP32 `.onnx` (`{ort_cpu['artifact_size_mb']} MB`) is also roughly "
            f"2x the disk size of `.pt` (`{pt_gpu['artifact_size_mb']} MB`)."
        )
    conclusion_block = (
        "\n## 3. Benchmark Conclusions\n\n" + "\n".join(conclusions) + "\n"
        if conclusions
        else ""
    )

    content = (
        "# SafeGuard Model Optimization & Runtime Benchmark Report\n\n"
        f"**Host Platform**: `{report['hardware']['platform']}` | "
        f"**CPU**: `{report['hardware']['cpu']}` | "
        f"**GPU**: `{report['hardware']['gpu']}`\n"
        f"**Validation Images Benchmarked**: `{report['images_benchmarked']}` "
        f"(warmup: `{report['warmup_runs']}`, `imgsz={report['imgsz']}`)\n\n"
        "## 1. Same-Hardware Runtime Comparison\n\n"
        "| Runtime | Size (MB) | Load (ms) | Pre Mean (ms) | Infer Mean (ms) | Post Mean (ms) | "
        "E2E Mean (ms) | E2E P95 (ms) | E2E FPS | Peak RSS (MB) | CUDA Alloc (MB) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{table}\n"
        f"{ver_block}"
        f"{conclusion_block}"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(content, encoding="utf-8")


def _log_benchmark_to_mlflow(
    tracking_uri: str,
    mlflow_run_id: str | None,
    report: dict[str, Any],
    json_path: Path,
    md_path: Path,
    plot_path: Path,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    ctx = (
        mlflow.start_run(run_id=mlflow_run_id)
        if mlflow_run_id
        else mlflow.start_run(run_name="Runtime-Benchmark")
    )
    with ctx:
        for r in report["runtimes"]:
            slug = (
                "onnx_cpu"
                if r["runtime"].startswith("onnxruntime") and r["device"] == "CPU"
                else ("pt_cuda" if r["device"] == "GPU" else "pt_cpu")
            )
            mlflow.log_metrics(
                {
                    f"bench_{slug}_load_ms": r["load_time_ms"],
                    f"bench_{slug}_pre_ms": r["preprocess_latency"]["mean_ms"],
                    f"bench_{slug}_inf_ms": r["inference_latency"]["mean_ms"],
                    f"bench_{slug}_post_ms": r["postprocess_latency"]["mean_ms"],
                    f"bench_{slug}_e2e_mean_ms": r["end_to_end_latency"]["mean_ms"],
                    f"bench_{slug}_e2e_p95_ms": r["end_to_end_latency"]["p95_ms"],
                    f"bench_{slug}_fps": r["end_to_end_latency"]["fps"],
                    f"bench_{slug}_rss_peak_mb": r["rss_peak_mb"],
                    f"bench_{slug}_rss_delta_mb": r["rss_delta_mb"],
                }
            )
        for p in (json_path, md_path, plot_path):
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="benchmark")


def _collect_benchmark_images(limit: int) -> list[Path]:
    rf_val = Path("data/processed/val_roboflow/images")
    ul_val = Path("data/processed/val_ultralytics/images")
    half = limit // 2
    rf_imgs = sorted(
        p
        for p in rf_val.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )[:half]
    ul_imgs = sorted(
        p
        for p in ul_val.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )[: limit - len(rf_imgs)]
    return rf_imgs + ul_imgs


def main() -> None:
    """CLI entrypoint for benchmarking PyTorch (CPU/GPU) and ONNX Runtime."""
    parser = argparse.ArgumentParser(description="Benchmark SafeGuard PyTorch & ONNX.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/candidates/EXP-E/EXP-E_seed42_best.pt"),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/production/safeguard_yolo11s.onnx"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--verification-report",
        type=Path,
        default=Path("reports/benchmark/verification_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/benchmark/benchmark_report.json"),
    )
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-run-id", type=str, default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument("--worker-payload", type=str, default=None)
    args = parser.parse_args()

    if args.worker_payload:
        cfg = json.loads(args.worker_payload)
        imgs = [Path(p) for p in cfg["image_paths"]]
        mode = cfg["mode"]
        if mode == "onnx_cpu":
            res = benchmark_onnx_model(
                onnx_path=Path(cfg["onnx"]),
                image_paths=imgs,
                imgsz=cfg["imgsz"],
                warmup_runs=cfg["warmup"],
                conf=cfg["conf"],
                iou=cfg["iou"],
                providers=["CPUExecutionProvider"],
            )
        elif mode == "pytorch_cpu":
            res = benchmark_pytorch_model(
                weights_path=Path(cfg["weights"]),
                image_paths=imgs,
                device="cpu",
                imgsz=cfg["imgsz"],
                warmup_runs=cfg["warmup"],
                conf=cfg["conf"],
                iou=cfg["iou"],
            )
        elif mode == "pytorch_cuda":
            res = benchmark_pytorch_model(
                weights_path=Path(cfg["weights"]),
                image_paths=imgs,
                device="0",
                imgsz=cfg["imgsz"],
                warmup_runs=cfg["warmup"],
                conf=cfg["conf"],
                iou=cfg["iou"],
            )
        else:
            raise ValueError(f"Unknown worker mode: {mode}")
        print(json.dumps(res))
        return

    import platform
    import onnxruntime as ort
    import torch

    imgs = _collect_benchmark_images(args.limit)
    base_payload = {
        "weights": str(args.weights),
        "onnx": str(args.onnx),
        "imgsz": args.imgsz,
        "warmup": args.warmup,
        "conf": args.conf,
        "iou": args.iou,
        "image_paths": [str(p) for p in imgs],
    }

    modes = ["pytorch_cpu", "onnx_cpu"]
    if torch.cuda.is_available():
        modes.append("pytorch_cuda")

    runtimes = [
        _run_isolated_benchmark_worker({**base_payload, "mode": mode}) for mode in modes
    ]

    report = {
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor() or "x86_64",
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
            ),
            "available_ort_providers": ort.get_available_providers(),
        },
        "images_benchmarked": len(imgs),
        "warmup_runs": args.warmup,
        "imgsz": args.imgsz,
        "runtimes": runtimes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    verification_data = (
        json.loads(args.verification_report.read_text(encoding="utf-8"))
        if args.verification_report.exists()
        else None
    )
    plot_path = args.output.parent / "latency_comparison.png"
    md_path = args.output.parent / "benchmark_comparison.md"
    _plot_latency_comparison(runtimes, plot_path)
    _write_markdown_report(report, verification_data, md_path)

    run_id = args.mlflow_run_id
    if run_id is None and args.onnx.with_suffix(".metadata.json").exists():
        meta = json.loads(
            args.onnx.with_suffix(".metadata.json").read_text(encoding="utf-8")
        )
        run_id = meta.get("mlflow_run_id")

    if not args.skip_mlflow:
        _log_benchmark_to_mlflow(
            tracking_uri=args.tracking_uri,
            mlflow_run_id=run_id,
            report=report,
            json_path=args.output,
            md_path=md_path,
            plot_path=plot_path,
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
