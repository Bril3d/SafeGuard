# SafeGuard Model Optimization & Runtime Benchmark Report

**Host Platform**: `Windows-11-10.0.26100-SP0` | **CPU**: `AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD` | **GPU**: `NVIDIA GeForce RTX 5070 Laptop GPU`
**Validation Images Benchmarked**: `50` (warmup: `10`, `imgsz=640`)

## 1. Same-Hardware Runtime Comparison

| Runtime | Size (MB) | Load (ms) | Pre Mean (ms) | Infer Mean (ms) | Post Mean (ms) | E2E Mean (ms) | E2E P95 (ms) | E2E FPS | Peak RSS (MB) | CUDA Alloc (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch (CPU) | 18.271 | 39.576 | 19.552 | 100.013 | 1.549 | 121.115 | 136.356 | 8.26 | 1076.64 (Δ+558.12) | 0.0 |
| onnxruntime (CPUExecutionProvider) | 36.139 | 96.301 | 17.443 | 49.698 | 0.783 | 67.924 | 85.858 | 14.72 | 646.95 (Δ+589.82) | 0.0 |
| pytorch (NVIDIA GeForce RTX 5070 Laptop GPU) | 18.271 | 265.612 | 17.61 | 13.753 | 6.637 | 38.001 | 58.548 | 26.32 | 1739.14 (Δ+1202.33) | 122.94 |

## 2. PyTorch vs. ONNX Parity Verification

- Images verified (validation-only): `30`
- Input tensor shape: `[1, 3, 640, 640]`, Output tensor shape: `[1, 9, 8400]`
- Class mapping consistent: `True` (mismatches: `0`)
- Raw forward tensor parity (`N=10`): `max_abs_diff=0.0026245`, `mean_abs_diff=1.14e-05`, `min_cosine_similarity=1.0`
- Same-tensor NMS parity (`N=10`): `PyTorch=42`, `ONNX=42`, `matched=42` (`match_rate=1.0`), `mean_iou=0.99989`, `max_conf_diff=2e-06`, `max_box_diff_px=0.005`
- End-to-end prediction parity (`conf=0.25`, `iou=0.5`): `PyTorch=171`, `ONNX=173`, `matched=167` (`match_rate=0.9766`), `mean_iou=0.9704`, `max_conf_diff=0.26121`, `max_box_diff_px=39.489`, `bounds_violations=0`
- Threshold sweep monotonic: PyTorch `True`, ONNX `True`

## 3. Benchmark Conclusions

- **CPU vs. CPU**: ONNX Runtime (`CPUExecutionProvider`) is **`2.01x` faster** on inference (`49.698 ms` vs. `100.013 ms`) and **`1.78x` faster** end-to-end (`67.924 ms` / `14.72 FPS` vs. `121.115 ms` / `8.26 FPS`), while reducing total process peak RSS from `1076.64 MB` to `646.95 MB`.
- **GPU vs. CPU caveat**: On this host, PyTorch CUDA (`pytorch (NVIDIA GeForce RTX 5070 Laptop GPU)`) remains faster than ONNX Runtime `CPUExecutionProvider` (`38.001 ms` / `26.32 FPS` vs. `67.924 ms` / `14.72 FPS`) because only CPU ONNX Runtime is installed in the lightweight runtime environment. FP32 `.onnx` (`36.139 MB`) is also roughly 2x the disk size of `.pt` (`18.271 MB`).
