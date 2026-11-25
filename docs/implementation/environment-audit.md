# Windows Development Environment Audit & Compatibility Recommendations

**Audit Date**: 2026-11-25  
**Target Workspace**: `c:\dev\SafeGuard`  
**Policy**: Read-only environment inspection. No packages, tools, or runtimes were installed automatically.

---

## 1. Executive Summary

Your Windows 11 workstation is well-equipped for local SafeGuard development, including **31.12 GB RAM**, **103.92 GB free disk space**, **Python 3.12.10**, **Node.js v24.13.1**, **Git 2.51.2**, **Docker CLI 29.2.0 + Compose v5.0.2**, and an **NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM, Blackwell `sm_120`, Driver 595.79, CUDA 13.2)**.

Two critical constraints were uncovered during inspection:
1. **Blackwell GPU (`sm_120`) Requirement**: The RTX 5070 Laptop GPU reports compute capability `12.0` (`sm_120`). Standard legacy PyTorch CUDA wheels (`cu118`, `cu121`, `cu124`) lack `sm_120` kernels and will fail on this GPU. Training requires PyTorch compiled with **CUDA 12.8 (`cu128`) or newer** (compatible with Driver `595.79` / CUDA `13.2`).
2. **Docker Desktop Daemon Not Running**: Docker CLI and Compose v5.0.2 are installed (`desktop-linux` context), but the Docker Engine daemon is currently stopped and must be started before running container builds or `docker compose up --build`.

---

## 2. Detected Environment Inventory

| Component | Detected Status / Version | Details & Notes |
|---|---|---|
| **Windows OS** | `Microsoft Windows 11 Home` (`10.0.26100`, Build `26100`, 64-bit) | Modern Windows 11 64-bit host. |
| **System RAM** | `31.12 GB` total / `18.69 GB` free | Ample for local training, Docker Compose, and IDE workloads. |
| **Disk Space (`C:`)** | `820.47 GB` used / `103.92 GB` free | Adequate, but monitor space once PyTorch CUDA wheels, datasets, and Docker images are added. |
| **Git** | `git version 2.51.2.windows.1` | Installed and working. |
| **Python** | `Python 3.12.10` (`pip 25.0.1`) | Installed at `C:\Users\n3tgg\AppData\Local\Programs\Python\Python312\python.exe`. `C:\Windows\System32\python.exe` is a 0-byte symbolic link. `py` launcher is **not** installed. |
| **Virtual Environments (`venv`)** | Supported (`python -m venv` available) | Verified `python -m venv --help` works on Python 3.12.10. |
| **`uv`** | **Not installed** | `uv` command not found on `PATH`. |
| **Node.js & npm** | `Node.js v24.13.1` / `npm 11.8.0` | Exceeds Vite 6+ (`20.19+` / `22.12+`) and React 19 (`18+`) minimums. |
| **Docker CLI** | `Docker version 29.2.0, build 0b9d198` | Buildx `v0.31.1-desktop.1`, context `desktop-linux`. |
| **Docker Compose** | `Docker Compose version v5.0.2` | Installed at `C:\Program Files\Docker\cli-plugins\docker-compose.exe`. |
| **Docker Desktop Daemon** | **Not running** | Pipe `//./pipe/dockerDesktopLinuxEngine` unavailable; no `*Docker*` process running. |
| **Primary GPU (Discrete)** | `NVIDIA GeForce RTX 5070 Laptop GPU` | `8151 MiB` (~8 GB) VRAM, Compute Capability `12.0` (`sm_120`), WDDM Driver `32.0.15.9579`. |
| **Integrated GPU** | `AMD Radeon(TM) 890M Graphics` | Driver `32.0.23033.1002` (display iGPU). |
| **NVIDIA Driver & `nvidia-smi`** | Driver `595.79` / Max CUDA `13.2` | `nvidia-smi` succeeds; `0%` utilization, `866 MiB / 8151 MiB` used. |
| **CUDA Toolkit (`nvcc`)** | **Not installed** | Not required if using PyTorch wheels that bundle CUDA runtime DLLs (`cu128`/`cu132`). |

---

## 3. Actual Command Outputs

### 3.1 Windows OS, Memory, Disk, and GPU Adapters
```text
=== WINDOWS ===

Caption        : Microsoft Windows 11 Home
Version        : 10.0.26100
BuildNumber    : 26100
OSArchitecture : 64-bit
TotalRAM_GB    : 31.12
FreeRAM_GB     : 18.69

=== DISK ===

Name UsedGB FreeGB
---- ------ ------
C    820.47 103.92

=== GPU CIM ===

Name          : AMD Radeon(TM) 890M Graphics
DriverVersion : 32.0.23033.1002
Status        : OK

Name          : NVIDIA GeForce RTX 5070 Laptop GPU
DriverVersion : 32.0.15.9579
Status        : OK
```

### 3.2 `nvidia-smi` & GPU Compute Capability
```text
Wed Nov 25 19:04:13 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.79                 Driver Version: 595.79         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 ...  WDDM  |   00000000:64:00.0  On |                  N/A |
| N/A   45C    P8              4W /   70W |     866MiB /   8151MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

name, driver_version, memory.total [MiB], compute_cap
NVIDIA GeForce RTX 5070 Laptop GPU, 595.79, 8151 MiB, 12.0
```

### 3.3 Git, Python, `venv`, `uv`, Node.js, and npm
```text
=== GIT ===
git version 2.51.2.windows.1

=== PYTHON ===
C:\Windows\System32\python.exe
C:\Users\n3tgg\AppData\Local\Programs\Python\Python312\python.exe
py : The term 'py' is not recognized as the name of a cmdlet, function, script file, or operable program.
Python 3.12.10
pip 25.0.1 from C:\Users\n3tgg\AppData\Local\Programs\Python\Python312\Lib\site-packages\pip (python 3.12)
usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear]
            [--upgrade] [--without-pip] [--prompt PROMPT] [--upgrade-deps]
            ENV_DIR [ENV_DIR ...]

=== UV ===
uv : The term 'uv' is not recognized as the name of a cmdlet, function, script file, or operable program.

=== NODE & NPM ===
v24.13.1
11.8.0

=== CUDA TOOLKIT / NVCC ===
nvcc : The term 'nvcc' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

### 3.4 Docker & Docker Compose
```text
=== DOCKER ===
Docker version 29.2.0, build 0b9d198
Docker Compose version v5.0.2
Client:
 Version:    29.2.0
 Context:    desktop-linux
 Plugins:
  buildx: Docker Buildx (Docker Inc.) - v0.31.1-desktop.1
  compose: Docker Compose (Docker Inc.) - v5.0.2

=== DOCKER DAEMON ===
docker : failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

---

## 4. Stack Version Compatibility Investigation (Official Docs)

| Stack Layer | Compatibility Findings from Official Documentation | SafeGuard Target Alignment |
|---|---|---|
| **Python** | Latest PyTorch, Ultralytics, FastAPI (Pydantic v2), ONNX Runtime, DVC, and MLflow all officially support **Python 3.12** (`manylinux` / Windows wheels available; PyTorch requires `>=3.11`, Ultralytics recommends `>=3.11`). | Standardize on **Python 3.12** (`3.12.10` locally, `python:3.12-slim` in Docker). |
| **PyTorch & CUDA Runtime** | Official PyTorch Windows stable supports Python `>=3.11` with CUDA `12.8`, `13.2`, `13.4`. Crucially, **RTX 5070 (Blackwell, `sm_120`) requires CUDA 12.8 (`cu128`) or newer** (`cu124` and older lack `sm_120` kernels). Installed NVIDIA Driver `595.79` supports up to CUDA `13.2`. | Install PyTorch (`torch`, `torchvision`) from the `cu128` (or `cu132`) wheel index **before** installing `ultralytics` so pip does not fall back to CPU-only PyTorch. |
| **Ultralytics YOLO** | Supports Python `>=3.8` (`>=3.11` recommended) and `YOLO11n`, `YOLO11s`, `YOLOv8n`. Official guide recommends installing the CUDA-enabled PyTorch build first, then `ultralytics`. | Install `ultralytics` after CUDA PyTorch in the isolated training environment only. |
| **FastAPI & Pydantic** | Modern FastAPI requires **Pydantic v2** (`pydantic>=2.0`) and fully supports Python 3.12 via `pydantic-core` wheels. | Use FastAPI + Pydantic v2 in `backend/requirements.txt`. |
| **ONNX Runtime** | Supports Python 3.12. CPU package `onnxruntime` (`CPUExecutionProvider`) produces small (~150–300 MB) portable Linux images; `onnxruntime-gpu` (`CUDAExecutionProvider`) requires matching CUDA/cuDNN libraries (`cuDNN 9.x` for CUDA 12/13) and `--gpus all` + NVIDIA Container Toolkit in Docker. **Never install both `onnxruntime` and `onnxruntime-gpu` in the same environment.** | Use **CPU `onnxruntime`** in `backend/requirements.txt` & Docker for cross-machine portability and CPU CI parity; optionally test `onnxruntime-gpu` locally in a separate benchmark venv if GPU inference benchmarks are compared. |
| **React, Vite & Node.js** | React 19 (`^19.0.0`) requires Node.js `>=18` and TypeScript modern JSX transform (`react-jsx`). Vite 6+ requires Node.js `20.19+` or `22.12+`. | Host **Node.js v24.13.1 / npm 11.8.0** satisfies React 19 + Vite + TypeScript; use `node:22-alpine` or `node:24-alpine` in `frontend/Dockerfile` multi-stage build. |
| **DVC & MLflow (Windows + Py 3.12)** | Both support Python 3.12. On Windows, DVC cache symlinks can fail without Developer Mode or `copy` fallback, long paths (>260 chars) can break deep venv/cache paths, and concurrent `dvc repro` writes to file-based SQLite MLflow can trigger locks. | Enable Windows Long Paths (recommended), configure DVC cache fallback (`copy,symlink`), and use a dedicated `mlflow server` container/process over HTTP. |

---

## 5. Missing Prerequisites & Potential Friction Points

1. **Docker Desktop Daemon Stopped (Required before Phase 6 / Compose tests)**: Start Docker Desktop and verify WSL 2 (`desktop-linux`) engine initializes cleanly (`docker info`).
2. **PyTorch CUDA `cu128+` Wheel Selection (Required before Phase 3 Training)**: Do not run plain `pip install torch ultralytics` without `--index-url https://download.pytorch.org/whl/cu128` (or `cu132`), or training will either install CPU-only PyTorch or fail on Blackwell `sm_120`.
3. **Optional Tooling (`uv` and `py` launcher)**: Neither `uv` nor `py` is installed. Standard `python -m venv` works out of the box; `uv` is optional (can be installed later if faster dependency resolution is desired).
4. **Windows Long Paths & DVC Cache**: Enabling Windows `LongPathsEnabled` (and/or Developer Mode for symlinks) is recommended before DVC datasets and large virtualenvs are populated.
5. **Disk Space Budget (`103.92 GB` free on `C:`)**: Sufficient for SafeGuard, provided raw/processed datasets, Docker build caches, and multiple virtual environments are kept tidy (`docker system prune`, single shared training venv).

---

## 6. Recommended SafeGuard Development Setup

### 6.1 Training Environment (Host Windows — Isolated `.venv-train`)
- **Python**: Host `Python 3.12.10` using a dedicated virtual environment (`.venv-train/` at repo root or outside repo, gitignored).
- **Packages**:
  1. Install PyTorch CUDA build (`cu128` or `cu132`) matching RTX 5070 Blackwell (`sm_120`) and Driver `595.79`.
  2. Verify `torch.cuda.is_available() == True`, compute capability `(12, 0)`, and a small CUDA matmul smoke test.
  3. Install `ultralytics`, `onnx`, `onnxruntime`, `dvc` (+ chosen remote extra), `mlflow`, `pytest`, `black`, `flake8`, `imagehash`, `pandas`, `pyyaml`.
- **Training Budget Guidance for 8 GB VRAM**: Run `YOLO11n`, `YOLO11s`, and `YOLOv8n` at `imgsz=640` with mixed precision (AMP FP16) and moderate batch size (e.g., 8–16) to fit comfortably inside `8151 MiB` VRAM without OOM. (Kaggle/Colab remains an optional fallback for teammates without NVIDIA GPUs.)

### 6.2 Backend Environment (Isolated `.venv-backend` + Docker `python:3.12-slim`)
- **Python**: Separate lightweight virtual environment (`.venv-backend/`) using `backend/requirements.txt` — **never** sharing `.venv-train/` so accidental imports of `torch`/`ultralytics`/`src/data/` fail fast during local dev and CI.
- **Runtime Inference Package**: CPU **`onnxruntime`** (with `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `numpy`, `pillow` / `opencv-python-headless`, `python-multipart`, `pyyaml`, `pytest`, `httpx`).
- **Why CPU `onnxruntime` for Backend**: Keeps the Docker image small, guarantees identical behavior across all 3 teammates' laptops and GitHub Actions CPU runners, and avoids requiring NVIDIA Container Toolkit inside Docker Desktop for the live demo (YOLO11n/YOLOv8n ONNX CPU inference is fast enough for image uploads and bounded webcam demos; GPU parity/benchmarking can still be recorded in `src/models/benchmark.py` if needed).

### 6.3 Frontend Environment (Host Windows Node `v24.13.1` + `npm 11.8.0`)
- **Tooling**: **Vite + React 19 + TypeScript (`strict`) + TailwindCSS** (or Material UI / Shadcn/UI) + ESLint + Playwright.
- **Host Compatibility**: `Node.js v24.13.1` and `npm 11.8.0` work directly without needing `nvm` or Node upgrades.

### 6.4 Docker Runtime Environment (`docker compose up --build`)
- **Prerequisite**: Launch Docker Desktop (`desktop-linux` WSL 2 backend).
- **Services**:
  - `backend`: Multi-stage `python:3.12-slim` image running FastAPI + CPU `onnxruntime` + pure-Python `src/compliance/` & `src/tracking/`, mounting/copying only `models/*.onnx` & metadata.
  - `frontend`: Multi-stage `node:22-alpine` (or `node:24-alpine`) build served via `nginx:alpine`.
  - `mlflow`: Dedicated MLflow tracking & Model Registry container backed by persistent Docker volumes (`mlruns` / local DB) so concurrent pipeline runs avoid Windows SQLite file-lock issues.
