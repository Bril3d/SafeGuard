# Backend API Contract (`backend/`)

**Sources**: [`docs/reference/professor_project_guide.pdf`](../reference/professor_project_guide.pdf) (Section 5.1, p.6) and [`docs/reference/safeguard_specification_v1.pdf`](../reference/safeguard_specification_v1.pdf) (Section 9, p.14).  
**Runtime Rules**:
- Framework: FastAPI + Pydantic v2 + ONNX Runtime.
- OpenAPI / Swagger UI must be accessible and documented at `/docs` (`http://localhost:8000/docs`).
- The ONNX model is loaded **once** during FastAPI startup/lifespan and reused across requests.
- `backend/` must **never** import training code (`src/data/`, `src/models/train.py`) or require training datasets.

---

## 1. Mandatory Endpoints (Professor's Guide & `SG-SPEC-001`)

### 1.1 `GET /health`
- **Requirement**: Mandatory (Prof. Guide p.6; `SG-SPEC-001` p.14).
- **Contract**: Returns server health status, model availability/readiness, ONNX Runtime/execution provider information, GPU availability where detectable, and loaded model version.
- **Expected Error Behavior**: Returns health payload cleanly; indicates degraded/unavailable model status when model is not loaded (exact HTTP status/schema for degraded `/health` vs `503` on inference routes to be finalized in Phase 3 API schema design).

### 1.2 `GET /model/metadata`
- **Requirement**: Mandatory (Prof. Guide p.6; `SG-SPEC-001` p.14).
- **Contract**: Returns model metadata including model name, architecture, version, canonical class names/mapping, input configuration (e.g., expected input resolution/format), validation metrics, and artifact metadata (e.g., ONNX artifact info/hash).
- **Expected Error Behavior**: `503 Service Unavailable` if model metadata/artifact is not initialized.

### 1.3 `POST /predict`
- **Requirement**: Mandatory (Prof. Guide p.6; `SG-SPEC-001` p.14).
- **Input**: Multipart `UploadFile` image.
- **Validation**:
  - Validate MIME type **and** actual image decoding (never trust filename extension or `Content-Type` header alone).
  - Enforce configured upload file size limit and accepted image dimension limits.
- **Output (Structured JSON)**:
  - Model version (`model_version`)
  - Structured detections list (`class_id`, `class_name`, `confidence`, `bbox_xyxy` — illustrative example from `SG-SPEC-001` p.14)
  - Image dimensions (`width`, `height`)
  - Server-side timing (`inference_time_ms` and, where practical, separate preprocessing/inference/postprocessing timings per `SG-SPEC-001` p.15)
  - Compliance / potential alert payload (if integrated in prediction response; raw detections remain separate from derived potential alerts per `SG-SPEC-001` p.13).

### 1.4 `POST /predict/visualize`
- **Requirement**: Mandatory (Prof. Guide p.6; `SG-SPEC-001` p.14).
- **Input**: Multipart `UploadFile` image (same validation rules as `POST /predict`).
- **Contract**: Runs the same preprocessing + ONNX Runtime inference + postprocessing pipeline and returns the annotated image rendered with bounding boxes (distinct colors per class), class labels, and confidence scores.

---

## 2. Permitted Additional Endpoints (`SG-SPEC-001` Extensions)

### 2.1 `GET /classes`
- **Requirement**: Convenience endpoint (`SG-SPEC-001` p.14).
- **Contract**: Returns the canonical class names and integer IDs loaded from the approved taxonomy (`configs/class_mapping.yaml` / model metadata).

### 2.2 `POST /predict/frame`
- **Requirement**: Optional webcam/video frame inference endpoint (`SG-SPEC-001` p.14; explicitly permitted by project instructions).
- **Contract**: Accepts a captured webcam/video frame upload and shares the `/predict` validation, preprocessing, ONNX inference, and postprocessing service layer.

---

## 3. Error Handling & Operational Requirements (`SG-SPEC-001` p.14)

### 3.1 Semantic HTTP Status Codes
| HTTP Status | Required Meaning |
|---|---|
| `400 Bad Request` | Invalid, corrupt, or unsupported image payload (including MIME/magic-byte mismatch or undecodable image). |
| `422 Unprocessable Entity` | Request validation / Pydantic schema failure (e.g., missing file field, empty/invalid parameter, or boundary limit breach where mapped to validation). |
| `503 Service Unavailable` | Model is not initialized or inference service is unavailable. |
| `500 Internal Server Error` | Unexpected server-side failure; must return a safe structured JSON error without leaking secrets, filesystem internals, or stack traces. |

### 3.2 Logging, Timing & Testing Requirements
- **Structured Logging**: Emit structured logs with per-request correlation/request IDs.
- **Latency Separation**: Measure server-side preprocessing, inference, and postprocessing durations separately from client-side end-to-end latency (`SG-SPEC-001` p.14–15).
- **Pytest Coverage**: `backend/tests/test_api.py` must verify valid images, invalid files, empty uploads, uninitialized model (`503`), and boundary cases (`SG-SPEC-001` p.14; Prof. Guide p.7).

---

## 4. Unresolved API Contract Decisions (Pending Approval)

The reference PDFs intentionally leave the following detailed schema and parameter choices open for Phase 3 API design:

1. **Exact Pydantic Field Names & Nested Schemas**:
   - Exact JSON structure for `GET /health` and `GET /model/metadata`.
   - Exact field names for image dimensions (`image_width`/`image_height` vs `image_size: {width, height}`) and timing breakdown (`preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_server_ms`) on `POST /predict`.
   - Whether `POST /predict` also returns derived geometric associations and potential alert objects alongside `detections`, or exposes them separately.
2. **Upload & Image Guardrail Limits**:
   - Maximum allowed upload size in MB (e.g., 5 MB or 10 MB) and maximum/minimum pixel dimensions.
   - Supported image formats (e.g., JPEG, PNG, WebP).
   - Default confidence and IoU/NMS query parameters or fixed validation-tuned defaults.
3. **`POST /predict/visualize` Response Media Format**:
   - Encoded image format returned (`image/png` vs `image/jpeg`) and whether server timing metadata is also returned in HTTP response headers.
4. **Webcam Ingestion Endpoint Choice**:
   - Whether the React webcam client calls `POST /predict` directly or uses a dedicated `POST /predict/frame` route (with optional temporary session/track ID state).
