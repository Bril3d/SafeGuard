import type {
  ClassesResponse,
  ComplianceMonitorResponse,
  ComplianceResetResponse,
  ErrorResponse,
  FramePredictResponse,
  HealthResponse,
  ModelMetadataResponse,
  PredictResponse,
} from '../types/api';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') || '/api';

export class ApiClientError extends Error {
  status: number;
  code: string;
  requestId?: string;

  constructor(message: string, status: number, code = 'api_error', requestId?: string) {
    super(message);
    this.name = 'ApiClientError';
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

async function parseErrorResponse(response: Response): Promise<ApiClientError> {
  try {
    const body = (await response.json()) as Partial<ErrorResponse>;
    if (body?.error?.message) {
      return new ApiClientError(
        body.error.message,
        response.status,
        body.error.code,
        body.error.request_id,
      );
    }
  } catch {
    // Fallback to HTTP status message if error body is non-JSON.
  }
  return new ApiClientError(
    `Request failed with HTTP status ${response.status}`,
    response.status,
  );
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  return (await response.json()) as HealthResponse;
}

export async function fetchModelMetadata(
  signal?: AbortSignal,
): Promise<ModelMetadataResponse> {
  const response = await fetch(`${API_BASE_URL}/model/metadata`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  return (await response.json()) as ModelMetadataResponse;
}

export async function fetchClasses(signal?: AbortSignal): Promise<ClassesResponse> {
  const response = await fetch(`${API_BASE_URL}/classes`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  return (await response.json()) as ClassesResponse;
}

export interface PredictOptions {
  conf?: number;
  iou?: number;
  onProgress?: (percent: number) => void;
  signal?: AbortSignal;
}

function buildQuery(params: Record<string, string | number | undefined>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined) {
      sp.set(k, String(v));
    }
  }
  const qs = sp.toString();
  return qs ? `?${qs}` : '';
}

export function uploadPredictWithProgress(
  file: Blob,
  fileName: string,
  options: PredictOptions = {},
): Promise<{ data: PredictResponse; clientE2EMs: number }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const qs = buildQuery({ conf: options.conf, iou: options.iou });
    const url = `${API_BASE_URL}/predict${qs}`;
    const t0 = performance.now();

    const onAbort = () => xhr.abort();
    if (options.signal) {
      if (options.signal.aborted) {
        reject(new DOMException('Aborted', 'AbortError'));
        return;
      }
      options.signal.addEventListener('abort', onAbort, { once: true });
    }

    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable && options.onProgress) {
        options.onProgress(Math.round((evt.loaded / evt.total) * 100));
      }
    };

    xhr.onload = () => {
      options.signal?.removeEventListener('abort', onAbort);
      const clientE2EMs = Math.round((performance.now() - t0) * 10) / 10;
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText) as PredictResponse;
          resolve({ data, clientE2EMs });
        } catch {
          reject(new ApiClientError('Malformed JSON response from /predict', xhr.status));
        }
        return;
      }
      try {
        const err = JSON.parse(xhr.responseText) as Partial<ErrorResponse>;
        reject(
          new ApiClientError(
            err.error?.message || `HTTP ${xhr.status}`,
            xhr.status,
            err.error?.code || 'api_error',
            err.error?.request_id,
          ),
        );
      } catch {
        reject(new ApiClientError(`HTTP ${xhr.status}`, xhr.status));
      }
    };

    xhr.onerror = () => {
      options.signal?.removeEventListener('abort', onAbort);
      reject(new ApiClientError('Network error while uploading image.', 0, 'network_error'));
    };

    xhr.onabort = () => {
      options.signal?.removeEventListener('abort', onAbort);
      reject(new DOMException('Aborted', 'AbortError'));
    };

    const form = new FormData();
    form.append('file', file, fileName);
    xhr.open('POST', url);
    xhr.send(form);
  });
}

export async function fetchAnnotatedImage(
  file: Blob,
  fileName: string,
  options: { conf?: number; iou?: number; format?: 'jpeg' | 'png'; signal?: AbortSignal } = {},
): Promise<{
  blobUrl: string;
  detectionCount: number;
  inferenceMs: number;
  totalMs: number;
  clientE2EMs: number;
}> {
  const qs = buildQuery({
    conf: options.conf,
    iou: options.iou,
    output_format: options.format ?? 'jpeg',
  });
  const form = new FormData();
  form.append('file', file, fileName);
  const t0 = performance.now();
  const response = await fetch(`${API_BASE_URL}/predict/visualize${qs}`, {
    method: 'POST',
    body: form,
    signal: options.signal,
  });
  const clientE2EMs = Math.round((performance.now() - t0) * 10) / 10;
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  const blob = await response.blob();
  const blobUrl = URL.createObjectURL(blob);
  return {
    blobUrl,
    detectionCount: Number(response.headers.get('X-Detection-Count') ?? '0'),
    inferenceMs: Number(response.headers.get('X-Inference-Ms') ?? '0'),
    totalMs: Number(response.headers.get('X-Total-Ms') ?? '0'),
    clientE2EMs,
  };
}

export async function predictFrame(
  frameBlob: Blob,
  options: { frameId?: string; conf?: number; iou?: number; signal?: AbortSignal } = {},
): Promise<{ data: FramePredictResponse; clientE2EMs: number }> {
  const qs = buildQuery({
    frame_id: options.frameId,
    conf: options.conf,
    iou: options.iou,
  });
  const form = new FormData();
  form.append('file', frameBlob, `${options.frameId ?? 'frame'}.jpg`);
  const t0 = performance.now();
  const response = await fetch(`${API_BASE_URL}/predict/frame${qs}`, {
    method: 'POST',
    body: form,
    signal: options.signal,
  });
  const clientE2EMs = Math.round((performance.now() - t0) * 10) / 10;
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  return { data: (await response.json()) as FramePredictResponse, clientE2EMs };
}

export async function predictComplianceFrame(
  frameBlob: Blob,
  options: {
    streamId?: string;
    frameId?: string;
    conf?: number;
    iou?: number;
    timestamp?: number;
    signal?: AbortSignal;
  } = {},
): Promise<{ data: ComplianceMonitorResponse; clientE2EMs: number }> {
  const qs = buildQuery({
    stream_id: options.streamId ?? 'default',
    frame_id: options.frameId,
    conf: options.conf,
    iou: options.iou,
    timestamp: options.timestamp,
  });
  const form = new FormData();
  form.append('file', frameBlob, `${options.frameId ?? 'frame'}.jpg`);
  const t0 = performance.now();
  const response = await fetch(`${API_BASE_URL}/predict/compliance${qs}`, {
    method: 'POST',
    body: form,
    signal: options.signal,
  });
  const clientE2EMs = Math.round((performance.now() - t0) * 10) / 10;
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  return {
    data: (await response.json()) as ComplianceMonitorResponse,
    clientE2EMs,
  };
}

export async function resetComplianceStream(
  streamId = 'default',
): Promise<ComplianceResetResponse> {
  const qs = buildQuery({ stream_id: streamId });
  const response = await fetch(`${API_BASE_URL}/compliance/reset${qs}`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  return (await response.json()) as ComplianceResetResponse;
}
