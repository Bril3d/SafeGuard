import { AlertCircle, Eye, Layers, UploadCloud } from 'lucide-react';
import { type ChangeEvent, type DragEvent, useEffect, useRef, useState } from 'react';
import { getClassStyle } from '../../lib/ppeColors';
import {
  ApiClientError,
  fetchAnnotatedImage,
  uploadPredictWithProgress,
} from '../../services/api';
import type { PredictResponse } from '../../types/api';
import type { InferenceConfigState } from '../../types/monitoring';
import { BoundingBoxCanvas } from '../canvas/BoundingBoxCanvas';
import { Panel, StatusBadge } from '../ui/Primitives';

const ALLOWED_TYPES = new Set([
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/webp',
  'image/bmp',
]);
const MAX_CLIENT_BYTES = 10 * 1024 * 1024;

interface ImageAnalysisPageProps {
  config: InferenceConfigState;
  onRecordSuccess: (params: {
    endpoint: string;
    sourcePage: 'image';
    detections: PredictResponse['detections'];
    timings: PredictResponse['timings_ms'];
    clientE2EMs: number;
  }) => void;
  onRecordError: (params: {
    endpoint: string;
    sourcePage: 'image';
    statusCode: number;
    clientE2EMs: number;
    message: string;
  }) => void;
}

export function ImageAnalysisPage({
  config,
  onRecordSuccess,
  onRecordError,
}: ImageAnalysisPageProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [annotatedUrl, setAnnotatedUrl] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'canvas' | 'opencv'>('canvas');
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<{
    data: PredictResponse;
    clientE2EMs: number;
  } | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
    };
  }, [annotatedUrl]);

  const handleSelectFile = (file: File | undefined) => {
    setErrorMsg(null);
    if (!file) return;

    if (file.type && !ALLOWED_TYPES.has(file.type.toLowerCase())) {
      setErrorMsg(`Unsupported file format (${file.type}). Use JPEG, PNG, WEBP, or BMP.`);
      return;
    }
    if (file.size === 0 || file.size > MAX_CLIENT_BYTES) {
      setErrorMsg('File must be non-empty and under 10 MB.');
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (annotatedUrl) {
      URL.revokeObjectURL(annotatedUrl);
      setAnnotatedUrl(null);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setUploadProgress(0);
  };

  const onFileInputChange = (evt: ChangeEvent<HTMLInputElement>) => {
    handleSelectFile(evt.target.files?.[0]);
  };

  const onDrop = (evt: DragEvent<HTMLDivElement>) => {
    evt.preventDefault();
    setIsDragging(false);
    handleSelectFile(evt.dataTransfer.files?.[0]);
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setErrorMsg(null);
    setUploadProgress(0);

    const t0 = performance.now();
    try {
      const [predOut, visOut] = await Promise.all([
        uploadPredictWithProgress(selectedFile, selectedFile.name, {
          conf: config.confThreshold,
          iou: config.iouThreshold,
          onProgress: (pct) => setUploadProgress(pct),
        }),
        fetchAnnotatedImage(selectedFile, selectedFile.name, {
          conf: config.confThreshold,
          iou: config.iouThreshold,
          format: 'jpeg',
        }),
      ]);

      if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
      setAnnotatedUrl(visOut.blobUrl);
      setResult(predOut);
      onRecordSuccess({
        endpoint: '/predict',
        sourcePage: 'image',
        detections: predOut.data.detections,
        timings: predOut.data.timings_ms,
        clientE2EMs: predOut.clientE2EMs,
      });
    } catch (err) {
      const elapsed = Math.round((performance.now() - t0) * 10) / 10;
      const status = err instanceof ApiClientError ? err.status : 0;
      const message =
        err instanceof Error ? err.message : 'Inference request failed.';
      setErrorMsg(message);
      onRecordError({
        endpoint: '/predict',
        sourcePage: 'image',
        statusCode: status,
        clientE2EMs: elapsed,
        message,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-[#0F172A]">
            Image Analysis
          </h1>
          <p className="mt-0.5 text-sm text-[#475569]">
            Upload construction-site images for ONNX inference, OpenCV visualization, and structured inspection.
          </p>
        </div>
        {result ? (
          <div className="flex items-center gap-2">
            <StatusBadge tone="info">
              Server Infer: {result.data.timings_ms.inference_ms.toFixed(1)} ms
            </StatusBadge>
            <StatusBadge tone="neutral">
              Client E2E: {result.clientE2EMs.toFixed(1)} ms
            </StatusBadge>
          </div>
        ) : null}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <Panel
          title="Inspection Input & Visual Output"
          subtitle="Drag and drop an image file or browse from disk"
          className="xl:col-span-7"
          actions={
            result ? (
              <div className="inline-flex rounded border border-[#D1D5DB] bg-[#F9FAFB] p-0.5 text-xs">
                <button
                  type="button"
                  onClick={() => setViewMode('canvas')}
                  className={`inline-flex items-center gap-1 rounded px-2 py-1 font-medium ${
                    viewMode === 'canvas'
                      ? 'bg-[#0F172A] text-white'
                      : 'text-[#475569] hover:text-[#0F172A]'
                  }`}
                >
                  <Layers className="h-3.5 w-3.5" />
                  Canvas Overlay
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('opencv')}
                  className={`inline-flex items-center gap-1 rounded px-2 py-1 font-medium ${
                    viewMode === 'opencv'
                      ? 'bg-[#0F172A] text-white'
                      : 'text-[#475569] hover:text-[#0F172A]'
                  }`}
                >
                  <Eye className="h-3.5 w-3.5" />
                  Server Annotated
                </button>
              </div>
            ) : null
          }
        >
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            className={`rounded border-2 border-dashed p-5 text-center transition-colors ${
              isDragging
                ? 'border-sky-600 bg-sky-50'
                : 'border-[#CBD5E1] bg-[#F8FAFC]'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              onChange={onFileInputChange}
              className="sr-only"
              data-testid="image-upload-input"
            />
            <UploadCloud className="mx-auto h-7 w-7 text-[#475569]" />
            <p className="mt-2 text-sm font-medium text-[#0F172A]">
              Drop image here or{' '}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="text-sky-700 underline hover:text-sky-900"
              >
                choose a file
              </button>
            </p>
            <p className="mt-1 text-xs text-[#64748B]">
              JPEG, PNG, WEBP, or BMP up to 10 MB ({selectedFile ? selectedFile.name : 'no file selected'})
            </p>

            {selectedFile ? (
              <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
                <button
                  type="button"
                  onClick={() => void runAnalysis()}
                  disabled={loading}
                  className="rounded border border-[#0F172A] bg-[#0F172A] px-4 py-1.5 text-xs font-medium text-white hover:bg-slate-800 disabled:opacity-50"
                >
                  {loading ? 'Analyzing...' : 'Run Inspection'}
                </button>
                <span className="text-xs text-[#475569] tabular-nums">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </span>
              </div>
            ) : null}

            {loading || uploadProgress > 0 ? (
              <div className="mx-auto mt-3 max-w-sm">
                <div className="flex justify-between text-[11px] text-[#475569] tabular-nums">
                  <span>Upload progress</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-slate-200">
                  <div
                    className="h-full bg-sky-700 transition-all"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            ) : null}
          </div>

          {errorMsg ? (
            <div className="mt-3 flex items-start gap-2 rounded border border-red-300 bg-red-50 p-3 text-xs text-red-900">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" />
              <span>{errorMsg}</span>
            </div>
          ) : null}

          <div className="mt-4">
            {viewMode === 'opencv' && annotatedUrl ? (
              <div className="overflow-hidden rounded border border-[#D1D5DB] bg-[#0F172A]">
                <img
                  src={annotatedUrl}
                  alt="Server-rendered OpenCV annotation output"
                  className="block h-auto max-h-[540px] w-full object-contain"
                />
              </div>
            ) : previewUrl && result ? (
              <BoundingBoxCanvas
                imageSrc={previewUrl}
                imageWidth={result.data.image_width}
                imageHeight={result.data.image_height}
                detections={result.data.detections}
                alt="Uploaded image with ONNX bounding boxes"
              />
            ) : previewUrl ? (
              <div className="overflow-hidden rounded border border-[#D1D5DB] bg-[#0F172A]">
                <img
                  src={previewUrl}
                  alt="Uploaded image preview"
                  className="block h-auto max-h-[540px] w-full object-contain"
                />
              </div>
            ) : (
              <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] py-12 text-center text-xs text-[#64748B]">
                Select an image to preview and execute inference.
              </div>
            )}
          </div>
        </Panel>

        <Panel
          title="Detection Results & Latency Breakdown"
          subtitle="Structured detections and server-side timings"
          className="xl:col-span-5"
        >
          {!result ? (
            <p className="py-6 text-xs text-[#64748B]">
              Run inspection to populate bounding-box detections, class counts, and stage-by-stage timing metrics.
            </p>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-2 text-xs tabular-nums">
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                  <div className="text-[#475569]">Resolution</div>
                  <div className="mt-0.5 font-mono font-semibold">
                    {result.data.image_width} × {result.data.image_height}
                  </div>
                </div>
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                  <div className="text-[#475569]">Detections Returned</div>
                  <div className="mt-0.5 font-semibold">
                    {result.data.detection_count}
                  </div>
                </div>
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                  <div className="text-[#475569]">Server Inference / Total</div>
                  <div className="mt-0.5 font-mono font-semibold">
                    {result.data.timings_ms.inference_ms.toFixed(1)} /{' '}
                    {result.data.timings_ms.total_ms.toFixed(1)} ms
                  </div>
                </div>
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                  <div className="text-[#475569]">Pre / Postprocess</div>
                  <div className="mt-0.5 font-mono font-semibold">
                    {result.data.timings_ms.preprocess_ms.toFixed(1)} /{' '}
                    {result.data.timings_ms.postprocess_ms.toFixed(1)} ms
                  </div>
                </div>
              </div>

              {result.data.detections.length === 0 ? (
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-3 text-xs text-[#475569]">
                  No detections exceeded confidence threshold{' '}
                  <code>{result.data.conf_threshold.toFixed(2)}</code>.
                </div>
              ) : (
                <div className="max-h-[380px] overflow-auto rounded border border-[#E5E7EB]">
                  <table className="w-full text-left text-xs tabular-nums">
                    <thead className="sticky top-0 bg-[#F9FAFB] text-[#475569]">
                      <tr className="border-b border-[#E5E7EB]">
                        <th className="px-3 py-2 font-medium">Class</th>
                        <th className="px-3 py-2 font-medium">Conf</th>
                        <th className="px-3 py-2 font-medium">Box (xyxy)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.data.detections.map((det, idx) => {
                        const style = getClassStyle(det.class_name);
                        return (
                          <tr
                            key={`${det.class_id}-${idx}`}
                            className="border-b border-[#F3F4F6] last:border-b-0"
                          >
                            <td className="px-3 py-2">
                              <span
                                className={`inline-block rounded border px-1.5 py-0.5 font-medium ${style.badgeBg} ${style.badgeText}`}
                              >
                                {style.label}
                              </span>
                            </td>
                            <td className="px-3 py-2 font-mono">
                              {(det.confidence * 100).toFixed(1)}%
                            </td>
                            <td className="px-3 py-2 font-mono text-[11px] text-[#475569]">
                              [{det.bbox_xyxy.map((v) => Math.round(v)).join(', ')}]
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
