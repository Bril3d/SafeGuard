import { AlertCircle, Play } from 'lucide-react';
import { useState } from 'react';
import { getClassStyle } from '../../lib/ppeColors';
import { ApiClientError, uploadPredictWithProgress } from '../../services/api';
import type { PredictResponse } from '../../types/api';
import {
  CURATED_EXAMPLES,
  type CuratedExample,
  type InferenceConfigState,
} from '../../types/monitoring';
import { BoundingBoxCanvas } from '../canvas/BoundingBoxCanvas';
import { Panel, StatusBadge } from '../ui/Primitives';

interface ExampleGalleryPageProps {
  config: InferenceConfigState;
  onRecordSuccess: (params: {
    endpoint: string;
    sourcePage: 'gallery';
    detections: PredictResponse['detections'];
    timings: PredictResponse['timings_ms'];
    clientE2EMs: number;
  }) => void;
  onRecordError: (params: {
    endpoint: string;
    sourcePage: 'gallery';
    statusCode: number;
    clientE2EMs: number;
    message: string;
  }) => void;
}

export function ExampleGalleryPage({
  config,
  onRecordSuccess,
  onRecordError,
}: ExampleGalleryPageProps) {
  const [activeExample, setActiveExample] = useState<CuratedExample>(CURATED_EXAMPLES[0]);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<{
    exampleId: string;
    data: PredictResponse;
    clientE2EMs: number;
  } | null>(null);

  const runExample = async (ex: CuratedExample) => {
    setActiveExample(ex);
    setLoadingId(ex.id);
    setErrorMsg(null);
    const t0 = performance.now();

    try {
      const imgResp = await fetch(ex.imagePath);
      if (!imgResp.ok) {
        throw new Error(`Unable to fetch local sample asset (${imgResp.status})`);
      }
      const blob = await imgResp.blob();
      const predOut = await uploadPredictWithProgress(blob, ex.fileName, {
        conf: config.confThreshold,
        iou: config.iouThreshold,
      });
      setResult({ exampleId: ex.id, ...predOut });
      onRecordSuccess({
        endpoint: '/predict',
        sourcePage: 'gallery',
        detections: predOut.data.detections,
        timings: predOut.data.timings_ms,
        clientE2EMs: predOut.clientE2EMs,
      });
    } catch (err) {
      const elapsed = Math.round((performance.now() - t0) * 10) / 10;
      const status = err instanceof ApiClientError ? err.status : 0;
      const message = err instanceof Error ? err.message : 'Inference failed.';
      setErrorMsg(message);
      onRecordError({
        endpoint: '/predict',
        sourcePage: 'gallery',
        statusCode: status,
        clientE2EMs: elapsed,
        message,
      });
    } finally {
      setLoadingId(null);
    }
  };

  const activeResult =
    result && result.exampleId === activeExample.id ? result : null;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-[#0F172A]">
          Example Gallery
        </h1>
        <p className="mt-0.5 text-sm text-[#475569]">
          Licensed validation samples (CC BY 4.0) wired directly to live FastAPI ONNX inference.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {CURATED_EXAMPLES.map((ex) => {
          const isActive = activeExample.id === ex.id;
          const isLoading = loadingId === ex.id;
          return (
            <div
              key={ex.id}
              className={`flex flex-col overflow-hidden rounded-md border bg-white ${
                isActive ? 'border-[#0F172A] ring-1 ring-[#0F172A]' : 'border-[#D1D5DB]'
              }`}
            >
              <img
                src={ex.imagePath}
                alt={ex.title}
                className="h-36 w-full object-cover bg-slate-900"
              />
              <div className="flex flex-1 flex-col justify-between p-3">
                <div>
                  <div className="text-xs font-semibold text-[#0F172A]">{ex.title}</div>
                  <div className="mt-0.5 text-[11px] text-[#475569]">{ex.sourceLabel}</div>
                  <p className="mt-1.5 text-xs text-[#475569]">{ex.notes}</p>
                </div>
                <button
                  type="button"
                  onClick={() => void runExample(ex)}
                  disabled={loadingId !== null}
                  className="mt-3 inline-flex w-full items-center justify-center gap-1.5 rounded border border-[#0F172A] bg-[#0F172A] px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800 disabled:opacity-50"
                >
                  <Play className="h-3.5 w-3.5" />
                  {isLoading ? 'Running...' : 'Run Inference'}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {errorMsg ? (
        <div className="flex items-start gap-2 rounded border border-red-300 bg-red-50 p-3 text-xs text-red-900">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" />
          <span>{errorMsg}</span>
        </div>
      ) : null}

      <Panel
        title={`Sample Inspection: ${activeExample.title}`}
        subtitle={activeExample.sourceLabel}
        actions={
          activeResult ? (
            <div className="flex items-center gap-2">
              <StatusBadge tone="info">
                Infer: {activeResult.data.timings_ms.inference_ms.toFixed(1)} ms
              </StatusBadge>
              <StatusBadge tone="neutral">
                E2E: {activeResult.clientE2EMs.toFixed(1)} ms
              </StatusBadge>
            </div>
          ) : null
        }
      >
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="lg:col-span-8">
            {activeResult ? (
              <BoundingBoxCanvas
                imageSrc={activeExample.imagePath}
                imageWidth={activeResult.data.image_width}
                imageHeight={activeResult.data.image_height}
                detections={activeResult.data.detections}
                alt={activeExample.title}
              />
            ) : (
              <div className="overflow-hidden rounded border border-[#D1D5DB] bg-[#0F172A]">
                <img
                  src={activeExample.imagePath}
                  alt={activeExample.title}
                  className="block h-auto max-h-[460px] w-full object-contain"
                />
              </div>
            )}
          </div>
          <div className="lg:col-span-4">
            {!activeResult ? (
              <p className="text-xs text-[#475569]">
                Click <strong>Run Inference</strong> on any sample card to execute real ONNX detection.
              </p>
            ) : (
              <div className="space-y-3 text-xs">
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5 tabular-nums">
                  Detected <strong>{activeResult.data.detection_count}</strong> objects in{' '}
                  <code>{activeResult.data.timings_ms.total_ms.toFixed(1)} ms</code> (server total).
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {activeResult.data.detections.map((d, i) => {
                    const st = getClassStyle(d.class_name);
                    return (
                      <span
                        key={`${d.class_id}-${i}`}
                        className={`rounded border px-2 py-0.5 font-medium tabular-nums ${st.badgeBg} ${st.badgeText}`}
                      >
                        {st.label} {(d.confidence * 100).toFixed(0)}%
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </Panel>
    </div>
  );
}
