import type { ClassesResponse, ModelMetadataResponse } from '../../types/api';
import type { InferenceConfigState } from '../../types/monitoring';
import { Panel, StatusBadge } from '../ui/Primitives';

interface SettingsPageProps {
  metadata: ModelMetadataResponse | null;
  classes: ClassesResponse | null;
  metadataError: string | null;
  config: InferenceConfigState;
  onConfigChange: (next: InferenceConfigState) => void;
  onResetSession: () => void;
}

export function SettingsPage({
  metadata,
  classes,
  metadataError,
  config,
  onConfigChange,
  onResetSession,
}: SettingsPageProps) {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-[#0F172A]">
          Settings & Model Information
        </h1>
        <p className="mt-0.5 text-sm text-[#475569]">
          Runtime inference parameters sent to FastAPI and live ONNX artifact metadata.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <Panel
          title="Runtime Thresholds & Stream Cadence"
          subtitle="Applied to Image Analysis, Example Gallery, and Live Monitoring"
          className="lg:col-span-5"
        >
          <div className="space-y-4 text-xs">
            <div>
              <div className="flex justify-between font-medium text-[#0F172A] tabular-nums">
                <label htmlFor="conf-slider">Confidence Threshold</label>
                <span className="font-mono">{config.confThreshold.toFixed(2)}</span>
              </div>
              <input
                id="conf-slider"
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={config.confThreshold}
                onChange={(e) =>
                  onConfigChange({
                    ...config,
                    confThreshold: Number(e.target.value),
                  })
                }
                className="mt-1.5 w-full accent-[#0F172A]"
              />
            </div>

            <div>
              <div className="flex justify-between font-medium text-[#0F172A] tabular-nums">
                <label htmlFor="iou-slider">NMS IoU Threshold</label>
                <span className="font-mono">{config.iouThreshold.toFixed(2)}</span>
              </div>
              <input
                id="iou-slider"
                type="range"
                min={0.10}
                max={0.90}
                step={0.05}
                value={config.iouThreshold}
                onChange={(e) =>
                  onConfigChange({
                    ...config,
                    iouThreshold: Number(e.target.value),
                  })
                }
                className="mt-1.5 w-full accent-[#0F172A]"
              />
            </div>

            <div>
              <div className="flex justify-between font-medium text-[#0F172A] tabular-nums">
                <label htmlFor="interval-slider">Live Capture Interval</label>
                <span className="font-mono">{config.liveCaptureIntervalMs} ms</span>
              </div>
              <input
                id="interval-slider"
                type="range"
                min={100}
                max={1000}
                step={50}
                value={config.liveCaptureIntervalMs}
                onChange={(e) =>
                  onConfigChange({
                    ...config,
                    liveCaptureIntervalMs: Number(e.target.value),
                  })
                }
                className="mt-1.5 w-full accent-[#0F172A]"
              />
            </div>

            <label className="flex items-center gap-2 font-medium text-[#0F172A]">
              <input
                type="checkbox"
                checked={config.enableComplianceOnLive}
                onChange={(e) =>
                  onConfigChange({
                    ...config,
                    enableComplianceOnLive: e.target.checked,
                  })
                }
                className="h-4 w-4 accent-[#0F172A]"
              />
              Enable anonymous tracking & compliance confirmation on Live stream
            </label>

            <div className="flex flex-wrap gap-2 pt-2">
              <button
                type="button"
                onClick={() =>
                  onConfigChange({
                    confThreshold: metadata?.default_conf ?? 0.25,
                    iouThreshold: metadata?.default_iou ?? 0.50,
                    liveCaptureIntervalMs: 250,
                    enableComplianceOnLive: true,
                  })
                }
                className="rounded border border-[#D1D5DB] bg-white px-3 py-1.5 font-medium text-[#0F172A] hover:bg-slate-50"
              >
                Restore Model Defaults
              </button>
              <button
                type="button"
                onClick={onResetSession}
                className="rounded border border-red-300 bg-red-50 px-3 py-1.5 font-medium text-red-900 hover:bg-red-100"
              >
                Clear Session Telemetry
              </button>
            </div>
          </div>
        </Panel>

        <Panel
          title="Loaded ONNX Model Artifact Metadata"
          subtitle="Retrieved live from GET /model/metadata and GET /classes"
          className="lg:col-span-7"
        >
          {metadataError ? (
            <p className="text-xs text-red-700">{metadataError}</p>
          ) : !metadata ? (
            <p className="text-xs text-[#64748B]">Model metadata unavailable.</p>
          ) : (
            <div className="space-y-4 text-xs">
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                  <div className="text-[#475569]">Artifact & Version</div>
                  <div className="mt-0.5 font-mono font-semibold">
                    {metadata.model_name} ({metadata.model_version})
                  </div>
                </div>
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                  <div className="text-[#475569]">Architecture / Opset / Size</div>
                  <div className="mt-0.5 font-mono font-semibold">
                    {metadata.architecture} · opset {metadata.opset ?? '-'} ·{' '}
                    {metadata.artifact_size_mb ?? '-'} MB
                  </div>
                </div>
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                  <div className="text-[#475569]">ONNX SHA-256</div>
                  <div className="mt-0.5 truncate font-mono text-[11px]" title={metadata.onnx_sha256 ?? ''}>
                    {metadata.onnx_sha256 ?? '-'}
                  </div>
                </div>
                <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                  <div className="text-[#475569]">MLflow Run ID</div>
                  <div className="mt-0.5 truncate font-mono text-[11px]">
                    {metadata.mlflow_run_id ?? '-'}
                  </div>
                </div>
              </div>

              <div>
                <div className="font-semibold text-[#0F172A]">Canonical Classes</div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(classes?.classes ?? []).map((c) => (
                    <StatusBadge key={c.id} tone="neutral">
                      {c.id}: {c.name}
                    </StatusBadge>
                  ))}
                </div>
              </div>

              <div>
                <div className="font-semibold text-[#0F172A]">ONNX Tensor Bindings</div>
                <div className="mt-1.5 space-y-1 font-mono text-[11px] text-[#475569]">
                  {metadata.inputs.map((inp) => (
                    <div key={inp.name}>
                      input {inp.name}: {JSON.stringify(inp.shape)} ({inp.type})
                    </div>
                  ))}
                  {metadata.outputs.map((out) => (
                    <div key={out.name}>
                      output {out.name}: {JSON.stringify(out.shape)} ({out.type})
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
