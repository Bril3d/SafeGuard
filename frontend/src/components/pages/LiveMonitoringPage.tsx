import { AlertCircle, Camera, RotateCcw, Square } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { useWebcamMonitor } from '../../hooks/useWebcamMonitor';
import { getClassStyle } from '../../lib/ppeColors';
import type {
  ComplianceEventSchema,
  DetectionSchema,
  TimingBreakdownSchema,
} from '../../types/api';
import type { InferenceConfigState } from '../../types/monitoring';
import { Panel, StatusBadge } from '../ui/Primitives';

interface LiveMonitoringPageProps {
  config: InferenceConfigState;
  onLiveActiveChange: (active: boolean) => void;
  onRecordSuccess: (params: {
    endpoint: string;
    sourcePage: 'live';
    detections: DetectionSchema[];
    timings: TimingBreakdownSchema;
    clientE2EMs: number;
    events?: ComplianceEventSchema[];
    streamId?: string;
  }) => void;
  onRecordError: (params: {
    endpoint: string;
    sourcePage: 'live';
    statusCode: number;
    clientE2EMs: number;
    message: string;
  }) => void;
  onFpsChange: (fps: number | null) => void;
}

export function LiveMonitoringPage({
  config,
  onLiveActiveChange,
  onRecordSuccess,
  onRecordError,
  onFpsChange,
}: LiveMonitoringPageProps) {
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const { videoRef, status, errorMsg, liveState, start, stop, resetStreamState } =
    useWebcamMonitor({
      config,
      onSuccess: onRecordSuccess,
      onError: onRecordError,
      onFpsChange,
    });

  useEffect(() => {
    onLiveActiveChange(status === 'running');
  }, [onLiveActiveChange, status]);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!liveState || liveState.imageWidth <= 0 || liveState.imageHeight <= 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    canvas.width = liveState.imageWidth;
    canvas.height = liveState.imageHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.font = '600 13px Inter, system-ui, sans-serif';
    for (const det of liveState.detections) {
      const style = getClassStyle(det.class_name);
      const [x1, y1, x2, y2] = det.bbox_xyxy;
      ctx.fillStyle = style.fill;
      ctx.fillRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
      ctx.strokeStyle = style.stroke;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));

      const label = `${style.label} ${(det.confidence * 100).toFixed(0)}%`;
      const tw = ctx.measureText(label).width;
      const py = y1 - 18 >= 0 ? y1 - 18 : y1;
      ctx.fillStyle = style.stroke;
      ctx.fillRect(x1, py, tw + 8, 18);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, x1 + 4, py + 13);
    }

    for (const assoc of liveState.associations) {
      const [x1, y1] = assoc.bbox_xyxy;
      const tag = `#${assoc.track_id} ${assoc.overall_status.replaceAll('_', ' ')}`;
      const bg =
        assoc.overall_status === 'potential_violation'
          ? '#B91C1C'
          : assoc.overall_status === 'pending_confirmation'
            ? '#B45309'
            : assoc.overall_status === 'compliant'
              ? '#15803D'
              : '#475569';
      const tw = ctx.measureText(tag).width;
      ctx.fillStyle = bg;
      ctx.fillRect(x1, Math.max(0, y1 + 2), tw + 10, 18);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(tag, x1 + 5, Math.max(14, y1 + 15));
    }
  }, [liveState]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-[#0F172A]">
            Live Monitoring
          </h1>
          <p className="mt-0.5 text-sm text-[#475569]">
            Bounded single-in-flight webcam capture with stale-frame dropping, anonymous tracking, and server/client latency readouts.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {status !== 'running' ? (
            <button
              type="button"
              onClick={() => void start()}
              disabled={status === 'starting'}
              className="inline-flex items-center gap-1.5 rounded border border-emerald-800 bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-50"
            >
              <Camera className="h-3.5 w-3.5" />
              {status === 'starting' ? 'Requesting Camera...' : 'Start Camera'}
            </button>
          ) : (
            <button
              type="button"
              onClick={stop}
              className="inline-flex items-center gap-1.5 rounded border border-red-800 bg-red-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-red-800"
            >
              <Square className="h-3.5 w-3.5" />
              Stop Camera
            </button>
          )}
          <button
            type="button"
            onClick={() => void resetStreamState()}
            className="inline-flex items-center gap-1.5 rounded border border-[#D1D5DB] bg-white px-3 py-1.5 text-xs font-medium text-[#0F172A] hover:bg-slate-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset Track State
          </button>
        </div>
      </div>

      {errorMsg ? (
        <div className="flex items-start gap-2 rounded border border-red-300 bg-red-50 p-3 text-xs text-red-900">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" />
          <span>{errorMsg}</span>
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <Panel
          title="Webcam Stream & Canvas Detection Overlay"
          subtitle={`Mode: ${config.enableComplianceOnLive ? 'Tracking + Compliance Rules (/predict/compliance)' : 'Stateless Frame (/predict/frame)'}`}
          className="xl:col-span-8"
          actions={
            <div className="flex flex-wrap gap-2">
              <StatusBadge tone={status === 'running' ? 'ok' : 'neutral'}>
                {status}
              </StatusBadge>
              <StatusBadge tone="info">
                FPS: {liveState ? liveState.fps.toFixed(1) : '0.0'}
              </StatusBadge>
            </div>
          }
        >
          <div className="relative flex min-h-[360px] items-center justify-center overflow-hidden rounded border border-[#D1D5DB] bg-[#0F172A]">
            <video
              ref={videoRef}
              muted
              playsInline
              className="block max-h-[480px] w-full object-contain"
            />
            <canvas
              ref={overlayRef}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain"
            />
            {status !== 'running' ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0F172A]/85 px-4 text-center text-slate-300">
                <Camera className="h-8 w-8 text-slate-400" />
                <p className="mt-2 text-xs">
                  Click <strong>Start Camera</strong> to grant browser camera access. No raw video frames are saved.
                </p>
              </div>
            ) : null}
          </div>
        </Panel>

        <Panel
          title="Live Stream Telemetry"
          subtitle="Separated server vs. end-to-end client timing"
          className="xl:col-span-4"
        >
          <div className="space-y-3 text-xs tabular-nums">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                <div className="text-[#475569]">Server Inference</div>
                <div className="mt-0.5 font-mono text-base font-semibold">
                  {liveState ? `${liveState.serverInferenceMs.toFixed(1)} ms` : '-'}
                </div>
              </div>
              <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                <div className="text-[#475569]">Client E2E</div>
                <div className="mt-0.5 font-mono text-base font-semibold">
                  {liveState ? `${liveState.clientE2EMs.toFixed(1)} ms` : '-'}
                </div>
              </div>
              <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                <div className="text-[#475569]">Processed Frames</div>
                <div className="mt-0.5 font-semibold">
                  {liveState?.processedFrames ?? 0}
                </div>
              </div>
              <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5">
                <div className="text-[#475569]">Dropped Stale</div>
                <div className="mt-0.5 font-semibold">
                  {liveState?.droppedFrames ?? 0}
                </div>
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-[#0F172A]">Tracked Associations</h3>
              {!liveState || liveState.associations.length === 0 ? (
                <p className="mt-1 text-[#64748B]">No active tracked persons in current frame.</p>
              ) : (
                <ul className="mt-2 space-y-1.5">
                  {liveState.associations.map((a) => (
                    <li
                      key={a.track_id}
                      className="flex items-center justify-between rounded border border-[#E5E7EB] bg-[#F9FAFB] px-2.5 py-2"
                    >
                      <span>
                        Track #{a.track_id} · Helmet: {a.helmet_status} · Vest: {a.vest_status}
                      </span>
                      <StatusBadge
                        tone={
                          a.overall_status === 'potential_violation'
                            ? 'alert'
                            : a.overall_status === 'pending_confirmation'
                              ? 'warn'
                              : a.overall_status === 'compliant'
                                ? 'ok'
                                : 'neutral'
                        }
                      >
                        {a.overall_status.replaceAll('_', ' ')}
                      </StatusBadge>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
