import { Activity, AlertTriangle, Camera, ImageIcon, Images } from 'lucide-react';
import type { HealthResponse, ModelMetadataResponse } from '../../types/api';
import type {
  InferenceConfigState,
  PageId,
  RequestTelemetryRecord,
  ReviewedComplianceEvent,
  SessionStats,
} from '../../types/monitoring';
import { MetricTile, Panel, StatusBadge } from '../ui/Primitives';

interface OverviewPageProps {
  health: HealthResponse | null;
  metadata: ModelMetadataResponse | null;
  stats: SessionStats;
  alerts: ReviewedComplianceEvent[];
  recentRecords: RequestTelemetryRecord[];
  config: InferenceConfigState;
  onNavigate: (page: PageId) => void;
}

export function OverviewPage({
  health,
  metadata,
  stats,
  alerts,
  recentRecords,
  config,
  onNavigate,
}: OverviewPageProps) {
  const totalPositivePpe =
    stats.ppeCounts.helmet + stats.ppeCounts.vest + stats.ppeCounts.gloves;
  const pendingAlerts = alerts.filter((a) => a.reviewStatus === 'pending_review');

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-[#0F172A]">
            Operational Overview
          </h1>
          <p className="mt-0.5 text-sm text-[#475569]">
            Real-time session telemetry from the active SafeGuard ONNX Runtime service.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => onNavigate('image')}
            className="inline-flex items-center gap-1.5 rounded border border-[#0F172A] bg-[#0F172A] px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800"
          >
            <ImageIcon className="h-3.5 w-3.5" />
            Analyze Image
          </button>
          <button
            type="button"
            onClick={() => onNavigate('live')}
            className="inline-flex items-center gap-1.5 rounded border border-[#D1D5DB] bg-white px-3 py-1.5 text-xs font-medium text-[#0F172A] hover:bg-slate-50"
          >
            <Camera className="h-3.5 w-3.5" />
            Open Live Monitor
          </button>
          <button
            type="button"
            onClick={() => onNavigate('gallery')}
            className="inline-flex items-center gap-1.5 rounded border border-[#D1D5DB] bg-white px-3 py-1.5 text-xs font-medium text-[#0F172A] hover:bg-slate-50"
          >
            <Images className="h-3.5 w-3.5" />
            Example Gallery
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Total Persons Detected"
          value={stats.totalPersonsDetected}
          detail={`Across ${stats.successfulRequests} completed inference calls`}
          tone="info"
        />
        <MetricTile
          label="PPE Detections"
          value={totalPositivePpe}
          detail={`Helmet ${stats.ppeCounts.helmet} | Vest ${stats.ppeCounts.vest} | Gloves ${stats.ppeCounts.gloves}`}
          tone="ok"
        />
        <MetricTile
          label="Missing-PPE Predictions"
          value={stats.missingPpePredictions}
          detail="Explicit no_helmet detections"
          tone={stats.missingPpePredictions > 0 ? 'warn' : 'neutral'}
        />
        <MetricTile
          label="Potential Violation Alerts"
          value={stats.potentialAlertsEmitted}
          detail={`${pendingAlerts.length} awaiting human review`}
          tone={pendingAlerts.length > 0 ? 'alert' : 'neutral'}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel
          title="Runtime & Threshold Configuration"
          subtitle="Active inference contract reported by FastAPI"
          className="lg:col-span-2"
        >
          <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
            <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-3">
              <div className="text-[#475569]">Architecture</div>
              <div className="mt-1 font-semibold text-[#0F172A]">
                {metadata?.architecture ?? 'Unavailable'}
              </div>
            </div>
            <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-3">
              <div className="text-[#475569]">Model Status</div>
              <div className="mt-1 font-semibold text-[#0F172A]">
                {health?.model_loaded ? 'Loaded' : 'Unavailable'}
              </div>
            </div>
            <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-3">
              <div className="text-[#475569]">ONNX Opset / Input</div>
              <div className="mt-1 font-mono font-semibold text-[#0F172A]">
                {metadata ? `opset ${metadata.opset ?? '-'} / ${metadata.imgsz}px` : '-'}
              </div>
            </div>
            <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-3">
              <div className="text-[#475569]">Confidence Cutoff</div>
              <div className="mt-1 font-mono font-semibold text-[#0F172A]">
                {config.confThreshold.toFixed(2)}
              </div>
            </div>
            <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-3">
              <div className="text-[#475569]">NMS IoU Cutoff</div>
              <div className="mt-1 font-mono font-semibold text-[#0F172A]">
                {config.iouThreshold.toFixed(2)}
              </div>
            </div>
            <div className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-3">
              <div className="text-[#475569]">Mean / P95 E2E Latency</div>
              <div className="mt-1 font-mono font-semibold text-[#0F172A] tabular-nums">
                {stats.meanApiResponseMs !== null
                  ? `${stats.meanApiResponseMs} / ${stats.p95ApiResponseMs} ms`
                  : 'No requests yet'}
              </div>
            </div>
          </div>

          <div className="mt-4 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-950">
            <strong>Safety Scope:</strong> Missing positive vest or gloves detections resolve as{' '}
            <code>unknown</code> rather than violations. Only confirmed explicit{' '}
            <code>no_helmet</code> detections raise potential review alerts.
          </div>
        </Panel>

        <Panel
          title="Recent Potential Alerts"
          subtitle="Requires operator verification"
          actions={
            <button
              type="button"
              onClick={() => onNavigate('alerts')}
              className="text-xs font-medium text-sky-700 hover:underline"
            >
              Open queue
            </button>
          }
        >
          {alerts.length === 0 ? (
            <div className="flex flex-col items-start gap-2 py-4 text-xs text-[#475569]">
              <AlertTriangle className="h-4 w-4 text-slate-400" />
              <p>
                No potential missing-helmet alerts recorded in this session yet. Run Live Monitoring or Image Compliance to evaluate frames.
              </p>
            </div>
          ) : (
            <ul className="space-y-2">
              {alerts.slice(0, 5).map((evt) => (
                <li
                  key={evt.id}
                  className="rounded border border-[#E5E7EB] bg-[#F9FAFB] p-2.5 text-xs"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-[#0F172A]">
                      Track #{evt.track_id}
                    </span>
                    <StatusBadge
                      tone={
                        evt.reviewStatus === 'dismissed_false_alarm'
                          ? 'neutral'
                          : evt.reviewStatus === 'verified_potential_event'
                            ? 'ok'
                            : 'alert'
                      }
                    >
                      {evt.reviewStatus.replaceAll('_', ' ')}
                    </StatusBadge>
                  </div>
                  <div className="mt-1 text-[#475569]">{evt.message}</div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <Panel
        title="Session Request Activity"
        subtitle="Latest real FastAPI inference responses"
        actions={
          <span className="inline-flex items-center gap-1 text-xs text-[#475569] tabular-nums">
            <Activity className="h-3.5 w-3.5" />
            {stats.totalRequests} requests ({stats.errorCount} errors)
          </span>
        }
      >
        {recentRecords.length === 0 ? (
          <p className="py-3 text-xs text-[#475569]">
            No inference requests executed yet. Upload an image in Image Analysis or trigger an Example Gallery sample to begin recording real telemetry.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs tabular-nums">
              <thead>
                <tr className="border-b border-[#E5E7EB] text-[#475569]">
                  <th className="py-2 pr-3 font-medium">Time</th>
                  <th className="py-2 pr-3 font-medium">Endpoint</th>
                  <th className="py-2 pr-3 font-medium">Status</th>
                  <th className="py-2 pr-3 font-medium">Detections</th>
                  <th className="py-2 pr-3 font-medium">Server Infer</th>
                  <th className="py-2 font-medium">Client E2E</th>
                </tr>
              </thead>
              <tbody>
                {[...recentRecords]
                  .reverse()
                  .slice(0, 6)
                  .map((r) => (
                    <tr key={r.id} className="border-b border-[#F3F4F6]">
                      <td className="py-2 pr-3 font-mono text-[#475569]">
                        {r.timestampIso.slice(11, 19)}
                      </td>
                      <td className="py-2 pr-3 font-mono text-[#0F172A]">
                        {r.endpoint}
                      </td>
                      <td className="py-2 pr-3">
                        <StatusBadge tone={r.ok ? 'ok' : 'alert'}>
                          {r.statusCode}
                        </StatusBadge>
                      </td>
                      <td className="py-2 pr-3">{r.detectionCount}</td>
                      <td className="py-2 pr-3">{r.serverInferenceMs.toFixed(1)} ms</td>
                      <td className="py-2">{r.clientE2EMs.toFixed(1)} ms</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
