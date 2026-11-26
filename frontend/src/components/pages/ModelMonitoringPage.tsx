import type { HealthResponse } from '../../types/api';
import type { RequestTelemetryRecord, SessionStats } from '../../types/monitoring';
import { MetricTile, Panel, StatusBadge } from '../ui/Primitives';

interface ModelMonitoringPageProps {
  health: HealthResponse | null;
  healthError: string | null;
  stats: SessionStats;
  records: RequestTelemetryRecord[];
}

export function ModelMonitoringPage({
  health,
  healthError,
  stats,
  records,
}: ModelMonitoringPageProps) {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-[#0F172A]">
          Model & Operational Monitoring
        </h1>
        <p className="mt-0.5 text-sm text-[#475569]">
          Tracks ONNX model availability, latency, FPS, and API error rate without ground-truth accuracy claims.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricTile
          label="Model Availability"
          value={healthError ? 'Unreachable' : health?.model_loaded ? 'Ready' : 'Unavailable'}
          detail={health?.runtime_provider ?? 'uninitialized'}
          tone={health?.model_loaded ? 'ok' : 'alert'}
        />
        <MetricTile
          label="Loaded Model Version"
          value={health?.model_version ?? 'unknown'}
          detail={`GPU EP active: ${health?.gpu_available ? 'Yes' : 'No'}`}
        />
        <MetricTile
          label="Mean / P95 API Latency"
          value={
            stats.meanApiResponseMs !== null
              ? `${stats.meanApiResponseMs} ms`
              : '-'
          }
          detail={
            stats.p95ApiResponseMs !== null
              ? `P95: ${stats.p95ApiResponseMs} ms`
              : 'Awaiting requests'
          }
          tone="info"
        />
        <MetricTile
          label="Last Server Inference"
          value={
            stats.lastServerInferenceMs !== null
              ? `${stats.lastServerInferenceMs.toFixed(1)} ms`
              : '-'
          }
          detail={
            stats.liveFps !== null ? `Live FPS: ${stats.liveFps.toFixed(1)}` : 'Live stream idle'
          }
        />
        <MetricTile
          label="API Error Count"
          value={stats.errorCount}
          detail={`Out of ${stats.totalRequests} total requests`}
          tone={stats.errorCount > 0 ? 'alert' : 'ok'}
        />
      </div>

      <Panel
        title="Operational Telemetry Log"
        subtitle="Full audit of FastAPI latency measurements and HTTP failures"
      >
        {records.length === 0 ? (
          <p className="py-6 text-xs text-[#64748B]">
            No API requests recorded in this session yet.
          </p>
        ) : (
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full text-left text-xs tabular-nums">
              <thead className="sticky top-0 bg-[#F9FAFB] text-[#475569]">
                <tr className="border-b border-[#E5E7EB]">
                  <th className="py-2 pr-3 font-medium">Time</th>
                  <th className="py-2 pr-3 font-medium">Page</th>
                  <th className="py-2 pr-3 font-medium">Endpoint</th>
                  <th className="py-2 pr-3 font-medium">HTTP</th>
                  <th className="py-2 pr-3 font-medium">Server Infer</th>
                  <th className="py-2 pr-3 font-medium">Server Total</th>
                  <th className="py-2 pr-3 font-medium">Client E2E</th>
                  <th className="py-2 font-medium">Notes</th>
                </tr>
              </thead>
              <tbody>
                {[...records].reverse().map((r) => (
                  <tr key={r.id} className="border-b border-[#F3F4F6]">
                    <td className="py-2 pr-3 font-mono text-[#475569]">
                      {r.timestampIso.slice(11, 19)}
                    </td>
                    <td className="py-2 pr-3">{r.sourcePage}</td>
                    <td className="py-2 pr-3 font-mono">{r.endpoint}</td>
                    <td className="py-2 pr-3">
                      <StatusBadge tone={r.ok ? 'ok' : 'alert'}>
                        {r.statusCode}
                      </StatusBadge>
                    </td>
                    <td className="py-2 pr-3">{r.serverInferenceMs.toFixed(1)} ms</td>
                    <td className="py-2 pr-3">{r.serverTotalMs.toFixed(1)} ms</td>
                    <td className="py-2 pr-3">{r.clientE2EMs.toFixed(1)} ms</td>
                    <td className="py-2 text-[#475569]">
                      {r.errorMessage ?? `${r.detectionCount} detections`}
                    </td>
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
