import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { RequestTelemetryRecord, SessionStats } from '../../types/monitoring';
import { MetricTile, Panel } from '../ui/Primitives';

interface AnalyticsPageProps {
  records: RequestTelemetryRecord[];
  stats: SessionStats;
}

export function AnalyticsPage({ records, stats }: AnalyticsPageProps) {
  const classChartData = [
    { name: 'Person', count: stats.totalPersonsDetected },
    { name: 'Helmet', count: stats.ppeCounts.helmet },
    { name: 'Safety vest', count: stats.ppeCounts.vest },
    { name: 'Gloves', count: stats.ppeCounts.gloves },
    { name: 'No helmet', count: stats.missingPpePredictions },
  ];

  const latencySeries = records
    .filter((r) => r.ok)
    .slice(-30)
    .map((r, idx) => ({
      req: `#${idx + 1}`,
      serverInferMs: r.serverInferenceMs,
      clientE2EMs: r.clientE2EMs,
    }));

  const positiveTotal =
    stats.ppeCounts.helmet + stats.ppeCounts.vest + stats.ppeCounts.gloves;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-[#0F172A]">
          Session Analytics
        </h1>
        <p className="mt-0.5 text-sm text-[#475569]">
          Observed detection distributions and latency behavior accumulated from real FastAPI responses.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Total Persons Detected"
          value={stats.totalPersonsDetected}
          tone="info"
        />
        <MetricTile
          label="Positive PPE Detections"
          value={positiveTotal}
          tone="ok"
        />
        <MetricTile
          label="Explicit no_helmet"
          value={stats.missingPpePredictions}
          tone={stats.missingPpePredictions > 0 ? 'warn' : 'neutral'}
        />
        <MetricTile
          label="Session Requests"
          value={stats.successfulRequests}
          detail={`${stats.errorCount} failed calls`}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel
          title="Observed Detections by Class"
          subtitle="Cumulative counts across uploaded images, examples, and webcam frames"
        >
          {stats.successfulRequests === 0 ? (
            <p className="py-12 text-center text-xs text-[#64748B]">
              No successful detections recorded in this session yet.
            </p>
          ) : (
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={classChartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#475569' }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 12, fill: '#475569' }} />
                  <Tooltip />
                  <Bar
                    dataKey="count"
                    name="Detections"
                    fill="#0F172A"
                    radius={[3, 3, 0, 0]}
                    isAnimationActive={false}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>

        <Panel
          title="Server Inference vs. Client E2E Latency"
          subtitle="Last 30 successful inference requests (ms)"
        >
          {latencySeries.length === 0 ? (
            <p className="py-12 text-center text-xs text-[#64748B]">
              Execute one or more inference calls to inspect real response-time telemetry.
            </p>
          ) : (
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={latencySeries} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis dataKey="req" tick={{ fontSize: 11, fill: '#475569' }} />
                  <YAxis tick={{ fontSize: 12, fill: '#475569' }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="serverInferMs"
                    name="Server Inference (ms)"
                    stroke="#15803D"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="clientE2EMs"
                    name="Client E2E (ms)"
                    stroke="#0369A1"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
