import {
  Activity,
  AlertTriangle,
  BarChart3,
  Camera,
  Image as ImageIcon,
  Images,
  LayoutDashboard,
  RefreshCw,
  Settings,
  ShieldCheck,
} from 'lucide-react';
import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';
import type { HealthResponse } from '../../types/api';
import type { PageId, SessionStats } from '../../types/monitoring';
import { StatusBadge } from '../ui/Primitives';

interface NavItem {
  id: PageId;
  label: string;
  icon: typeof LayoutDashboard;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'live', label: 'Live Monitoring', icon: Camera },
  { id: 'image', label: 'Image Analysis', icon: ImageIcon },
  { id: 'gallery', label: 'Example Gallery', icon: Images },
  { id: 'alerts', label: 'Alert History', icon: AlertTriangle },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'monitoring', label: 'Model Monitoring', icon: Activity },
  { id: 'settings', label: 'Settings & Model', icon: Settings },
];

interface AppShellProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  onRefreshHealth: () => void;
  stats: SessionStats;
  pendingAlertCount: number;
  liveActive: boolean;
  children: ReactNode;
}

export function AppShell({
  activePage,
  onNavigate,
  health,
  healthLoading,
  healthError,
  onRefreshHealth,
  stats,
  pendingAlertCount,
  liveActive,
  children,
}: AppShellProps) {
  const modelReady = Boolean(health?.model_loaded);

  return (
    <div className="min-h-screen bg-[#F3F4F6] text-[#0F172A] lg:flex">
      <aside className="flex w-full shrink-0 flex-col border-b border-[#1F2937] bg-[#111827] text-slate-100 lg:min-h-screen lg:w-60 lg:border-r lg:border-b-0">
        <div className="flex items-center justify-between border-b border-[#1F2937] px-4 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded border border-emerald-500/40 bg-emerald-950/60 text-emerald-400">
              <ShieldCheck className="h-4 w-4" />
            </div>
            <div>
              <div className="text-sm font-semibold tracking-tight text-white">
                SafeGuard
              </div>
              <div className="text-[11px] text-slate-400">PPE Vision Console</div>
            </div>
          </div>
          {liveActive ? (
            <span className="inline-flex items-center gap-1 rounded border border-emerald-500/40 bg-emerald-950/60 px-1.5 py-0.5 text-[11px] font-medium text-emerald-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Live
            </span>
          ) : null}
        </div>

        <nav className="flex gap-1 overflow-x-auto px-2 py-3 lg:flex-1 lg:flex-col" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activePage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                className={cn(
                  'flex shrink-0 items-center justify-between gap-2 rounded px-3 py-2 text-left text-xs font-medium transition-colors',
                  active
                    ? 'bg-[#1F2937] text-white'
                    : 'text-slate-300 hover:bg-[#1F2937]/60 hover:text-white',
                )}
              >
                <span className="flex items-center gap-2.5">
                  <Icon className="h-4 w-4 shrink-0 text-slate-400" />
                  <span>{item.label}</span>
                </span>
                {item.id === 'alerts' && pendingAlertCount > 0 ? (
                  <span className="rounded bg-red-600 px-1.5 py-0.5 text-[11px] font-semibold text-white tabular-nums">
                    {pendingAlertCount}
                  </span>
                ) : null}
              </button>
            );
          })}
        </nav>

        <div className="hidden border-t border-[#1F2937] p-3 text-[11px] text-slate-400 lg:block">
          <div className="font-medium text-slate-300">Decision-support notice</div>
          <p className="mt-1 leading-relaxed">
            Alerts flag potential missing-helmet events for human verification only. No biometric identification or footage retention.
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[#D1D5DB] bg-white px-5 py-3">
          <div className="flex flex-wrap items-center gap-2">
            {healthError ? (
              <StatusBadge tone="alert">Backend unreachable</StatusBadge>
            ) : modelReady ? (
              <StatusBadge tone="ok">ONNX Model Ready</StatusBadge>
            ) : (
              <StatusBadge tone="warn">Model Unavailable (503)</StatusBadge>
            )}
            <StatusBadge tone="neutral">
              Version: {health?.model_version ?? 'unknown'}
            </StatusBadge>
            <StatusBadge tone="neutral">
              Provider: {health?.runtime_provider ?? 'uninitialized'}
            </StatusBadge>
            {stats.lastServerInferenceMs !== null ? (
              <StatusBadge tone="info">
                Infer: {stats.lastServerInferenceMs.toFixed(1)} ms
              </StatusBadge>
            ) : null}
            {stats.liveFps !== null ? (
              <StatusBadge tone="info">
                Live: {stats.liveFps.toFixed(1)} FPS
              </StatusBadge>
            ) : null}
          </div>

          <button
            type="button"
            onClick={onRefreshHealth}
            disabled={healthLoading}
            className="inline-flex items-center gap-1.5 rounded border border-[#D1D5DB] bg-white px-2.5 py-1.5 text-xs font-medium text-[#0F172A] hover:bg-slate-50 disabled:opacity-50"
          >
            <RefreshCw className={cn('h-3.5 w-3.5', healthLoading ? 'animate-spin' : '')} />
            Refresh Status
          </button>
        </header>

        <main className="flex-1 p-5">{children}</main>
      </div>
    </div>
  );
}
