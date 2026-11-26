import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn(
        'rounded-md border border-[#D1D5DB] bg-white',
        className,
      )}
    >
      {title ? (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E5E7EB] px-4 py-3">
          <div>
            <h2 className="text-sm font-semibold text-[#0F172A]">{title}</h2>
            {subtitle ? (
              <p className="mt-0.5 text-xs text-[#475569]">{subtitle}</p>
            ) : null}
          </div>
          {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
        </header>
      ) : null}
      <div className="p-4">{children}</div>
    </section>
  );
}

export type BadgeTone = 'neutral' | 'ok' | 'warn' | 'alert' | 'info';

const TONE_MAP: Record<BadgeTone, string> = {
  neutral: 'border-slate-300 bg-slate-100 text-slate-800',
  ok: 'border-emerald-300 bg-emerald-50 text-emerald-800',
  warn: 'border-amber-300 bg-amber-50 text-amber-900',
  alert: 'border-red-300 bg-red-50 text-red-800',
  info: 'border-sky-300 bg-sky-50 text-sky-900',
};

export function StatusBadge({
  tone = 'neutral',
  children,
  className,
}: {
  tone?: BadgeTone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-medium tabular-nums',
        TONE_MAP[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function MetricTile({
  label,
  value,
  detail,
  tone = 'neutral',
}: {
  label: string;
  value: string | number;
  detail?: string;
  tone?: BadgeTone;
}) {
  const accentBorder =
    tone === 'alert'
      ? 'border-l-red-600'
      : tone === 'warn'
        ? 'border-l-amber-600'
        : tone === 'ok'
          ? 'border-l-emerald-700'
          : tone === 'info'
            ? 'border-l-sky-700'
            : 'border-l-slate-500';

  return (
    <div
      className={cn(
        'rounded-md border border-[#D1D5DB] border-l-4 bg-white px-3.5 py-3',
        accentBorder,
      )}
    >
      <div className="text-xs font-medium text-[#475569]">{label}</div>
      <div
        className={cn(
          'mt-1 font-semibold tracking-tight text-[#0F172A] tabular-nums',
          typeof value === 'string' && value.length > 12 ? 'text-lg break-words' : 'text-2xl',
        )}
      >
        {value}
      </div>
      {detail ? <div className="mt-1 text-xs text-[#64748B] tabular-nums">{detail}</div> : null}
    </div>
  );
}
