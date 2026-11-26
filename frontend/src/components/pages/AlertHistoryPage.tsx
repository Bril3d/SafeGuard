import { useState } from 'react';
import type { AlertReviewStatus, ReviewedComplianceEvent } from '../../types/monitoring';
import { Panel, StatusBadge } from '../ui/Primitives';

interface AlertHistoryPageProps {
  alerts: ReviewedComplianceEvent[];
  onUpdateReview: (alertId: string, status: AlertReviewStatus) => void;
}

export function AlertHistoryPage({
  alerts,
  onUpdateReview,
}: AlertHistoryPageProps) {
  const [filter, setFilter] = useState<'all' | AlertReviewStatus>('all');

  const visible =
    filter === 'all' ? alerts : alerts.filter((a) => a.reviewStatus === filter);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-[#0F172A]">
            Alert History & Human Review Queue
          </h1>
          <p className="mt-0.5 text-sm text-[#475569]">
            Potential missing-helmet events triggered strictly by explicit <code>no_helmet</code> confirmation.
          </p>
        </div>

        <div className="inline-flex rounded border border-[#D1D5DB] bg-white p-0.5 text-xs">
          {(
            [
              ['all', 'All'],
              ['pending_review', 'Pending Review'],
              ['verified_potential_event', 'Verified'],
              ['dismissed_false_alarm', 'Dismissed'],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setFilter(id)}
              className={`rounded px-2.5 py-1 font-medium ${
                filter === id
                  ? 'bg-[#0F172A] text-white'
                  : 'text-[#475569] hover:text-[#0F172A]'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <Panel title={`Potential Compliance Events (${visible.length})`}>
        {visible.length === 0 ? (
          <p className="py-6 text-xs text-[#64748B]">
            No potential alerts match this filter yet. Potential alerts appear here once <code>no_helmet</code> is confirmed across consecutive frames.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs tabular-nums">
              <thead>
                <tr className="border-b border-[#E5E7EB] text-[#475569]">
                  <th className="py-2 pr-3 font-medium">Recorded</th>
                  <th className="py-2 pr-3 font-medium">Track</th>
                  <th className="py-2 pr-3 font-medium">Type</th>
                  <th className="py-2 pr-3 font-medium">Frames / Conf</th>
                  <th className="py-2 pr-3 font-medium">Status</th>
                  <th className="py-2 font-medium">Review Action</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((evt) => (
                  <tr key={evt.id} className="border-b border-[#F3F4F6]">
                    <td className="py-2.5 pr-3 font-mono text-[#475569]">
                      {evt.recordedAtIso.slice(11, 19)}
                    </td>
                    <td className="py-2.5 pr-3 font-semibold">#{evt.track_id}</td>
                    <td className="py-2.5 pr-3">{evt.event_type}</td>
                    <td className="py-2.5 pr-3 font-mono">
                      {evt.confirmed_frames}f / {(evt.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 pr-3">
                      <StatusBadge
                        tone={
                          evt.reviewStatus === 'pending_review'
                            ? 'alert'
                            : evt.reviewStatus === 'verified_potential_event'
                              ? 'ok'
                              : 'neutral'
                        }
                      >
                        {evt.reviewStatus.replaceAll('_', ' ')}
                      </StatusBadge>
                    </td>
                    <td className="py-2.5">
                      <div className="flex flex-wrap gap-1.5">
                        <button
                          type="button"
                          onClick={() =>
                            onUpdateReview(evt.id, 'verified_potential_event')
                          }
                          className="rounded border border-emerald-700 px-2 py-0.5 font-medium text-emerald-800 hover:bg-emerald-50"
                        >
                          Verify
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            onUpdateReview(evt.id, 'dismissed_false_alarm')
                          }
                          className="rounded border border-slate-300 px-2 py-0.5 font-medium text-slate-700 hover:bg-slate-100"
                        >
                          Dismiss
                        </button>
                      </div>
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
