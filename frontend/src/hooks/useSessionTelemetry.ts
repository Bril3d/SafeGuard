import { useCallback, useMemo, useState } from 'react';
import type {
  ComplianceEventSchema,
  DetectionSchema,
  TimingBreakdownSchema,
} from '../types/api';
import type {
  AlertReviewStatus,
  RequestTelemetryRecord,
  ReviewedComplianceEvent,
  SessionStats,
} from '../types/monitoring';

const MAX_TELEMETRY_RECORDS = 120;
const MAX_ALERT_RECORDS = 200;

export function useSessionTelemetry() {
  const [records, setRecords] = useState<RequestTelemetryRecord[]>([]);
  const [alerts, setAlerts] = useState<ReviewedComplianceEvent[]>([]);
  const [liveFps, setLiveFps] = useState<number | null>(null);

  const recordSuccess = useCallback(
    (params: {
      endpoint: string;
      sourcePage: 'image' | 'gallery' | 'live';
      detections: DetectionSchema[];
      timings: TimingBreakdownSchema;
      clientE2EMs: number;
      events?: ComplianceEventSchema[];
      streamId?: string;
    }) => {
      let personCount = 0;
      let helmetCount = 0;
      let vestCount = 0;
      let glovesCount = 0;
      let noHelmetCount = 0;

      for (const det of params.detections) {
        if (det.class_name === 'person') personCount += 1;
        else if (det.class_name === 'helmet') helmetCount += 1;
        else if (det.class_name === 'vest') vestCount += 1;
        else if (det.class_name === 'gloves') glovesCount += 1;
        else if (det.class_name === 'no_helmet') noHelmetCount += 1;
      }

      const nowIso = new Date().toISOString();
      const rec: RequestTelemetryRecord = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestampIso: nowIso,
        endpoint: params.endpoint,
        sourcePage: params.sourcePage,
        ok: true,
        statusCode: 200,
        detectionCount: params.detections.length,
        personCount,
        helmetCount,
        vestCount,
        glovesCount,
        noHelmetCount,
        serverInferenceMs: params.timings.inference_ms,
        serverTotalMs: params.timings.total_ms,
        clientE2EMs: params.clientE2EMs,
      };

      setRecords((prev) => [...prev.slice(-(MAX_TELEMETRY_RECORDS - 1)), rec]);

      if (params.events && params.events.length > 0) {
        const mapped: ReviewedComplianceEvent[] = params.events.map((evt) => ({
          ...evt,
          id: `${evt.event_id}-${Date.now()}`,
          sourcePage: params.sourcePage,
          streamId: params.streamId ?? 'default',
          recordedAtIso: nowIso,
          reviewStatus: 'pending_review',
        }));
        setAlerts((prev) => [...mapped, ...prev].slice(0, MAX_ALERT_RECORDS));
      }
    },
    [],
  );

  const recordError = useCallback(
    (params: {
      endpoint: string;
      sourcePage: 'image' | 'gallery' | 'live';
      statusCode: number;
      clientE2EMs: number;
      message: string;
    }) => {
      const rec: RequestTelemetryRecord = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestampIso: new Date().toISOString(),
        endpoint: params.endpoint,
        sourcePage: params.sourcePage,
        ok: false,
        statusCode: params.statusCode,
        detectionCount: 0,
        personCount: 0,
        helmetCount: 0,
        vestCount: 0,
        glovesCount: 0,
        noHelmetCount: 0,
        serverInferenceMs: 0,
        serverTotalMs: 0,
        clientE2EMs: params.clientE2EMs,
        errorMessage: params.message,
      };
      setRecords((prev) => [...prev.slice(-(MAX_TELEMETRY_RECORDS - 1)), rec]);
    },
    [],
  );

  const updateAlertReview = useCallback(
    (alertId: string, reviewStatus: AlertReviewStatus) => {
      setAlerts((prev) =>
        prev.map((a) => (a.id === alertId ? { ...a, reviewStatus } : a)),
      );
    },
    [],
  );

  const resetSession = useCallback(() => {
    setRecords([]);
    setAlerts([]);
    setLiveFps(null);
  }, []);

  const stats: SessionStats = useMemo(() => {
    let successfulRequests = 0;
    let errorCount = 0;
    let totalPersonsDetected = 0;
    let helmet = 0;
    let vest = 0;
    let gloves = 0;
    let missingPpePredictions = 0;
    const responseTimes: number[] = [];
    let lastOk: RequestTelemetryRecord | null = null;

    for (const r of records) {
      if (r.ok) {
        successfulRequests += 1;
        totalPersonsDetected += r.personCount;
        helmet += r.helmetCount;
        vest += r.vestCount;
        gloves += r.glovesCount;
        missingPpePredictions += r.noHelmetCount;
        responseTimes.push(r.clientE2EMs);
        lastOk = r;
      } else {
        errorCount += 1;
      }
    }

    let meanApiResponseMs: number | null = null;
    let p95ApiResponseMs: number | null = null;
    if (responseTimes.length > 0) {
      const sum = responseTimes.reduce((acc, v) => acc + v, 0);
      meanApiResponseMs = Math.round((sum / responseTimes.length) * 10) / 10;
      const sorted = [...responseTimes].sort((a, b) => a - b);
      const p95Idx = Math.min(
        sorted.length - 1,
        Math.floor(0.95 * (sorted.length - 1)),
      );
      p95ApiResponseMs = Math.round(sorted[p95Idx] * 10) / 10;
    }

    return {
      totalRequests: records.length,
      successfulRequests,
      errorCount,
      totalPersonsDetected,
      ppeCounts: { helmet, vest, gloves },
      missingPpePredictions,
      potentialAlertsEmitted: alerts.length,
      lastServerInferenceMs: lastOk ? lastOk.serverInferenceMs : null,
      lastClientE2EMs: lastOk ? lastOk.clientE2EMs : null,
      meanApiResponseMs,
      p95ApiResponseMs,
      liveFps,
    };
  }, [alerts.length, liveFps, records]);

  return {
    records,
    alerts,
    stats,
    liveFps,
    setLiveFps,
    recordSuccess,
    recordError,
    updateAlertReview,
    resetSession,
  };
}
