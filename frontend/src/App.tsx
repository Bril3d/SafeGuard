import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { AppShell } from './components/layout/AppShell';
import { AlertHistoryPage } from './components/pages/AlertHistoryPage';
import { ExampleGalleryPage } from './components/pages/ExampleGalleryPage';
import { ImageAnalysisPage } from './components/pages/ImageAnalysisPage';
import { LiveMonitoringPage } from './components/pages/LiveMonitoringPage';
import { ModelMonitoringPage } from './components/pages/ModelMonitoringPage';
import { OverviewPage } from './components/pages/OverviewPage';
import { SettingsPage } from './components/pages/SettingsPage';

const AnalyticsPage = lazy(() =>
  import('./components/pages/AnalyticsPage').then((mod) => ({
    default: mod.AnalyticsPage,
  })),
);
import { useSessionTelemetry } from './hooks/useSessionTelemetry';
import { fetchClasses, fetchHealth, fetchModelMetadata } from './services/api';
import type {
  ClassesResponse,
  HealthResponse,
  ModelMetadataResponse,
} from './types/api';
import type { InferenceConfigState, PageId } from './types/monitoring';

export function App() {
  const [activePage, setActivePage] = useState<PageId>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<ModelMetadataResponse | null>(null);
  const [classes, setClasses] = useState<ClassesResponse | null>(null);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [liveActive, setLiveActive] = useState(false);

  const [config, setConfig] = useState<InferenceConfigState>({
    confThreshold: 0.25,
    iouThreshold: 0.50,
    liveCaptureIntervalMs: 250,
    enableComplianceOnLive: true,
  });

  const {
    records,
    alerts,
    stats,
    setLiveFps,
    recordSuccess,
    recordError,
    updateAlertReview,
    resetSession,
  } = useSessionTelemetry();

  const loadBackendState = useCallback(async (signal?: AbortSignal) => {
    setHealthLoading(true);
    setHealthError(null);
    setMetadataError(null);

    const [hRes, mRes, cRes] = await Promise.allSettled([
      fetchHealth(signal),
      fetchModelMetadata(signal),
      fetchClasses(signal),
    ]);

    if (signal?.aborted) return;

    if (hRes.status === 'fulfilled') {
      setHealth(hRes.value);
    } else {
      setHealth(null);
      setHealthError(
        hRes.reason instanceof Error
          ? hRes.reason.message
          : 'Unable to reach SafeGuard API',
      );
    }

    if (mRes.status === 'fulfilled') {
      setMetadata(mRes.value);
    } else {
      setMetadata(null);
      setMetadataError(
        mRes.reason instanceof Error ? mRes.reason.message : 'Metadata unavailable',
      );
    }

    if (cRes.status === 'fulfilled') {
      setClasses(cRes.value);
    } else {
      setClasses(null);
    }

    setHealthLoading(false);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadBackendState(controller.signal);
    return () => controller.abort();
  }, [loadBackendState]);

  const pendingAlertCount = alerts.filter(
    (a) => a.reviewStatus === 'pending_review',
  ).length;

  return (
    <AppShell
      activePage={activePage}
      onNavigate={setActivePage}
      health={health}
      healthLoading={healthLoading}
      healthError={healthError}
      onRefreshHealth={() => void loadBackendState()}
      stats={stats}
      pendingAlertCount={pendingAlertCount}
      liveActive={liveActive}
    >
      {activePage === 'overview' ? (
        <OverviewPage
          health={health}
          metadata={metadata}
          stats={stats}
          alerts={alerts}
          recentRecords={records}
          config={config}
          onNavigate={setActivePage}
        />
      ) : null}

      {activePage === 'live' ? (
        <LiveMonitoringPage
          config={config}
          onLiveActiveChange={setLiveActive}
          onRecordSuccess={recordSuccess}
          onRecordError={recordError}
          onFpsChange={setLiveFps}
        />
      ) : null}

      {activePage === 'image' ? (
        <ImageAnalysisPage
          config={config}
          onRecordSuccess={recordSuccess}
          onRecordError={recordError}
        />
      ) : null}

      {activePage === 'gallery' ? (
        <ExampleGalleryPage
          config={config}
          onRecordSuccess={recordSuccess}
          onRecordError={recordError}
        />
      ) : null}

      {activePage === 'alerts' ? (
        <AlertHistoryPage alerts={alerts} onUpdateReview={updateAlertReview} />
      ) : null}

      {activePage === 'analytics' ? (
        <Suspense
          fallback={
            <div className="rounded border border-[#D1D5DB] bg-white p-6 text-xs text-[#475569]">
              Loading analytics charts...
            </div>
          }
        >
          <AnalyticsPage records={records} stats={stats} />
        </Suspense>
      ) : null}

      {activePage === 'monitoring' ? (
        <ModelMonitoringPage
          health={health}
          healthError={healthError}
          stats={stats}
          records={records}
        />
      ) : null}

      {activePage === 'settings' ? (
        <SettingsPage
          metadata={metadata}
          classes={classes}
          metadataError={metadataError}
          config={config}
          onConfigChange={setConfig}
          onResetSession={resetSession}
        />
      ) : null}
    </AppShell>
  );
}

export default App;
