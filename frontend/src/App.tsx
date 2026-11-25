import { useCallback, useEffect, useState } from 'react';
import { fetchHealth, getApiBaseUrl } from './services/api';
import type { HealthResponse } from './types/api';

type HealthState =
  | { status: 'loading' }
  | { status: 'ready'; data: HealthResponse }
  | { status: 'error'; message: string };

export function App() {
  const [health, setHealth] = useState<HealthState>({ status: 'loading' });

  const loadHealth = useCallback(async (signal?: AbortSignal) => {
    setHealth({ status: 'loading' });
    try {
      const data = await fetchHealth(signal);
      setHealth({ status: 'ready', data });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return;
      }
      const message =
        err instanceof Error ? err.message : 'Unable to reach SafeGuard API';
      setHealth({ status: 'error', message });
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadHealth(controller.signal);
    return () => controller.abort();
  }, [loadHealth]);

  return (
    <main className="container">
      <div className="card">
        <h1>SafeGuard</h1>
        <p>
          Real-Time Construction PPE Monitoring — Repository Foundation Shell
          (Axis 3)
        </p>

        <p>
          API Endpoint: <code>{getApiBaseUrl()}/health</code>
        </p>

        {health.status === 'loading' && (
          <span className="badge badge-warn">Checking backend health...</span>
        )}

        {health.status === 'error' && (
          <div>
            <span className="badge badge-err">Backend Unreachable</span>
            <p>{health.message}</p>
          </div>
        )}

        {health.status === 'ready' && (
          <div>
            <span className="badge badge-ok">
              API Status: {health.data.status}
            </span>
            <div className="grid">
              <div className="kv">
                <div className="kv-label">Service</div>
                <div className="kv-value">{health.data.service}</div>
              </div>
              <div className="kv">
                <div className="kv-label">Environment</div>
                <div className="kv-value">{health.data.environment}</div>
              </div>
              <div className="kv">
                <div className="kv-label">Model Loaded</div>
                <div className="kv-value">
                  {health.data.model_loaded ? 'Yes' : 'No (Foundation Phase)'}
                </div>
              </div>
              <div className="kv">
                <div className="kv-label">Runtime Provider</div>
                <div className="kv-value">{health.data.runtime_provider}</div>
              </div>
            </div>
          </div>
        )}

        <button
          type="button"
          onClick={() => void loadHealth()}
          disabled={health.status === 'loading'}
        >
          Refresh Health Check
        </button>
      </div>
    </main>
  );
}

export default App;
