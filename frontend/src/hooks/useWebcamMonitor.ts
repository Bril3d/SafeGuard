import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ApiClientError,
  predictComplianceFrame,
  predictFrame,
  resetComplianceStream,
} from '../services/api';
import type {
  ComplianceEventSchema,
  DetectionSchema,
  PersonAssociationSchema,
  TimingBreakdownSchema,
} from '../types/api';
import type { InferenceConfigState } from '../types/monitoring';

export interface LiveFrameState {
  detections: DetectionSchema[];
  associations: PersonAssociationSchema[];
  imageWidth: number;
  imageHeight: number;
  serverInferenceMs: number;
  serverTotalMs: number;
  clientE2EMs: number;
  droppedFrames: number;
  processedFrames: number;
  fps: number;
  suppressedDuplicates: number;
}

interface UseWebcamMonitorOptions {
  config: InferenceConfigState;
  onSuccess: (params: {
    endpoint: string;
    sourcePage: 'live';
    detections: DetectionSchema[];
    timings: TimingBreakdownSchema;
    clientE2EMs: number;
    events?: ComplianceEventSchema[];
    streamId?: string;
  }) => void;
  onError: (params: {
    endpoint: string;
    sourcePage: 'live';
    statusCode: number;
    clientE2EMs: number;
    message: string;
  }) => void;
  onFpsChange: (fps: number | null) => void;
}

export function useWebcamMonitor({
  config,
  onSuccess,
  onError,
  onFpsChange,
}: UseWebcamMonitorOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const droppedRef = useRef(0);
  const processedRef = useRef(0);
  const timestampsRef = useRef<number[]>([]);
  const frameCounterRef = useRef(0);

  const [status, setStatus] = useState<'idle' | 'starting' | 'running' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [liveState, setLiveState] = useState<LiveFrameState | null>(null);

  const stop = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    abortRef.current?.abort();
    abortRef.current = null;
    inFlightRef.current = false;

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus('idle');
    onFpsChange(null);
  }, [onFpsChange]);

  const sendSingleFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) {
      return;
    }

    // Strict concurrency cap: drop stale frames whenever an inference request is already in flight.
    if (inFlightRef.current) {
      droppedRef.current += 1;
      return;
    }

    let canvas = captureCanvasRef.current;
    if (!canvas) {
      canvas = document.createElement('canvas');
      captureCanvasRef.current = canvas;
    }

    // Preserve aspect ratio while capping capture long edge at 640px for fast transport.
    const srcW = video.videoWidth;
    const srcH = video.videoHeight;
    const scale = Math.min(1, 640 / Math.max(srcW, srcH));
    const capW = Math.max(1, Math.round(srcW * scale));
    const capH = Math.max(1, Math.round(srcH * scale));
    canvas.width = capW;
    canvas.height = capH;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, capW, capH);

    inFlightRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;

    const frameNum = ++frameCounterRef.current;
    const frameId = `webcam-${frameNum}`;
    const tReqStart = performance.now();

    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas!.toBlob(resolve, 'image/jpeg', 0.82),
      );
      if (!blob) {
        return;
      }

      let detections: DetectionSchema[] = [];
      let associations: PersonAssociationSchema[] = [];
      let timings: TimingBreakdownSchema;
      let clientE2EMs = 0;
      let events: ComplianceEventSchema[] | undefined;
      let suppressed = 0;
      let endpoint = '/predict/compliance';

      if (config.enableComplianceOnLive) {
        const res = await predictComplianceFrame(blob, {
          streamId: 'webcam-live',
          frameId,
          conf: config.confThreshold,
          iou: config.iouThreshold,
          timestamp: Date.now() / 1000,
          signal: controller.signal,
        });
        detections = res.data.detections;
        associations = res.data.associations;
        timings = res.data.timings_ms;
        clientE2EMs = res.clientE2EMs;
        events = res.data.events;
        suppressed = res.data.summary.suppressed_duplicates;
      } else {
        endpoint = '/predict/frame';
        const res = await predictFrame(blob, {
          frameId,
          conf: config.confThreshold,
          iou: config.iouThreshold,
          signal: controller.signal,
        });
        detections = res.data.detections;
        timings = res.data.timings_ms;
        clientE2EMs = res.clientE2EMs;
      }

      processedRef.current += 1;
      const now = performance.now();
      const recent = [...timestampsRef.current.filter((t) => now - t <= 3000), now];
      timestampsRef.current = recent;
      const fps =
        recent.length >= 2
          ? Math.round(
              ((recent.length - 1) / ((recent[recent.length - 1] - recent[0]) / 1000)) *
                10,
            ) / 10
          : 0;

      onFpsChange(fps);
      setLiveState({
        detections,
        associations,
        imageWidth: capW,
        imageHeight: capH,
        serverInferenceMs: timings.inference_ms,
        serverTotalMs: timings.total_ms,
        clientE2EMs,
        droppedFrames: droppedRef.current,
        processedFrames: processedRef.current,
        fps,
        suppressedDuplicates: suppressed,
      });

      onSuccess({
        endpoint,
        sourcePage: 'live',
        detections,
        timings,
        clientE2EMs,
        events,
        streamId: 'webcam-live',
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return;
      }
      const elapsed = Math.round((performance.now() - tReqStart) * 10) / 10;
      const statusCode = err instanceof ApiClientError ? err.status : 0;
      const message =
        err instanceof Error ? err.message : 'Live frame inference failed.';
      setErrorMsg(message);
      onError({
        endpoint: config.enableComplianceOnLive ? '/predict/compliance' : '/predict/frame',
        sourcePage: 'live',
        statusCode,
        clientE2EMs: elapsed,
        message,
      });
    } finally {
      inFlightRef.current = false;
    }
  }, [config.confThreshold, config.enableComplianceOnLive, config.iouThreshold, onError, onFpsChange, onSuccess]);

  const start = useCallback(async () => {
    setErrorMsg(null);
    setStatus('starting');
    droppedRef.current = 0;
    processedRef.current = 0;
    timestampsRef.current = [];

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('error');
      setErrorMsg('Browser camera access (getUserMedia) is not supported in this context.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus('running');
      timerRef.current = window.setInterval(
        () => void sendSingleFrame(),
        config.liveCaptureIntervalMs,
      );
    } catch (err) {
      setStatus('error');
      const message =
        err instanceof Error
          ? `Camera permission denied or unavailable: ${err.message}`
          : 'Unable to access webcam.';
      setErrorMsg(message);
    }
  }, [config.liveCaptureIntervalMs, sendSingleFrame]);

  const resetStreamState = useCallback(async () => {
    try {
      await resetComplianceStream('webcam-live');
    } catch {
      // Non-fatal if backend is disconnected.
    }
  }, []);

  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    videoRef,
    status,
    errorMsg,
    liveState,
    start,
    stop,
    resetStreamState,
  };
}
