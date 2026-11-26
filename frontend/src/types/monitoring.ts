import type { ComplianceEventSchema } from './api';

export type PageId =
  | 'overview'
  | 'live'
  | 'image'
  | 'gallery'
  | 'alerts'
  | 'analytics'
  | 'monitoring'
  | 'settings';

export type AlertReviewStatus =
  | 'pending_review'
  | 'verified_potential_event'
  | 'dismissed_false_alarm';

export interface ReviewedComplianceEvent extends ComplianceEventSchema {
  id: string;
  sourcePage: 'image' | 'gallery' | 'live';
  streamId: string;
  recordedAtIso: string;
  reviewStatus: AlertReviewStatus;
}

export interface RequestTelemetryRecord {
  id: string;
  timestampIso: string;
  endpoint: string;
  sourcePage: 'image' | 'gallery' | 'live';
  ok: boolean;
  statusCode: number;
  detectionCount: number;
  personCount: number;
  helmetCount: number;
  vestCount: number;
  glovesCount: number;
  noHelmetCount: number;
  serverInferenceMs: number;
  serverTotalMs: number;
  clientE2EMs: number;
  errorMessage?: string;
}

export interface SessionStats {
  totalRequests: number;
  successfulRequests: number;
  errorCount: number;
  totalPersonsDetected: number;
  ppeCounts: {
    helmet: number;
    vest: number;
    gloves: number;
  };
  missingPpePredictions: number;
  potentialAlertsEmitted: number;
  lastServerInferenceMs: number | null;
  lastClientE2EMs: number | null;
  meanApiResponseMs: number | null;
  p95ApiResponseMs: number | null;
  liveFps: number | null;
}

export interface InferenceConfigState {
  confThreshold: number;
  iouThreshold: number;
  liveCaptureIntervalMs: number;
  enableComplianceOnLive: boolean;
}

export interface CuratedExample {
  id: string;
  title: string;
  sourceLabel: string;
  imagePath: string;
  fileName: string;
  notes: string;
}

export const CURATED_EXAMPLES: CuratedExample[] = [
  {
    id: 'ex-ul-1',
    title: 'Multi-worker site crew (Warehouse & Yard)',
    sourceLabel: 'Ultralytics PPE Val (CC BY 4.0)',
    imagePath: '/examples/ul_ultralytics_ppe__val__0c28da30faaa.jpg',
    fileName: 'ul_ultralytics_ppe__val__0c28da30faaa.jpg',
    notes: 'Dense crew scene testing small-object helmet, vest, gloves, and no_helmet.',
  },
  {
    id: 'ex-ul-2',
    title: 'Heavy equipment zone inspection',
    sourceLabel: 'Ultralytics PPE Val (CC BY 4.0)',
    imagePath: '/examples/ul_ultralytics_ppe__val__24f563c3cd69.jpg',
    fileName: 'ul_ultralytics_ppe__val__24f563c3cd69.jpg',
    notes: 'Evaluates mixed compliance cues across partially occluded workers.',
  },
  {
    id: 'ex-ul-3',
    title: 'Active scaffold & concrete assembly',
    sourceLabel: 'Ultralytics PPE Val (CC BY 4.0)',
    imagePath: '/examples/ul_ultralytics_ppe__val__28d1cd1b6f4a.jpg',
    fileName: 'ul_ultralytics_ppe__val__28d1cd1b6f4a.jpg',
    notes: 'Multi-PPE association across working posture variations.',
  },
  {
    id: 'ex-rf-1',
    title: 'Structural beam work (Outdoor daylight)',
    sourceLabel: 'Roboflow Safety v30 Valid (CC BY 4.0)',
    imagePath: '/examples/rf_roboflow_v30__valid__117b59e57e3e.jpg',
    fileName: 'rf_roboflow_v30__valid__117b59e57e3e.jpg',
    notes: 'Cross-domain generalization test case from Roboflow validation split.',
  },
  {
    id: 'ex-rf-2',
    title: 'Site walkthrough & supervision check',
    sourceLabel: 'Roboflow Safety v30 Valid (CC BY 4.0)',
    imagePath: '/examples/rf_roboflow_v30__valid__692cc1463b07.jpg',
    fileName: 'rf_roboflow_v30__valid__692cc1463b07.jpg',
    notes: 'Tests distinction between hardhat-wearing and uncovered individuals.',
  },
  {
    id: 'ex-rf-3',
    title: 'Elevated work & safety harness scene',
    sourceLabel: 'Roboflow Safety v30 Valid (CC BY 4.0)',
    imagePath: '/examples/rf_roboflow_v30__valid__0a74d41263b2.jpg',
    fileName: 'rf_roboflow_v30__valid__0a74d41263b2.jpg',
    notes: 'Verifies bounding-box localization under complex backgrounds.',
  },
];
