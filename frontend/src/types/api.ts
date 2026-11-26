export interface ErrorDetail {
  code: string;
  message: string;
  request_id: string;
  details?: Array<Record<string, string>> | null;
}

export interface ErrorResponse {
  error: ErrorDetail;
}

export interface HealthResponse {
  status: string;
  service: string;
  environment: string;
  model_loaded: boolean;
  model_version: string;
  gpu_available: boolean;
  runtime_provider: string;
}

export interface TensorSpecSchema {
  name: string;
  shape: Array<number | string>;
  type: string;
}

export interface ClassItemSchema {
  id: number;
  name: string;
}

export interface ClassesResponse {
  model_version: string;
  classes: ClassItemSchema[];
}

export interface ModelMetadataResponse {
  model_name: string;
  model_version: string;
  architecture: string;
  opset: number | null;
  imgsz: number;
  default_conf: number;
  default_iou: number;
  onnx_sha256: string | null;
  mlflow_run_id: string | null;
  artifact_size_mb: number | null;
  runtime_provider: string;
  class_names: Record<string, string>;
  inputs: TensorSpecSchema[];
  outputs: TensorSpecSchema[];
}

export interface DetectionSchema {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
}

export interface TimingBreakdownSchema {
  decode_ms: number;
  preprocess_ms: number;
  inference_ms: number;
  postprocess_ms: number;
  total_ms: number;
}

export interface PredictResponse {
  request_id: string;
  model_version: string;
  image_width: number;
  image_height: number;
  conf_threshold: number;
  iou_threshold: number;
  detection_count: number;
  detections: DetectionSchema[];
  timings_ms: TimingBreakdownSchema;
}

export interface FramePredictResponse extends PredictResponse {
  frame_id: string | null;
}

export interface TrackSchema {
  track_id: number;
  bbox_xyxy: [number, number, number, number];
  confidence: number;
  age: number;
  hits: number;
  time_since_update: number;
  state: string;
}

export interface PersonAssociationSchema {
  track_id: number;
  person_confidence: number;
  bbox_xyxy: [number, number, number, number];
  helmet_boxes: DetectionSchema[];
  no_helmet_boxes: DetectionSchema[];
  vest_boxes: DetectionSchema[];
  gloves_boxes: DetectionSchema[];
  helmet_status: string;
  vest_status: string;
  overall_status: string;
  uncertainty_reasons: string[];
  violation_streak: number;
  visibility_status?: string;
}

export interface ComplianceEventSchema {
  event_id: string;
  event_type: string;
  severity: string;
  track_id: number;
  timestamp: number;
  frame_index: number;
  confirmed_frames: number;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
  requires_human_review: boolean;
  message: string;
  violation_category?: string;
  event_status?: string;
  evidence_bbox_xyxy?: [number, number, number, number] | null;
  evidence_frame_b64?: string | null;
}

export interface ComplianceSummarySchema {
  tracked_persons: number;
  compliant_count: number;
  potential_violation_count: number;
  pending_confirmation_count: number;
  unknown_count: number;
  emitted_event_count: number;
  suppressed_duplicates: number;
  uncertain_count?: number;
  insufficient_visibility_count?: number;
  lost_tracks_count?: number;
}

export interface ComplianceMonitorResponse {
  request_id: string;
  stream_id: string;
  frame_id: string | null;
  model_version: string;
  image_width: number;
  image_height: number;
  frame_index: number;
  timestamp: number;
  detections: DetectionSchema[];
  tracks: TrackSchema[];
  lost_tracks?: TrackSchema[];
  associations: PersonAssociationSchema[];
  unmatched_ppe: DetectionSchema[];
  events: ComplianceEventSchema[];
  summary: ComplianceSummarySchema;
  timings_ms: TimingBreakdownSchema;
  compliance_ms: number;
}

export interface ComplianceResetResponse {
  stream_id: string;
  reset: boolean;
  message: string;
}
