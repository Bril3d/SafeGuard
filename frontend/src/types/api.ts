export interface HealthResponse {
  status: string;
  service: string;
  environment: string;
  model_loaded: boolean;
  model_version: string;
  gpu_available: boolean;
  runtime_provider: string;
}
