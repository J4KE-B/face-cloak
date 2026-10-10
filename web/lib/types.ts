/**
 * TypeScript mirror of server/schema.py (the live-demo backend contract).
 * Keep names and shapes aligned with that file.
 */

export type JobState = "queued" | "running" | "done" | "error" | "expired";
export type Stage = "detecting" | "cloaking" | "swapping" | "judging";
export type Mode = "fast" | "robust";

export interface Health {
  status: "ok" | "loading";
  version: string;
  /** 'cuda' or 'cpu' */
  device: string;
  models_loaded: boolean;
  /** jobs queued + running */
  queue_length: number;
  default_steps: number;
  max_eps: number;
  /** glintr100 identity threshold at FAR=1% (null if the backend sent something unusable) */
  tau: number | null;
  /** largest accepted PGD step count on this server's device */
  max_steps: number;
  /** text from the backend; shown only as a labelled message, never as our own consent text */
  consent_notice: string | null;
}

export interface Target {
  id: string;
  name: string | null;
  /** validated data:image/(png|jpeg);base64,... URL (<=256 px) */
  thumbnail: string | null;
  /** validated https: URL of the original image, or null */
  source: string | null;
  license: string | null;
}

export interface JobCreated {
  job_id: string;
  /** 0 = running now / next, 1 = one job ahead, ... */
  queue_position: number;
}

export interface Progress {
  stage: Stage;
  /** PGD step (estimated from elapsed time while cloaking), else 0 */
  step: number;
  total: number;
}

export interface Scores {
  id_sim_original: number | null;
  /** null if the attacker could not even detect the cloaked face */
  id_sim_cloaked: number | null;
  tau: number | null;
  judge: string;
  /** 'uploaded' = separate reference photo, 'self' = original upload */
  reference: "uploaded" | "self";
}

export interface Quality {
  psnr: number | null;
  ssim: number | null;
  lpips: number | null;
  linf_255: number | null;
}

export interface Timings {
  cloak_s: number | null;
  total_s: number | null;
}

export interface Params {
  eps: number;
  steps: number;
  mode: Mode;
}

export interface JobResult {
  /** Validated PNG/JPEG data URL (null if the backend sent anything else), long side <= 1024 */
  original: string | null;
  cloaked: string | null;
  /** |cloaked-original| x10, clipped to [0,255] */
  perturbation: string | null;
  /** attacker's swap using the original photo (SYNTHETIC-labelled) */
  swap_original: string | null;
  /** null if the attacker failed to detect the face */
  swap_cloaked: string | null;
  /** full-resolution (as processed, <=1024 px) cloaked PNG data URL; null if missing/invalid */
  cloaked_download: string | null;
  scores: Scores;
  quality: Quality;
  /** id_sim_cloaked < tau and id_sim_original >= tau; null if the clean swap itself did not match */
  protected: boolean | null;
  note: string | null;
  timings: Timings;
  params: Params;
}

export interface JobStatus {
  job_id: string;
  status: JobState;
  queue_position: number | null;
  progress: Progress | null;
  error: string | null;
  result: JobResult | null;
}

/** Parameters of POST /api/jobs (multipart form). */
export interface JobParams {
  photo: Blob;
  reference?: Blob | null;
  target_id: string;
  eps: number;
  steps: number;
  mode: Mode;
}

/** Shape of web/public/results/summary.json, written by scripts/export_web_results.py. */
export interface ResultRow {
  attack: string;
  swapper: string;
  judge?: string;
  transform?: string;
  eps?: number;
  steps?: number;
  n?: number;
  n_valid?: number;
  psr?: number | null;
  psr_lo?: number | null;
  psr_hi?: number | null;
  psr_ci?: string;
  psr_failasprot?: number | null;
  mean_id_sim?: number | null;
  det_fail?: number | null;
  d_id_sim?: number | null;
  psnr?: number | null;
  ssim?: number | null;
  lpips?: number | null;
  cloak_seconds?: number | null;
  [k: string]: unknown;
}

export interface ResultsSummary {
  generated: string;
  configs: string[];
  whitebox_judges: string[];
  /** config -> judge -> threshold (a config may have no thresholds: null) */
  tau: Record<string, Record<string, number> | null>;
  /** judge -> threshold actually used for the PSR numbers */
  tau_used?: Record<string, number> | null;
  tau_source?: string | null;
  tables: Record<string, ResultRow[]>;
  figures: string[];
  n_swap_rows: number;
  note: string;
}
