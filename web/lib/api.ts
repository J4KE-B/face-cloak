import { MOCK_URL } from "./backend";
import type { Health, JobCreated, JobParams, JobResult, JobState, JobStatus, Mode, Stage, Target } from "./types";
import * as mock from "./mock";
import { ApiError } from "./errors";

export { ApiError } from "./errors";
export type { ErrorKind } from "./errors";

export interface FriendlyError {
  title: string;
  message: string;
  retryAfter: number | null;
}

export function friendlyError(e: unknown, base?: string | null): FriendlyError {
  if (!(e instanceof ApiError)) {
    return { title: "Something went wrong", message: e instanceof Error ? e.message : "Unknown error.", retryAfter: null };
  }
  const d = e.detail;
  switch (e.kind) {
    case "network":
    case "timeout": {
      const mixed = typeof window !== "undefined" && window.location.protocol === "https:" && base?.startsWith("http://");
      if (mixed && e.kind === "network") {
        return {
          title: "Backend blocked by the browser",
          message:
            "This page is https but the backend is plain http://. Browsers may block that as mixed content, ask for a local-network permission (Chrome), or fail the CORS check, and Safari always blocks it. Use the https://...trycloudflare.com tunnel address instead (the laptop script prints one).",
          retryAfter: null,
        };
      }
      return {
        title: "Backend unreachable",
        message: "Backend unreachable. Is the Kaggle notebook (or your laptop script) still running? The tunnel address changes every time it restarts.",
        retryAfter: null,
      };
    }
    case "aborted":
      return { title: "Cancelled", message: "The request was cancelled.", retryAfter: null };
  }
  switch (e.status) {
    case 400:
      return { title: "Photo problem", message: d ?? "No usable face found. Use a clear, front-facing photo with one face.", retryAfter: null };
    case 401:
      return { title: "Wrong passcode", message: "The backend rejected the passcode. Check the DEMO_KEY you set in the notebook.", retryAfter: e.retryAfter };
    case 404:
      return { title: "Job expired", message: "This job expired or was already deleted. Results are removed from the backend after 10 minutes; start a new run.", retryAfter: null };
    case 413:
      return { title: "File too large", message: "That photo is too large for the backend (limit 5 MB, 24 megapixels).", retryAfter: null };
    case 415:
      return { title: "Unsupported file", message: "The backend could not read that file type. Use a JPEG or PNG photo.", retryAfter: null };
    case 422:
      return { title: "Invalid settings", message: d ?? "The backend rejected one of the settings. Lower the steps or epsilon and retry.", retryAfter: null };
    case 429:
      return {
        title: "Rate limit reached",
        message: d ?? "Too many requests from this connection. Please wait before trying again.",
        retryAfter: e.retryAfter,
      };
    case 503:
      return { title: "Backend busy", message: d ?? "The demo is busy right now. Please try again in a few minutes.", retryAfter: e.retryAfter ?? 60 };
    default:
      return { title: `Backend error (${e.status})`, message: d ?? "The backend returned an unexpected error.", retryAfter: e.retryAfter };
  }
}


/* ------------------------------------------------------------------------------------------ validation */
// The backend is untrusted input (anyone can run a tunnel). Every field is checked before it reaches the DOM.
const IMG_RE = /^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=]+$/;
const MAX_DATA_URL = 30_000_000;

function isObj(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}
export function safeImage(x: unknown): string | null {
  return typeof x === "string" && x.length < MAX_DATA_URL && IMG_RE.test(x) ? x : null;
}
function safeHttps(x: unknown): string | null {
  if (typeof x !== "string") return null;
  try {
    const u = new URL(x);
    return u.protocol === "https:" ? u.href : null;
  } catch {
    return null;
  }
}
function str(x: unknown, max = 500): string | null {
  return typeof x === "string" ? x.slice(0, max) : null;
}
function numOrNull(x: unknown): number | null {
  return typeof x === "number" && Number.isFinite(x) ? x : null;
}
function bool3(x: unknown): boolean | null {
  return typeof x === "boolean" ? x : null;
}
const STATES: JobState[] = ["queued", "running", "done", "error", "expired"];
const STAGES: Stage[] = ["detecting", "cloaking", "swapping", "judging"];

function bad(): never {
  throw new ApiError("http", 502, "Malformed response from backend", "The backend sent a response this page does not understand.");
}

export function parseHealth(j: unknown): Health {
  if (!isObj(j)) bad();
  return {
    status: j.status === "ok" ? "ok" : "loading",
    version: str(j.version, 60) ?? "",
    device: str(j.device, 20) ?? "cpu",
    models_loaded: j.models_loaded === true,
    queue_length: numOrNull(j.queue_length) ?? 0,
    default_steps: numOrNull(j.default_steps) ?? 40,
    max_eps: numOrNull(j.max_eps) ?? 16,
    tau: numOrNull(j.tau),
    max_steps: numOrNull(j.max_steps) ?? 60,
    consent_notice: str(j.consent_notice, 600),
  };
}

export function parseTargets(j: unknown): Target[] {
  if (!Array.isArray(j)) bad();
  const out: Target[] = [];
  for (const t of j.slice(0, 50)) {
    if (!isObj(t) || typeof t.id !== "string" || !/^[\w.-]{1,64}$/.test(t.id)) continue;
    const thumbnail = safeImage(t.thumbnail);
    if (!thumbnail) continue;
    out.push({ id: t.id, name: str(t.name, 120), thumbnail, source: safeHttps(t.source), license: str(t.license, 200) });
  }
  return out;
}

function parseResult(j: unknown): JobResult | null {
  if (!isObj(j)) return null;
  const sc = isObj(j.scores) ? j.scores : {};
  const q = isObj(j.quality) ? j.quality : {};
  const tm = isObj(j.timings) ? j.timings : {};
  const pr = isObj(j.params) ? j.params : {};
  const mode: Mode = pr.mode === "robust" ? "robust" : "fast";
  return {
    original: safeImage(j.original),
    cloaked: safeImage(j.cloaked),
    perturbation: safeImage(j.perturbation),
    swap_original: safeImage(j.swap_original),
    swap_cloaked: safeImage(j.swap_cloaked),
    cloaked_download: safeImage(j.cloaked_download),
    scores: {
      id_sim_original: numOrNull(sc.id_sim_original),
      id_sim_cloaked: numOrNull(sc.id_sim_cloaked),
      tau: numOrNull(sc.tau),
      judge: str(sc.judge, 40) ?? "glintr100",
      reference: sc.reference === "uploaded" ? "uploaded" : "self",
    },
    quality: { psnr: numOrNull(q.psnr), ssim: numOrNull(q.ssim), lpips: numOrNull(q.lpips), linf_255: numOrNull(q.linf_255) },
    protected: bool3(j.protected),
    note: str(j.note, 600),
    timings: { cloak_s: numOrNull(tm.cloak_s), total_s: numOrNull(tm.total_s) },
    params: { eps: numOrNull(pr.eps) ?? 0, steps: numOrNull(pr.steps) ?? 0, mode },
  };
}

export function parseJobCreated(j: unknown): JobCreated {
  if (!isObj(j) || typeof j.job_id !== "string" || !/^[\w.-]{1,128}$/.test(j.job_id)) bad();
  return { job_id: j.job_id, queue_position: numOrNull(j.queue_position) ?? 0 };
}

export function parseJobStatus(j: unknown): JobStatus {
  if (!isObj(j) || typeof j.job_id !== "string") bad();
  const status = STATES.find((x) => x === j.status);
  if (!status) bad();
  let progress: JobStatus["progress"] = null;
  if (isObj(j.progress)) {
    const stage = STAGES.find((x) => x === (j.progress as Record<string, unknown>).stage);
    if (stage) {
      const pg = j.progress as Record<string, unknown>;
      progress = { stage, step: numOrNull(pg.step) ?? 0, total: numOrNull(pg.total) ?? 0 };
    }
  }
  return {
    job_id: j.job_id,
    status,
    queue_position: numOrNull(j.queue_position),
    progress,
    error: str(j.error, 600),
    result: status === "done" ? parseResult(j.result) : null,
  };
}

/* ----------------------------------------------------------------------------------------------- http */
interface ReqOpts {
  key?: string;
  method?: "GET" | "POST" | "DELETE";
  body?: FormData;
  /** time allowed until response headers arrive */
  timeoutMs: number;
  /** time allowed for reading the body + JSON parse (done payloads carry several MB of base64) */
  bodyTimeoutMs?: number;
  signal?: AbortSignal;
}

/** Performs the request and, if `parse` is given, reads + validates the JSON body while the timeout is still armed. */
async function request<T = void>(base: string, path: string, o: ReqOpts, parse?: (j: unknown) => T): Promise<T> {
  const ctrl = new AbortController();
  let timedOut = false;
  let timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, o.timeoutMs);
  const onAbort = () => ctrl.abort();
  o.signal?.addEventListener("abort", onAbort);
  if (o.signal?.aborted) ctrl.abort();
  try {
    const headers: Record<string, string> = {};
    if (o.key) headers["X-Demo-Key"] = o.key;
    const res = await fetch(base + path, {
      method: o.method ?? "GET",
      headers,
      body: o.body,
      signal: ctrl.signal,
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
    });
    // Headers are in: re-arm the timer for the body read.
    clearTimeout(timer);
    timer = setTimeout(() => {
      timedOut = true;
      ctrl.abort();
    }, o.bodyTimeoutMs ?? Math.max(o.timeoutMs, 60_000));
    if (!res.ok) {
      let detail: string | null = null;
      try {
        const j = (await res.json()) as { detail?: unknown };
        if (typeof j.detail === "string") detail = j.detail.slice(0, 400);
      } catch {
        /* non-JSON error body */
      }
      const ra = Number(res.headers.get("Retry-After"));
      throw new ApiError("http", res.status, detail ?? `HTTP ${res.status}`, detail, Number.isFinite(ra) && ra > 0 ? ra : null);
    }
    if (!parse) return undefined as T;
    let j: unknown;
    try {
      j = await res.json();
    } catch (e) {
      if (ctrl.signal.aborted) throw e;
      bad();
    }
    return parse(j);
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if (timedOut) throw new ApiError("timeout", 0, "Request timed out");
    if (o.signal?.aborted) throw new ApiError("aborted", 0, "Aborted");
    throw new ApiError("network", 0, "Network error (backend down, blocked by CORS, or mixed content)");
  } finally {
    clearTimeout(timer);
    o.signal?.removeEventListener("abort", onAbort);
  }
}

export async function getHealth(base: string, signal?: AbortSignal): Promise<Health> {
  if (base === MOCK_URL) return mock.health();
  return request(base, "/health", { timeoutMs: 8000, bodyTimeoutMs: 15_000, signal }, parseHealth);
}

export async function getTargets(base: string, key: string, signal?: AbortSignal): Promise<Target[]> {
  if (base === MOCK_URL) return mock.targets(key);
  return request(base, "/api/targets", { key, timeoutMs: 15000, bodyTimeoutMs: 60_000, signal }, parseTargets);
}

export async function createJob(base: string, key: string, p: JobParams, signal?: AbortSignal): Promise<JobCreated> {
  if (base === MOCK_URL) return mock.createJob(key, p);
  const fd = new FormData();
  fd.append("photo", p.photo, "photo.jpg");
  if (p.reference) fd.append("reference", p.reference, "reference.jpg");
  fd.append("target_id", p.target_id);
  fd.append("eps", String(p.eps));
  fd.append("steps", String(p.steps));
  fd.append("mode", p.mode);
  return request(base, "/api/jobs", { key, method: "POST", body: fd, timeoutMs: 90_000, bodyTimeoutMs: 30_000, signal }, parseJobCreated);
}

export async function getJob(base: string, key: string, id: string, signal?: AbortSignal): Promise<JobStatus> {
  if (base === MOCK_URL) return mock.getJob(id);
  return request(base, `/api/jobs/${encodeURIComponent(id)}`, { key, timeoutMs: 20_000, bodyTimeoutMs: 90_000, signal }, parseJobStatus);
}

export async function deleteJob(base: string, key: string, id: string, signal?: AbortSignal): Promise<void> {
  if (base === MOCK_URL) return mock.deleteJob(id);
  await request(base, `/api/jobs/${encodeURIComponent(id)}`, { key, method: "DELETE", timeoutMs: 10_000, signal });
}
