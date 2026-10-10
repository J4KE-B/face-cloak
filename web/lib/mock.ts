/**
 * In-browser mock of the backend contract (server/schema.py), used when the backend URL is "mock".
 * No network is involved: images are generated on <canvas>. Handy for UI development and screenshots.
 * Passcode "wrong" simulates a 401, passcode "busy" a 503, passcode "limit" a 429 (Retry-After: 20).
 */
import { ApiError } from "./errors";
import type { Health, JobCreated, JobParams, JobResult, JobStatus, Stage, Target } from "./types";

const LICENSE = "Mock placeholder (generated in your browser)";

const HEALTH: Health = {
  status: "ok",
  version: "mock-1.0",
  device: "cuda",
  models_loaded: true,
  queue_length: 0,
  default_steps: 40,
  max_eps: 16,
  tau: 0.28,
  max_steps: 200,
  consent_notice:
    "Only upload photos of yourself or people who consented. Images are processed in memory and deleted within 10 minutes.",
};

export async function health(): Promise<Health> {
  await sleep(120);
  return { ...HEALTH, queue_length: jobs.size > 0 ? 1 : 0 };
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function checkKey(key: string) {
  if (key === "wrong" || !key) throw new ApiError("http", 401, "Invalid passcode", "Invalid or missing passcode.");
}

function canvas(w: number, h: number) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("canvas unavailable");
  return { c, ctx };
}

function portrait(label: string, hue: number, size: number): string {
  const { c, ctx } = canvas(size, size);
  const g = ctx.createLinearGradient(0, 0, size, size);
  g.addColorStop(0, `hsl(${hue} 60% 22%)`);
  g.addColorStop(1, `hsl(${(hue + 50) % 360} 70% 38%)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = `hsl(${hue} 30% 80%)`;
  ctx.beginPath();
  ctx.arc(size / 2, size * 0.42, size * 0.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(size / 2, size * 0.98, size * 0.34, size * 0.34, 0, Math.PI, 0);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,.85)";
  ctx.font = `600 ${Math.round(size / 11)}px sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText(label, size / 2, size * 0.1);
  return c.toDataURL("image/jpeg", 0.8);
}

export async function targets(key: string): Promise<Target[]> {
  await sleep(150);
  checkKey(key);
  return ["Mock A", "Mock B", "Mock C", "Mock D", "Mock E", "Mock F"].map((n, i) => ({
    id: `mock-${i + 1}`,
    name: `${n} (placeholder)`,
    thumbnail: portrait(n, (i * 57 + 200) % 360, 160),
    source: "https://github.com/J4KE-B/face-cloak",
    license: LICENSE,
  }));
}

// ---------------------------------------------------------------------------------------------- jobs
interface MockJob {
  started: number;
  total: number;
  steps: number;
  result: Promise<JobResult>;
  resolved: JobResult | null;
  failed: string | null;
}
const jobs = new Map<string, MockJob>();
let counter = 0;

async function loadBitmap(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob);
}

function mulberry(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function band(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = "rgba(0,0,0,.55)";
  ctx.fillRect(0, h - 34, w, 34);
  ctx.fillStyle = "rgba(255,255,255,.9)";
  ctx.font = "600 15px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("SYNTHETIC · face-cloak demo (mock)", w / 2, h - 12);
}

async function buildResult(p: JobParams): Promise<JobResult> {
  const bmp = await loadBitmap(p.photo);
  const scale = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * scale);
  const h = Math.round(bmp.height * scale);
  const o = canvas(w, h);
  o.ctx.drawImage(bmp, 0, 0, w, h);
  const orig = o.ctx.getImageData(0, 0, w, h);

  const cl = canvas(w, h);
  const pe = canvas(w, h);
  const cd = new ImageData(new Uint8ClampedArray(orig.data), w, h);
  const pd = new ImageData(w, h);
  const rnd = mulberry(1234 + p.eps);
  const amp = Math.min(p.eps, 8);
  for (let i = 0; i < orig.data.length; i += 4) {
    for (let k = 0; k < 3; k++) {
      const d = Math.round((rnd() * 2 - 1) * amp);
      const v = Math.max(0, Math.min(255, orig.data[i + k]! + d));
      cd.data[i + k] = v;
      pd.data[i + k] = Math.min(255, Math.abs(v - orig.data[i + k]!) * 10);
    }
    cd.data[i + 3] = 255;
    pd.data[i + 3] = 255;
  }
  cl.ctx.putImageData(cd, 0, 0);
  pe.ctx.putImageData(pd, 0, 0);

  const sw = 512;
  const sh = Math.round((512 * h) / w);
  const s1 = canvas(sw, sh);
  s1.ctx.drawImage(bmp, 0, 0, sw, sh);
  s1.ctx.fillStyle = "rgba(40,120,255,.12)";
  s1.ctx.fillRect(0, 0, sw, sh);
  band(s1.ctx, sw, sh);
  const s2 = canvas(sw, sh);
  try {
    s2.ctx.filter = "hue-rotate(150deg) saturate(1.6) blur(4px)";
  } catch {
    /* filter unsupported */
  }
  s2.ctx.drawImage(bmp, 0, 0, sw, sh);
  s2.ctx.filter = "none";
  const r2 = mulberry(7);
  for (let i = 0; i < 90; i++) {
    s2.ctx.fillStyle = `hsla(${r2() * 360},80%,60%,.35)`;
    s2.ctx.fillRect(r2() * sw, r2() * sh, 14 + r2() * 40, 4 + r2() * 14);
  }
  band(s2.ctx, sw, sh);
  bmp.close();

  const png = o.c.toDataURL("image/png");
  const cloaked = cl.c.toDataURL("image/png");
  const eps = p.eps;
  return {
    original: png,
    cloaked,
    perturbation: pe.c.toDataURL("image/png"),
    swap_original: s1.c.toDataURL("image/png"),
    swap_cloaked: s2.c.toDataURL("image/png"),
    cloaked_download: cloaked,
    scores: {
      id_sim_original: 0.61,
      id_sim_cloaked: 0.09,
      tau: HEALTH.tau,
      judge: "glintr100",
      reference: p.reference ? "uploaded" : "self",
    },
    quality: { psnr: 40.1 - eps * 0.35, ssim: 0.985 - eps * 0.003, lpips: 0.012 + eps * 0.004, linf_255: eps },
    protected: true,
    note: "MOCK DATA: these are canned numbers and generated images; no model ran. Switch the backend URL to a real tunnel for real results.",
    timings: { cloak_s: 4.2, total_s: 9.1 },
    params: { eps, steps: p.steps, mode: p.mode },
  };
}

export async function createJob(key: string, p: JobParams): Promise<JobCreated> {
  await sleep(250);
  checkKey(key);
  if (key === "busy") throw new ApiError("http", 503, "busy", "The demo is busy right now. Please try again in a few minutes.", 60);
  if (key === "limit") throw new ApiError("http", 429, "limit", "Rate limit: 5 jobs per hour per connection.", 20);
  const id = `mock-job-${++counter}`;
  const job: MockJob = {
    started: Date.now(),
    total: 9000,
    steps: p.steps,
    result: buildResult(p),
    resolved: null,
    failed: null,
  };
  job.result.then(
    (r) => (job.resolved = r),
    (e: unknown) => (job.failed = e instanceof Error ? e.message : "mock failure"),
  );
  jobs.set(id, job);
  return { job_id: id, queue_position: 1 };
}

export async function getJob(id: string): Promise<JobStatus> {
  await sleep(80);
  const job = jobs.get(id);
  if (!job) throw new ApiError("http", 404, "Unknown job.", "Unknown job.");
  if (job.failed) return { job_id: id, status: "error", queue_position: null, progress: null, error: job.failed, result: null };
  const t = Date.now() - job.started;
  if (t < 1500) return { job_id: id, status: "queued", queue_position: t < 700 ? 1 : 0, progress: null, error: null, result: null };
  if (t >= job.total && job.resolved) {
    return { job_id: id, status: "done", queue_position: null, progress: null, error: null, result: job.resolved };
  }
  let stage: Stage = "detecting";
  let step = 0;
  if (t > 2300) {
    stage = "cloaking";
    step = Math.min(job.steps, Math.round(((t - 2300) / 4700) * job.steps));
  }
  if (t > 7000) {
    stage = "swapping";
    step = 0;
  }
  if (t > 8200) stage = "judging";
  return { job_id: id, status: "running", queue_position: null, progress: { stage, step, total: job.steps }, error: null, result: null };
}

export async function deleteJob(id: string): Promise<void> {
  await sleep(60);
  if (!jobs.delete(id)) throw new ApiError("http", 404, "Unknown job.", "Unknown job.");
}
