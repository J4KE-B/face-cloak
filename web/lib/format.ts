export const pct = (x: number | null | undefined, d = 1) =>
  x == null || !Number.isFinite(x) ? "n/a" : `${(100 * x).toFixed(d)}%`;

export const num = (x: number | null | undefined, d = 3) =>
  x == null || !Number.isFinite(x) ? "n/a" : x.toFixed(d);

export function estimateSeconds(device: string | undefined, steps: number, mode: "fast" | "robust") {
  const robust = mode === "robust" ? 2 : 1;
  const gpu = device === "cuda";
  const per = gpu ? 0.12 : 1.3;
  const fixed = gpu ? 12 : 25;
  return Math.round((fixed + steps * per) * robust);
}

export function fmtDuration(s: number) {
  if (s < 90) return `~${Math.max(5, Math.round(s / 5) * 5)} s`;
  return `~${Math.round(s / 60)} min`;
}
