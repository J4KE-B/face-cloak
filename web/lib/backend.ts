/**
 * Backend URL policy. The browser sends the passcode to whatever host is configured, so only a tight allow-list
 * is accepted: Cloudflare quick tunnels (https) and loopback (http), plus the in-browser "mock".
 */

export const MOCK_URL = "mock";

const TRYCLOUDFLARE = /^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*\.trycloudflare\.com$/;

export function normalizeBackendUrl(input: string): string | null {
  const raw = input.trim();
  if (!raw) return null;
  if (raw.toLowerCase() === MOCK_URL) return MOCK_URL;
  let u: URL;
  try {
    u = new URL(raw);
  } catch {
    return null;
  }
  if (u.username || u.password) return null;
  const host = u.hostname.toLowerCase();
  if (u.protocol === "https:") {
    if (!TRYCLOUDFLARE.test(host)) return null;
    if (u.port && u.port !== "443") return null;
    return `https://${host}`;
  }
  if (u.protocol === "http:") {
    if (host !== "localhost" && host !== "127.0.0.1") return null;
    return `http://${host}${u.port ? `:${u.port}` : ""}`;
  }
  return null;
}

export const URL_HELP =
  "Only https://<name>.trycloudflare.com tunnels, http://localhost:PORT or http://127.0.0.1:PORT are accepted (or 'mock').";

export function hostOf(url: string): string {
  if (url === MOCK_URL) return "in-browser mock";
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
