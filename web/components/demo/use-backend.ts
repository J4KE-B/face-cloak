"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, friendlyError, getHealth, getTargets } from "@/lib/api";
import { MOCK_URL, URL_HELP, normalizeBackendUrl } from "@/lib/backend";
import { HEALTH_INTERVAL_MS, STORAGE_KEY } from "@/lib/config";
import type { Health, Target } from "@/lib/types";
import { useToast } from "@/components/toast";

export interface Conn {
  url: string;
  key: string;
  /** true while this host came from a ?backend= link and the user has not yet confirmed the first upload */
  viaLink?: boolean;
}

export type HealthPhase = "idle" | "checking" | "offline" | "waking" | "online";

interface HealthRec {
  forUrl: string;
  data: Health | null; // null = unreachable
}
interface TargetsRec {
  forConn: string;
  state: "ok" | "unauthorized" | "error";
  items: Target[];
}

function loadStored(): Conn | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const j = JSON.parse(raw) as Partial<Conn>;
    const url = typeof j.url === "string" ? normalizeBackendUrl(j.url) : null;
    if (!url) return null;
    return { url, key: typeof j.key === "string" ? j.key : "", viaLink: j.viaLink === true };
  } catch {
    return null;
  }
}
function store(c: Conn | null) {
  try {
    if (c) localStorage.setItem(STORAGE_KEY, JSON.stringify(c));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage blocked (private mode): the connection just is not remembered */
  }
}

export function useBackend() {
  const { toast } = useToast();
  const [urlInput, setUrlInputRaw] = useState("");
  const [keyInput, setKeyInputRaw] = useState("");
  // The backend host the passcode currently typed/stored in keyInput was entered for.
  const keyHost = useRef<string | null>(null);
  const urlNorm = useRef<string | null>(null); // normalized value of the URL field
  const [nonce, setNonce] = useState(0);
  const [conn, setConn] = useState<Conn | null>(null);
  const [fromLink, setFromLink] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthRec | null>(null);
  const [targetsRec, setTargetsRec] = useState<TargetsRec | null>(null);
  const [hydrated, setHydrated] = useState(false);

  // Restore saved connection and handle ?backend= once on mount.
  useEffect(() => {
    const stored = loadStored();
    let linkUrl: string | null = null;
    try {
      const p = new URL(window.location.href);
      const raw = p.searchParams.get("backend");
      if (raw !== null) {
        const n = normalizeBackendUrl(raw);
        if (n) linkUrl = n;
        else toast({ kind: "error", title: "Backend link ignored", message: URL_HELP });
        p.searchParams.delete("backend");
        window.history.replaceState(null, "", p.pathname + (p.search || "") + p.hash);
      }
    } catch {
      /* ignore */
    }
    /* eslint-disable react-hooks/set-state-in-effect -- one-time hydration from localStorage / URL (client-only data) */
    if (linkUrl && (!stored || stored.url !== linkUrl)) {
      // A link to a NEW host: never reuse a stored passcode for it. The user must type it and press Connect.
      setUrlInputRaw(linkUrl);
      urlNorm.current = linkUrl;
      setFromLink(linkUrl);
    } else if (stored) {
      setUrlInputRaw(stored.url);
      urlNorm.current = stored.url;
      setKeyInputRaw(stored.key);
      keyHost.current = stored.url;
      if (stored.key) setConn(stored);
    }
    setHydrated(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [toast]);

  const setKeyInput = useCallback((v: string) => {
    setKeyInputRaw(v);
    if (v === "") {
      keyHost.current = null;
      return;
    }
    // Remember which host this passcode was typed for (null until a valid URL exists).
    keyHost.current = urlNorm.current;
  }, []);

  // Switching to a different backend host never carries the old passcode along.
  const setUrlInput = useCallback((v: string) => {
    setUrlInputRaw(v);
    const n = normalizeBackendUrl(v);
    urlNorm.current = n;
    if (keyHost.current === null) {
      if (n) keyHost.current = n; // passcode typed before the URL: adopt this first valid host
      return;
    }
    if (n !== keyHost.current) {
      setKeyInputRaw("");
      keyHost.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    const url = normalizeBackendUrl(urlInput);
    if (!url) {
      toast({ kind: "error", title: "Backend URL not allowed", message: URL_HELP });
      return false;
    }
    const key = keyInput.trim() || (url === MOCK_URL ? "mock" : "");
    if (!key) {
      toast({ kind: "error", title: "Passcode needed", message: "Enter the DEMO_KEY you set on the backend." });
      return false;
    }
    const c: Conn = { url, key, viaLink: fromLink === url ? true : url === conn?.url ? conn.viaLink === true : false };
    setUrlInputRaw(url);
    urlNorm.current = url;
    setKeyInputRaw(key);
    keyHost.current = url;
    setFromLink(null);
    // Reconnect must really re-run the health + targets fetches even with an unchanged URL/key.
    setTargetsRec(null);
    setHealth(null);
    setNonce((n) => n + 1);
    setConn(c);
    store(c);
    return true;
  }, [urlInput, keyInput, toast, fromLink, conn]);

  /** The user explicitly confirmed sending photos to a host that came from a link. */
  const confirmLink = useCallback(() => {
    setConn((c) => {
      if (!c) return c;
      const n = { ...c, viaLink: false };
      store(n);
      return n;
    });
  }, []);

  const disconnect = useCallback(() => {
    setConn(null);
    setKeyInputRaw("");
    keyHost.current = null;
    setHealth(null);
    setTargetsRec(null);
    store(null);
  }, []);

  // Health polling (no auth needed), every 10 s.
  const url = conn?.url ?? null;
  useEffect(() => {
    if (!url) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const ctrl = new AbortController();
    const tick = async () => {
      let data: Health | null = null;
      try {
        data = await getHealth(url, ctrl.signal);
      } catch {
        data = null;
      }
      if (stop) return;
      setHealth({ forUrl: url, data });
      timer = setTimeout(tick, HEALTH_INTERVAL_MS);
    };
    void tick();
    return () => {
      stop = true;
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [url, nonce]);

  const healthData = health && health.forUrl === url ? health.data : null;
  const phase: HealthPhase = !conn
    ? "idle"
    : !health || health.forUrl !== url
      ? "checking"
      : !healthData
        ? "offline"
        : healthData.status !== "ok" || !healthData.models_loaded
          ? "waking"
          : "online";

  // Targets need the passcode; fetched as soon as the backend is online.
  const key = conn?.key ?? "";
  const online = phase === "online";
  const connId = conn ? `${conn.url}|${conn.key}|${nonce}` : "";
  useEffect(() => {
    if (!url || !online) return;
    const ctrl = new AbortController();
    getTargets(url, key, ctrl.signal)
      .then((items) => setTargetsRec({ forConn: connId, state: "ok", items }))
      .catch((e: unknown) => {
        if (e instanceof ApiError && e.kind === "aborted") return;
        const unauth = e instanceof ApiError && e.status === 401;
        const f = friendlyError(e, url);
        toast({ kind: "error", title: f.title, message: f.message, retryAfter: f.retryAfter });
        setTargetsRec({ forConn: connId, state: unauth ? "unauthorized" : "error", items: [] });
      });
    return () => ctrl.abort();
  }, [url, key, online, connId, nonce, toast]);

  const targets = targetsRec && targetsRec.forConn === connId ? targetsRec : null;
  const targetsState: "idle" | "loading" | "ok" | "unauthorized" | "error" = !online ? "idle" : !targets ? "loading" : targets.state;

  return useMemo(
    () => ({
      urlInput, setUrlInput, keyInput, setKeyInput, conn, connect, disconnect, confirmLink, fromLink, hydrated,
      phase, health: healthData, targets: targets?.items ?? [], targetsState,
    }),
    [urlInput, setUrlInput, keyInput, setKeyInput, conn, connect, disconnect, confirmLink, fromLink, hydrated, phase, healthData, targets, targetsState],
  );
}

export type BackendApi = ReturnType<typeof useBackend>;
