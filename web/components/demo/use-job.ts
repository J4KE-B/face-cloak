"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, createJob, deleteJob, friendlyError, getJob, type FriendlyError } from "@/lib/api";
import { POLL_INTERVAL_MS } from "@/lib/config";
import type { JobParams, JobResult, JobStatus } from "@/lib/types";
import { useToast } from "@/components/toast";
import type { Conn } from "./use-backend";

export type JobPhase = "idle" | "submitting" | "active" | "done" | "failed";

export function useJob(conn: Conn | null) {
  const { toast } = useToast();
  const [phase, setPhase] = useState<JobPhase>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [failure, setFailure] = useState<FriendlyError | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const submitCtrl = useRef<AbortController | null>(null);
  // Bumped whenever the current submit is cancelled/superseded; a late POST response is then deleted immediately.
  const submitToken = useRef(0);
  // The job (and the backend it lives on) that this tab is responsible for deleting.
  const activeJob = useRef<{ url: string; key: string; id: string } | null>(null);

  const url = conn?.url ?? null;
  const key = conn?.key ?? "";

  const fail = useCallback(
    (e: unknown) => {
      const f = friendlyError(e, url);
      setFailure(f);
      setPhase("failed");
      toast({ kind: "error", title: f.title, message: f.message, retryAfter: f.retryAfter });
    },
    [toast, url],
  );

  const start = useCallback(
    async (p: JobParams) => {
      if (!url) return;
      setPhase("submitting");
      setStatus(null);
      setResult(null);
      setFailure(null);
      setStartedAt(Date.now());
      const ctrl = new AbortController();
      submitCtrl.current = ctrl;
      const token = ++submitToken.current;
      try {
        const created = await createJob(url, key, p, ctrl.signal);
        if (token !== submitToken.current || ctrl.signal.aborted) {
          // Cancelled (or connection changed / unmounted) while the POST was in flight: do not leave the job behind.
          void deleteJob(url, key, created.job_id).catch(() => {});
          return;
        }
        activeJob.current = { url, key, id: created.job_id };
        setStatus({ job_id: created.job_id, status: "queued", queue_position: created.queue_position, progress: null, error: null, result: null });
        setJobId(created.job_id);
        setPhase("active");
      } catch (e) {
        if (token !== submitToken.current) return;
        if (e instanceof ApiError && e.kind === "aborted") {
          setPhase("idle");
          return;
        }
        fail(e);
      }
    },
    [url, key, fail],
  );

  // Poll the active job every 1.5 s.
  useEffect(() => {
    if (phase !== "active" || !jobId || !url) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    let netFails = 0;
    const ctrl = new AbortController();
    const tick = async () => {
      try {
        const st = await getJob(url, key, jobId, ctrl.signal);
        if (stop) return;
        netFails = 0;
        setStatus(st);
        if (st.status === "done" && st.result) {
          setResult(st.result);
          setPhase("done");
          // The result now lives in this tab only: free the server-side copy right away (best effort).
          activeJob.current = null;
          void deleteJob(url, key, jobId).catch(() => {});
          return;
        }
        if (st.status === "error" || st.status === "expired") {
          activeJob.current = null;
          fail(new ApiError("http", st.status === "expired" ? 404 : 500, "job failed", st.error ?? (st.status === "expired" ? "The job expired (results are deleted after 10 minutes)." : "The backend reported an error.")));
          return;
        }
      } catch (e) {
        if (stop) return;
        const transient = e instanceof ApiError && (e.kind === "network" || e.kind === "timeout");
        if (!transient || ++netFails >= 4) {
          fail(e);
          return;
        }
      }
      timer = setTimeout(tick, POLL_INTERVAL_MS);
    };
    void tick();
    return () => {
      stop = true;
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [phase, jobId, url, key, fail]);

  const reset = useCallback(() => {
    submitToken.current++;
    setPhase("idle");
    setJobId(null);
    setStatus(null);
    setResult(null);
    setFailure(null);
    setStartedAt(null);
  }, []);

  const cancel = useCallback(async () => {
    submitCtrl.current?.abort();
    const id = jobId;
    activeJob.current = null;
    reset();
    if (id && url) {
      try {
        await deleteJob(url, key, id);
        toast({ kind: "info", title: "Cancelled", message: "The job and its images were deleted from the backend." });
      } catch {
        toast({ kind: "info", title: "Cancelled", message: "Stopped waiting. The backend deletes the job automatically after 10 minutes." });
      }
    }
  }, [jobId, url, key, reset, toast]);

  // Connection changed, Forget pressed, or the page is closing: abort the submit, delete the job on its own
  // backend (best effort) and drop all job state. Cleanup also runs on unmount.
  useEffect(() => {
    const token = submitToken;
    return () => {
      submitCtrl.current?.abort();
      token.current++;
      const j = activeJob.current;
      activeJob.current = null;
      if (j) void deleteJob(j.url, j.key, j.id).catch(() => {});
      setPhase("idle");
      setJobId(null);
      setStatus(null);
      setResult(null);
      setFailure(null);
      setStartedAt(null);
    };
  }, [url, key]);

  return { phase, status, result, failure, startedAt, start, cancel, reset };
}
