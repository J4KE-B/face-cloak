"""In-memory job queue with a single worker thread (one heavy job at a time), TTL expiry and rate limiting.

Nothing is written to disk and uploaded pixels are dropped as soon as a job finishes. Logs contain short job-id hashes,
stages and timings only.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from server import imaging
from server.engine import Engine, Face, JobCancelled
from server.schema import JobResult, JobStatus, Params, Progress, Quality, Scores, Timings
from server.settings import Settings

log = logging.getLogger("facecloak.server")


def log_id(job_id: str) -> str:
    """Short hash of a job id for logs: events can be correlated but the (secret) id never reaches a log."""
    return hashlib.sha256(job_id.encode()).hexdigest()[:8]


class RateLimiter:
    """Sliding-window limiter: ``limit`` events per ``window`` seconds per key."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float]) -> None:
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        q = self._hits.setdefault(key, deque())
        while q and q[0] <= now - self.window:
            q.popleft()
        if not q:
            self._hits.pop(key, None)
            q = self._hits.setdefault(key, deque())
        return q

    def retry_after(self, key: str) -> int | None:
        """Seconds until a new event is allowed, or None if allowed now."""
        with self._lock:
            now = self.clock()
            q = self._prune(key, now)
            if len(q) < self.limit:
                if not q:
                    self._hits.pop(key, None)
                return None
            return max(1, int(q[0] + self.window - now) + 1)

    def record(self, key: str) -> None:
        with self._lock:
            self._prune(key, self.clock()).append(self.clock())

    def reserve(self, key: str, cost: int = 1) -> int | None:
        """Atomically check-and-charge ``cost`` events. Returns None if charged, else seconds until allowed."""
        with self._lock:
            now = self.clock()
            if len(self._hits) > 20_000:  # bound memory under address-spraying
                for k in list(self._hits):
                    self._prune(k, now)
            q = self._prune(key, now)
            over = len(q) + cost - self.limit
            if over > 0:
                if over > len(q):  # cost alone exceeds the limit
                    return max(1, int(self.window))
                return max(1, int(q[over - 1] + self.window - now) + 1)
            q.extend([now] * cost)
            return None

    def refund(self, key: str, cost: int = 1) -> None:
        """Undo a ``reserve`` (newest entries first)."""
        with self._lock:
            q = self._hits.get(key)
            for _ in range(cost):
                if q:
                    q.pop()
            if not q:
                self._hits.pop(key, None)


@dataclass
class Job:
    id: str
    ip: str
    created: float
    eps: int
    steps: int
    mode: str
    target_id: str
    photo: np.ndarray | None
    face: Face | None
    reference: np.ndarray | None
    reference_face: Face | None
    status: str = "queued"
    progress: Progress | None = None
    error: str | None = None
    result: JobResult | None = None
    finished: float | None = None
    cancelled: bool = False
    expired_at: float | None = None


class JobManager:
    def __init__(
        self, engine: Engine, settings: Settings, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.engine, self.s, self.clock = engine, settings, clock
        self.limiter = RateLimiter(settings.rate_limit_per_hour, 3600.0, clock)  # per client IP
        self.global_limiter = RateLimiter(settings.global_hourly_cap, 3600.0, clock)  # all clients together
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._running: str | None = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="facecloak-worker", daemon=True)

    # ---- lifecycle
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5)

    # ---- admission
    def queue_length(self) -> int:
        with self._cv:
            return len(self._queue) + (1 if self._running else 0)

    def queue_full(self) -> bool:
        with self._cv:
            return len(self._queue) >= self.s.queue_cap

    def submit(self, ip: str, job: Job) -> int:
        with self._cv:
            if len(self._queue) >= self.s.queue_cap:
                raise OverflowError("queue full")
            self._sweep_locked()
            self._jobs[job.id] = job
            self._queue.append(job.id)
            pos = len(self._queue) - 1 + (1 if self._running else 0)
            self._cv.notify()
            return pos

    def charge(self, ip: str, cost: int = 1) -> tuple[int, str] | None:
        """Reserve ``cost`` job slots for ``ip`` (per-IP hourly limit, then the global hourly cap).

        Returns None when charged, else (retry_after_seconds, "ip" | "global"). Atomic per limiter.
        """
        retry = self.limiter.reserve(ip, cost)
        if retry is not None:
            return retry, "ip"
        retry = self.global_limiter.reserve("*", cost)
        if retry is not None:
            self.limiter.refund(ip, cost)
            return retry, "global"
        return None

    def refund(self, ip: str, cost: int = 1) -> None:
        self.limiter.refund(ip, cost)
        self.global_limiter.refund("*", cost)

    @staticmethod
    def new_id() -> str:
        return secrets.token_urlsafe(16)

    # ---- queries
    def _position_locked(self, job: Job) -> int | None:
        if job.status == "queued":
            return list(self._queue).index(job.id) + (1 if self._running else 0)
        return None

    def status(self, job_id: str) -> JobStatus | None:
        with self._cv:
            self._sweep_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return JobStatus(
                job_id=job.id,
                status=job.status,  # type: ignore[arg-type]
                queue_position=self._position_locked(job),
                progress=job.progress if job.status in ("running", "queued") else None,
                error=job.error,
                result=job.result if job.status == "done" else None,
            )

    def delete(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            job.cancelled = True
            if job.id in self._queue:
                self._queue.remove(job.id)
            job.photo = job.reference = job.result = None
            return True

    def _sweep_locked(self) -> None:
        now = self.clock()
        for jid, job in list(self._jobs.items()):
            if (
                job.status in ("done", "error")
                and job.finished is not None
                and now - job.finished >= self.s.job_ttl_s
            ):
                job.status, job.result, job.error, job.expired_at = "expired", None, None, now
            elif (
                job.status == "expired"
                and job.expired_at is not None
                and now - job.expired_at >= self.s.job_ttl_s
            ):
                del self._jobs[jid]  # tombstone gone -> 404

    # ---- worker
    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait(timeout=5)
                    self._sweep_locked()
                if self._stop:
                    return
                jid = self._queue.popleft()
                job = self._jobs.get(jid)
                if job is None:
                    continue
                job.status = "running"
                self._running = jid
            try:
                self._run(job)
            except JobCancelled:
                log.info("job %s cancelled", log_id(job.id))
            except Exception as e:  # never leak details/pixels to the client
                log.exception("job %s failed (%s)", log_id(job.id), type(e).__name__)
                job.status = "error"
                job.error = "Processing failed on the server. Please try again or use a different photo."
                job.finished = self.clock()
            finally:
                job.photo = job.reference = None
                with self._cv:
                    self._running = None

    def _progress(self, job: Job, stage: str, step: int = 0) -> None:
        if job.cancelled:
            raise JobCancelled
        job.progress = Progress(stage=stage, step=step, total=job.steps)  # type: ignore[arg-type]

    def _run(self, job: Job) -> None:
        eng, t_all = self.engine, time.perf_counter()
        img, face = job.photo, job.face
        assert img is not None and face is not None
        self._progress(job, "detecting")
        if job.reference is not None and job.reference_face is not None:
            ref_emb, ref_kind = eng.embed_judge(job.reference, job.reference_face), "uploaded"
        else:
            ref_emb, ref_kind = eng.embed_judge(img, face), "self"

        self._progress(job, "cloaking", 0)
        t_c = time.perf_counter()
        cloaked = eng.cloak(
            img, face, job.eps, job.steps, job.mode, lambda s, n: self._progress(job, "cloaking", min(s, n))
        )
        cloak_s = time.perf_counter() - t_c

        self._progress(job, "swapping")
        swap_o = eng.swap(img, job.target_id)
        self._progress(job, "swapping")
        swap_c = eng.swap(cloaked, job.target_id)

        self._progress(job, "judging")
        sim_o = eng.judge_swap(swap_o, job.target_id, ref_emb) if swap_o is not None else None
        sim_c = eng.judge_swap(swap_c, job.target_id, ref_emb) if swap_c is not None else None
        tau = eng.tau
        psnr, ssim, lp = eng.quality(img, cloaked)
        linf = int(np.abs(cloaked.astype(np.int16) - img.astype(np.int16)).max())

        protected, note = verdict(sim_o, sim_c, tau)
        if ref_kind == "self":
            note = (note or "") + SELF_REFERENCE_NOTE
        job.result = JobResult(
            original=imaging.png_data_url(img, 1024),
            cloaked=imaging.png_data_url(cloaked, 1024),
            perturbation=imaging.png_data_url(imaging.perturbation_image(img, cloaked)),
            swap_original=imaging.png_data_url(imaging.swap_output(swap_o)) if swap_o is not None else None,
            swap_cloaked=imaging.png_data_url(imaging.swap_output(swap_c)) if swap_c is not None else None,
            cloaked_download=imaging.png_data_url(cloaked),
            scores=Scores(
                id_sim_original=_f(sim_o),
                id_sim_cloaked=_f(sim_c),
                tau=tau,
                judge="glintr100",
                reference=ref_kind,  # type: ignore[arg-type]
            ),
            quality=Quality(psnr=_f(psnr), ssim=_f(ssim), lpips=_f(lp), linf_255=linf),
            protected=protected,
            note=note,
            timings=Timings(cloak_s=round(cloak_s, 2), total_s=round(time.perf_counter() - t_all, 2)),
            params=Params(eps=job.eps, steps=job.steps, mode=job.mode),  # type: ignore[arg-type]
        )
        job.progress = Progress(stage="judging", step=job.steps, total=job.steps)
        job.finished = self.clock()
        job.status = "done"
        log.info("job %s done in %.1fs (cloak %.1fs)", log_id(job.id), time.perf_counter() - t_all, cloak_s)


SELF_REFERENCE_NOTE = (
    " Note: no separate reference photo was uploaded, so this verdict compares the swap against the same photo "
    "that was cloaked, which is optimistic. Upload a second, different photo of the same person for a more "
    "honest test."
)


def _f(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return 99.0 if x == float("inf") else round(x, 4)


def verdict(sim_o: float | None, sim_c: float | None, tau: float) -> tuple[bool | None, str | None]:
    """protected = id_sim_cloaked < tau and id_sim_original >= tau; null when the clean attack itself fails."""
    if sim_o is None or sim_o < tau:
        return None, (
            "Inconclusive: the face swap built from your unprotected photo did not match the reference identity "
            "(similarity below the threshold), so there was nothing to protect against in this test. "
            "Try a sharper, front-facing photo or a different target."
        )
    if sim_c is None:
        return True, "Protected: the swap tool could not even find a face in the cloaked photo."
    if sim_c < tau:
        return True, "Protected: the swap built from the cloaked photo no longer matches your identity."
    return (
        False,
        "Not protected: the swap from the cloaked photo still matches your identity. Try a larger eps or robust mode.",
    )
