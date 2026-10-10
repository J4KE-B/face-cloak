"""FastAPI app of the live demo. Build with ``create_app()``; run with ``python -m server``."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartParser
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from server import imaging
from server.engine import Engine, Face
from server.jobs import Job, JobManager, RateLimiter, log_id
from server.schema import CONSENT_NOTICE, Health, JobCreated, JobStatus, Mode, Target
from server.settings import MAX_EPS, MAX_STEPS, MAX_STEPS_CPU, MAX_STEPS_CUDA, MIN_KEY_LEN, Settings

VERSION = "0.1.0"
log = logging.getLogger("facecloak.server")

# Keep uploaded images in RAM: Starlette spools multipart files >1 MB to a temp file on disk by default.
MultiPartParser.spool_max_size = 16 * 1024 * 1024


def _norm_ip(raw: str) -> str | None:
    """Canonical rate-limit key for an IP literal: IPv4 as is, IPv4-mapped IPv6 unmapped, other IPv6 by /64."""
    try:
        ip = ipaddress.ip_address(raw.strip())
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network((ip, 64), strict=False))  # one customer owns a whole /64
    return str(ip)


def _is_loopback(raw: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def client_ip(scope: Scope) -> str:
    """Rate-limit key of the caller.

    ``CF-Connecting-IP`` is trusted ONLY when the socket peer is loopback (i.e. the request came through the local
    cloudflared tunnel); anyone who can reach the port directly could otherwise forge it and dodge every limit.
    Otherwise the socket peer is used. IPv6 addresses are keyed by their /64.
    """
    c = scope.get("client")
    peer = c[0] if c else "unknown"
    if _is_loopback(peer):
        for k, v in scope.get("headers", []):
            if k == b"cf-connecting-ip":
                n = _norm_ip(v.decode("latin-1"))
                if n:
                    return n
                break
    return _norm_ip(peer) or peer[:64]


def _json(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


class Guard:
    """ASGI middleware for /api/*: auth (+ wrong-passcode throttle), rate limit, queue cap, streamed body limit.

    Runs *before* the multipart body is parsed. Every authenticated ``POST /api/jobs`` is charged to the
    per-IP and global hourly limiters up front (atomic reserve), refunded only when the response is 413 or 503, so
    garbage uploads cannot be used to burn CPU for free.
    """

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings,
        get_jobs: Callable[[], JobManager | None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app, self.s, self.get_jobs = app, settings, get_jobs
        self.max_body = 2 * settings.max_upload_bytes + 64 * 1024  # photo + reference + multipart framing
        self.key = settings.demo_key.encode() if settings.demo_key else None
        self.auth_fails = RateLimiter(settings.auth_fail_limit, settings.auth_fail_window_s, clock)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v for k, v in scope["headers"]}
        if (
            self.s.insecure_dev
            and self.key is None
            and ("cf-ray" in headers or "cf-connecting-ip" in headers)
        ):
            # unauthenticated dev server reached through Cloudflare => someone exposed it publicly by mistake
            return await _json(403, "Dev mode (no passcode) refuses requests that arrive through a tunnel.")(
                scope, receive, send
            )
        if not scope["path"].startswith("/api/") or scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        jobs = self.get_jobs()
        if jobs is None:
            return await _json(503, "The server is still starting up.")(scope, receive, send)
        ip = client_ip(scope)
        if self.key is not None:
            retry = self.auth_fails.retry_after(ip)
            if retry is not None:
                return await _json(
                    429, "Too many wrong passcodes. Try again later.", {"Retry-After": str(retry)}
                )(scope, receive, send)
            given = headers.get("x-demo-key", b"")
            if not hmac.compare_digest(given, self.key):
                self.auth_fails.reserve(ip)
                return await _json(401, "Missing or wrong passcode.")(scope, receive, send)
        if scope["method"] == "POST" and scope["path"] == "/api/jobs":
            denied = jobs.charge(ip)
            if denied is not None:
                retry, why = denied
                msg = (
                    f"Rate limit reached ({self.s.rate_limit_per_hour} photos per hour). Please try again later."
                    if why == "ip"
                    else "The demo has reached its hourly capacity. Please try again later."
                )
                return await _json(429, msg, {"Retry-After": str(retry)})(scope, receive, send)
            charge = [1]  # slots reserved for this request; create_job may add one (robust mode on CPU)
            scope["facecloak_charge"] = charge

            def refund() -> None:
                if charge[0]:
                    jobs.refund(ip, charge[0])
                    charge[0] = 0

            if jobs.queue_full():
                refund()
                return await _json(
                    503,
                    "The demo is busy right now. Please try again in a few minutes.",
                    {"Retry-After": "60"},
                )(scope, receive, send)
            cl = headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > self.max_body:
                refund()
                return await _json(413, "Upload too large (max 5 MB per photo).")(scope, receive, send)
            seen = 0

            async def limited_receive() -> Message:
                nonlocal seen
                msg = await receive()
                if msg["type"] == "http.request":
                    seen += len(msg.get("body", b""))
                    if seen > self.max_body:  # chunked / lying Content-Length
                        raise HTTPException(413, "Upload too large (max 5 MB per photo).")
                return msg

            async def tracking_send(msg: Message) -> None:
                if msg["type"] == "http.response.start" and msg["status"] in (413, 503):
                    refund()
                await send(msg)

            return await self.app(scope, limited_receive, tracking_send)
        return await self.app(scope, receive, send)


def create_app(
    settings: Settings | None = None,
    engine_factory: Callable[[], Engine] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    s = settings or Settings.from_env()
    if not s.demo_key and not s.insecure_dev:
        raise RuntimeError(
            "DEMO_KEY is not set. Set DEMO_KEY=<passcode>, or pass --insecure-dev / FACECLOAK_DEV=1 "
            "(localhost only, no authentication)."
        )
    if s.demo_key and len(s.demo_key) < MIN_KEY_LEN:
        raise RuntimeError(
            f"DEMO_KEY is too short ({len(s.demo_key)} chars; minimum {MIN_KEY_LEN}). Generate one with: "
            "python -c 'import secrets; print(secrets.token_urlsafe(12))'"
        )
    if engine_factory is None:
        from server.real_engine import RealEngine

        engine_factory = RealEngine

    holder: dict[str, JobManager] = {}
    # global admission control: at most N requests decode + detect at once (CPU/RAM bound), the rest wait here
    admission = asyncio.Semaphore(s.admission_slots)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = await run_in_threadpool(engine_factory)  # models are loaded ONCE, before serving
        jobs = JobManager(engine, s, clock)
        jobs.start()
        holder["jobs"] = jobs
        app.state.jobs, app.state.engine, app.state.targets = jobs, engine, engine.targets()
        log.info("engine ready on %s", engine.device)
        try:
            yield
        finally:
            holder.pop("jobs", None)
            jobs.stop()

    docs_on = s.insecure_dev and not s.demo_key  # API docs only on an unauthenticated localhost dev server
    app = FastAPI(
        title="face-cloak demo backend",
        version=VERSION,
        lifespan=lifespan,
        docs_url="/docs" if docs_on else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs_on else None,
    )
    app.state.admission = admission
    app.add_middleware(Guard, settings=s, get_jobs=lambda: holder.get("jobs"), clock=clock)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins(),
        allow_origin_regex=s.cors_origin_regex(),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["X-Demo-Key", "Content-Type"],
        allow_credentials=False,
        max_age=600,
    )

    def _step_cap(eng: Engine) -> int:
        return MAX_STEPS_CUDA if eng.device == "cuda" else MAX_STEPS_CPU

    def jobs_of(request: Request) -> JobManager:
        return request.app.state.jobs

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        eng: Engine = request.app.state.engine
        return Health(
            status="ok",
            version=VERSION,
            device=eng.device,
            models_loaded=True,
            queue_length=jobs_of(request).queue_length(),
            default_steps=eng.default_steps,
            max_eps=MAX_EPS,
            tau=round(float(eng.tau), 4),
            max_steps=_step_cap(eng),
            consent_notice=CONSENT_NOTICE,
        )

    @app.get("/api/targets", response_model=list[Target])
    def targets(request: Request) -> list[Target]:
        return request.app.state.targets

    async def _read_image(upload, what: str) -> bytes:
        data = await upload.read(s.max_upload_bytes + 1)
        if len(data) > s.max_upload_bytes:
            raise imaging.UploadError(413, f"The {what} is larger than 5 MB.")
        if not data:
            raise imaging.UploadError(422, f"The {what} is empty.")
        return data

    @app.post("/api/jobs", response_model=JobCreated, status_code=202)
    async def create_job(
        request: Request,
        photo: Annotated[UploadFile, File(description="Photo to protect (JPEG/PNG/WebP, <=5 MB)")],
        target_id: Annotated[str, Form()],
        reference: Annotated[
            UploadFile | None, File(description="Optional second photo of the same person")
        ] = None,
        eps: Annotated[int, Form(ge=2, le=MAX_EPS)] = 8,
        steps: Annotated[int | None, Form(ge=1, le=MAX_STEPS)] = None,
        mode: Annotated[Mode, Form()] = "fast",
    ):
        eng: Engine = request.app.state.engine
        jobs = jobs_of(request)
        ip = client_ip(request.scope)
        if not eng.has_target(target_id):
            return _json(422, "Unknown target_id.")
        cap = _step_cap(eng)
        if steps is not None and steps > cap:
            return _json(422, f"steps is limited to {cap} on this server ({eng.device}).")
        if mode == "robust" and eng.device != "cuda":
            # robust (2 recognizers + EOT) costs ~2x on CPU: it also uses up a second hourly slot
            denied = jobs.charge(ip)
            if denied is not None:
                return _json(
                    429,
                    "Robust mode counts as 2 photos on this CPU server and would exceed your hourly limit.",
                    {"Retry-After": str(denied[0])},
                )
            charge = request.scope.get("facecloak_charge")
            if charge is not None:
                charge[0] += 1
        try:
            photo_bytes = await _read_image(photo, "photo")
            ref_bytes = None
            if reference is not None and reference.filename:
                ref_bytes = await _read_image(reference, "reference photo")
        except imaging.UploadError as e:
            return _json(e.status, e.message)
        async with admission:  # decode + detection are the CPU/RAM-heavy part of a request
            try:
                img = await run_in_threadpool(imaging.decode_upload, photo_bytes, "photo")
                ref_img = None
                if ref_bytes is not None:
                    ref_img = await run_in_threadpool(imaging.decode_upload, ref_bytes, "reference photo")
            except imaging.UploadError as e:
                return _json(e.status, e.message)
            face: Face | None = await run_in_threadpool(eng.detect, img)
            if face is None:
                return _json(
                    400,
                    "No face was found in your photo. Use a clear, front-facing photo with one visible face.",
                )
            ref_face = None
            if ref_img is not None:
                ref_face = await run_in_threadpool(eng.detect, ref_img)
                if ref_face is None:
                    return _json(400, "No face was found in the reference photo.")
        job = Job(
            id=jobs.new_id(),
            ip=ip,
            created=clock(),
            eps=eps,
            steps=steps or eng.default_steps,
            mode=mode,
            target_id=target_id,
            photo=img,
            face=face,
            reference=ref_img,
            reference_face=ref_face,
        )
        try:
            pos = jobs.submit(job.ip, job)
        except OverflowError:
            return _json(
                503, "The demo is busy right now. Please try again in a few minutes.", {"Retry-After": "60"}
            )
        log.info("job %s queued (eps=%d steps=%d mode=%s pos=%d)", log_id(job.id), eps, job.steps, mode, pos)
        return JobCreated(job_id=job.id, queue_position=pos)

    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    def get_job(job_id: str, request: Request):
        st = jobs_of(request).status(job_id)
        if st is None:
            return _json(404, "Unknown job.")
        return st

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, request: Request):
        if not jobs_of(request).delete(job_id):
            return _json(404, "Unknown job.")
        return Response(status_code=204)

    return app
