# face-cloak live-demo backend

FastAPI service behind the website's `/demo` page. A visitor uploads a photo; the server cloaks it (PGD against
ArcFace, optionally + FaceNet with EOT), runs the inswapper face-swap attack on the original and on the cloaked
photo, and judges both swaps with a held-out recogniser (glintr100) against the visitor's identity.

```
browser (Vercel) --HTTPS--> Cloudflare quick tunnel --> uvicorn 127.0.0.1:8000 (this app, one worker thread)
```

## Run

**Kaggle GPU (primary):** open `notebooks/kaggle_demo_server.ipynb`, set Accelerator = GPU and Internet = ON,
add a Kaggle Secret `DEMO_KEY`, Run All, click the auto-connect link printed by cell 6 and keep the tab open.

**Laptop CPU (fallback):**
```bash
DEMO_KEY=mypass ./server/run_local.sh      # prints tunnel URL, passcode, and the auto-connect link
```
It runs the backend under `systemd-run --scope` (MemoryMax=9G, CPUQuota=800%, nice 10, 8 threads), starts
`cloudflared` (downloaded to `~/.local/bin` if missing), and stops both on Ctrl-C.

**Directly:** `DEMO_KEY=... uv run python -m server --host 127.0.0.1 --port 8000`
(`--insecure-dev` / `FACECLOAK_DEV=1` runs without a passcode, loopback only; also enables `/docs`).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DEMO_KEY` | required | passcode, sent by the client as `X-Demo-Key` (constant-time compare). The server refuses to start without it unless dev mode is on |
| `FACECLOAK_DEV` | off | `1` = no auth, refuses non-loopback host |
| `ALLOWED_ORIGINS` | empty | extra CORS origins, comma separated (`http://localhost:3000` is always allowed) |
| `VERCEL_PROJECT` | `face-cloak` | allows `https://<project>.vercel.app` and `https://<project>-*.vercel.app` previews |
| `RATE_LIMIT_PER_HOUR` | 5 | job creations per client IP per hour (429 + `Retry-After`) |
| `QUEUE_CAP` | 5 | max queued jobs (503 beyond) |
| `JOB_TTL_S` | 600 | seconds a finished job is kept (then `expired`, then 404 after another TTL) |
| `FACECLOAK_THREADS` | 8 | CPU thread cap (existing facecloak variable) |

## API (all `/api/*` need `X-Demo-Key`; JSON models in `schema.py`, OpenAPI in `openapi.json`)

* `GET /health` (no auth) -> `Health`
* `GET /api/targets` -> `Target[]`
* `POST /api/jobs` multipart: `photo` (required), `reference` (optional second photo of the same person; else the
  original upload is the reference and `scores.reference="self"`), `target_id`, `eps` (2..16, default 8),
  `steps` (1..200, default 30 CPU / 100 CUDA), `mode` (`fast` | `robust`) -> 202 `JobCreated`
* `GET /api/jobs/{id}` -> `JobStatus` (poll every ~1 s); `DELETE /api/jobs/{id}` -> 204 (cancels if queued/running)
* Errors are `{"detail": "..."}`: 400 no face, 401, 404, 413 too large (>5 MB or >40 MP), 415 not JPEG/PNG/WebP,
  422 bad parameter or tiny image, 429 rate limit, 503 queue full.

`protected = id_sim_cloaked < tau and id_sim_original >= tau`. If the swap of the *unprotected* photo does not match
the reference, `protected` is `null` and `note` explains the test was inconclusive. If the attacker cannot detect any
face in the cloaked photo, `swap_cloaked`/`id_sim_cloaked` are `null` and `protected` is `true`. `tau` is the glintr100
FAR=1% threshold from `results/calibration.json` (else `calibration_quick.json`).

## Design notes

* **One heavy job at a time.** A single worker thread consumes an in-memory FIFO. Face detection of the upload runs in
  the request thread (cheap, ONNX sessions are thread-safe) so "no face" is an immediate 400.
* **Progress is real.** `Cloaker.cloak` has no callback, so `RealEngine` wraps each attack recognizer in a transparent
  proxy that counts grad-enabled `embed_image` calls (exactly one per model per PGD step) and reports `step/total`.
  No change to the core package.
* **Models load once** in the lifespan handler (CUDA if available). `Engine` is a protocol; tests inject `StubEngine`.
* **Perturbation image** = `|cloaked - original| x 10`, clipped to 0..255.
* **Images** are decoded with PIL (EXIF orientation applied, so no double rotation), downscaled to <=1024 px.

## Security notes

* Auth, rate limit, queue cap and the body-size cap run in an ASGI middleware *before* the multipart body is parsed.
  The body is counted as it streams (Content-Length is not trusted) and capped at ~10 MB; each file at 5 MB.
* Upload checks: magic bytes (JPEG/PNG/WebP), pixel count read from the header before decoding (40 MP cap).
* Everything lives in RAM: multipart spooling to temp files is disabled (`spool_max_size` raised), results are dropped
  after the TTL, and the uvicorn access log is off. Logs contain job ids, parameters and timings only; never pixels.
* Rate limiting trusts `CF-Connecting-IP`. That is only sound because the server listens on loopback and is reached
  solely through the tunnel; do not expose the port directly. Quick-tunnel URLs are public: the passcode is the access
  control. CORS is not an auth boundary (no credentials are used); the Vercel preview regex is intentionally loose.
* Error responses never include exception text. `/docs` and `/openapi.json` are disabled unless dev mode.
* The inswapper / InsightFace weights are non-commercial research-only; use the demo accordingly.

## Tests / schema

```bash
OMP_NUM_THREADS=2 nice -n 10 uv run pytest -q tests/test_server.py   # stub engine, no weights
uv run python -m server.export_schema                                 # rewrites server/openapi.json
```
