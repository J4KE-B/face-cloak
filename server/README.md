# face-cloak demo backend

FastAPI backend of the live demo (`https://face-cloak.vercel.app/demo`). Photos are processed in memory, jobs
expire after 10 minutes, nothing is written to disk. Models are loaded once at startup.

```
python -m server [--host 127.0.0.1] [--port 8000] [--insecure-dev]
./server/run_local.sh            # laptop: CPU backend + pinned, checksum-verified cloudflared quick tunnel
notebooks/kaggle_demo_server.ipynb   # Kaggle GPU variant
```

## Consent notice (frontend must show it)

`GET /health` returns `consent_notice`; display it next to the upload control. Current text:

> Only upload photos of yourself or people who consented. Images are processed in memory and deleted within 10 minutes.

Face-swap outputs (`swap_original`, `swap_cloaked`) are capped at 512 px on the long side and carry a visible,
semi-transparent `SYNTHETIC · face-cloak demo` band.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `DEMO_KEY` | required | Passcode (`X-Demo-Key` header), **>= 12 characters** or the server refuses to start |
| `DEMO_SITE` | `https://face-cloak.vercel.app` | Exact production origin allowed by CORS |
| `VERCEL_TEAM_SLUG` | unset | If set, also allows previews `https://<project>-<hash-or-git-branch>-<slug>.vercel.app`; unset = no preview origins |
| `VERCEL_PROJECT` | `face-cloak` | Project prefix used by the preview pattern |
| `ALLOWED_ORIGINS` | empty | Extra exact origins (comma separated) |
| `RATE_LIMIT_PER_HOUR` | 5 | Jobs per client IP per hour |
| `GLOBAL_HOURLY_CAP` | 40 | Jobs per hour across all clients |
| `QUEUE_CAP`, `JOB_TTL_S` | 5, 600 | Queue length, result lifetime |
| `FACECLOAK_DEV=1` / `--insecure-dev` | off | No passcode, localhost only, `/docs` enabled, `localhost:3000` CORS |

Always generate the passcode, never invent one:

```bash
export DEMO_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')"
```

## Abuse and resource protections

- **Charge before parse.** Every authenticated `POST /api/jobs` is charged to the per-IP and global hourly limiter
  (atomic reserve) in the ASGI guard *before* the multipart body is read, so malformed uploads are not free. The
  charge is refunded only if the response is 413 or 503.
- **Robust mode on CPU counts as 2 jobs** in the per-IP limiter (it runs two recognizers + EOT, roughly 2x the cost).
  On CUDA it counts as 1.
- **Step caps per device:** CPU max 60, CUDA max 200 (`/health` reports `max_steps`).
- **Admission control:** at most 2 requests decode + detect at the same time (asyncio semaphore; decoding runs in the
  threadpool). uvicorn runs with `limit_concurrency=32`.
- **Decode limits:** 5 MB per photo, 24 MP, JPEGs are DCT-downscaled (`draft`) before decoding, format sniffed and
  enforced (`Image.open(..., formats=[...])`). Oversized chunked bodies get 413.
- **Wrong passcodes:** 20 per IP per 10 minutes, then 429 for that IP.
- **Client IP:** `CF-Connecting-IP` is trusted only when the socket peer is loopback (the local cloudflared tunnel);
  otherwise the socket peer is used. IPv6 clients are keyed by /64.
- **Dev mode** (no passcode) rejects any request carrying `cf-ray` / `cf-connecting-ip` (403): it must never sit
  behind a tunnel. `/docs` exists only in that mode.
- Logs contain short SHA-256 prefixes of job ids, never the ids themselves (they act as bearer tokens).

## Keeping the Kaggle notebook clean

The notebook is committed with **no outputs**. Outputs contain the live tunnel URL, so before committing or sharing:

```bash
pip install nbstripout            # or: uv tool install nbstripout
nbstripout notebooks/kaggle_demo_server.ipynb      # one-off
# optional, per clone (not installed for you): nbstripout --install --attributes .gitattributes
```

On Kaggle: never make the notebook (or any version) public and never "Save Version" with outputs.

## cloudflared pin

`run_local.sh` and the notebook download `cloudflared` **2026.10.0** (`cloudflared-linux-amd64`) from the GitHub
release URL and verify sha256 `d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db`, which is the
checksum published in that release's notes (and equals the GitHub asset digest). To bump: change the version and
copy the new checksum from the release page in both files.

## Tests / schema

```bash
OMP_NUM_THREADS=2 nice -n 10 uv run pytest -q tests/test_server.py   # stub engine, no weights
uv run python -m server.export_schema                                  # regenerates server/openapi.json
```
