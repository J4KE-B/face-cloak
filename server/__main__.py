"""``python -m server [--host 127.0.0.1] [--port 8000] [--insecure-dev]``"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from server.app import create_app
from server.settings import Settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--insecure-dev", action="store_true", help="run without DEMO_KEY (localhost only)")
    a = ap.parse_args()
    if a.insecure_dev:
        os.environ["FACECLOAK_DEV"] = "1"
    s = Settings.from_env()
    if s.insecure_dev and not s.demo_key and a.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("--insecure-dev (no DEMO_KEY) is only allowed on a loopback host")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        app = create_app(s)
    except RuntimeError as e:
        raise SystemExit(str(e)) from None
    # access log off: it would record client IPs and paths of every poll; app logs short job-id hashes only.
    # limit_concurrency: beyond 32 open connections uvicorn answers 503 instead of piling up work.
    uvicorn.run(app, host=a.host, port=a.port, access_log=False, log_level="info", limit_concurrency=32)


if __name__ == "__main__":
    main()
