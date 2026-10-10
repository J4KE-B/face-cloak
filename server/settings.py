"""Runtime configuration (environment variables) for the demo backend."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_SIDE = 1024
MIN_SIDE = 64
MAX_EPS = 16
MAX_STEPS = 200  # schema-level ceiling; the per-device caps below are enforced in the endpoint
MAX_STEPS_CPU = 60
MAX_STEPS_CUDA = 200
MIN_KEY_LEN = 12
ADMISSION_SLOTS = 2  # concurrent decode+detect operations across all requests
DEFAULT_SITE = "https://face-cloak.vercel.app"


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    demo_key: str | None = None
    insecure_dev: bool = False
    allowed_origins: list[str] = field(default_factory=list)
    demo_site: str = DEFAULT_SITE  # exact production origin allowed by CORS
    vercel_project: str = "face-cloak"
    vercel_team_slug: str | None = None  # preview deployments are allowed only if this is set
    rate_limit_per_hour: int = 5
    queue_cap: int = 5
    job_ttl_s: float = 600.0
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    global_hourly_cap: int = 40  # jobs per hour across ALL clients
    auth_fail_limit: int = 20  # wrong passcodes per IP ...
    auth_fail_window_s: float = 600.0  # ... per this window, then 429 for that IP
    admission_slots: int = ADMISSION_SLOTS

    @classmethod
    def from_env(cls) -> Settings:
        origins = [
            o.strip().rstrip("/") for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
        ]
        return cls(
            demo_key=os.environ.get("DEMO_KEY") or None,
            insecure_dev=_env_bool("FACECLOAK_DEV"),
            allowed_origins=origins,
            demo_site=os.environ.get("DEMO_SITE", DEFAULT_SITE).strip().rstrip("/") or DEFAULT_SITE,
            vercel_team_slug=(os.environ.get("VERCEL_TEAM_SLUG", "").strip().lower() or None),
            vercel_project=os.environ.get("VERCEL_PROJECT", "face-cloak"),
            rate_limit_per_hour=int(os.environ.get("RATE_LIMIT_PER_HOUR", "5")),
            queue_cap=int(os.environ.get("QUEUE_CAP", "5")),
            job_ttl_s=float(os.environ.get("JOB_TTL_S", "600")),
            global_hourly_cap=int(os.environ.get("GLOBAL_HOURLY_CAP", "40")),
        )

    def cors_origins(self) -> list[str]:
        origins = [self.demo_site, *self.allowed_origins]
        if self.insecure_dev:
            origins.append("http://localhost:3000")  # local frontend dev only
        return list(dict.fromkeys(origins))

    def cors_origin_regex(self) -> str | None:
        """Vercel preview deployments (<project>-<hash|git-branch>-<team>.vercel.app), anchored to YOUR team.

        Without VERCEL_TEAM_SLUG there is no regex at all: anyone can deploy to <project>-x-<theirteam>.vercel.app.
        """
        if not self.vercel_team_slug:
            return None
        return rf"^https://{re.escape(self.vercel_project)}-[a-z0-9-]+-{re.escape(self.vercel_team_slug)}\.vercel\.app$"
