"""Filesystem locations. Override with FACECLOAK_WEIGHTS / FACECLOAK_DATA / FACECLOAK_RESULTS."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = Path(os.environ.get("FACECLOAK_WEIGHTS", REPO_ROOT / "weights"))
DATA_DIR = Path(os.environ.get("FACECLOAK_DATA", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("FACECLOAK_RESULTS", REPO_ROOT / "results"))
CACHE_DIR = RESULTS_DIR / "cache"
