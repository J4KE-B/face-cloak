"""Run (or resume) one experiment config.

Examples:
    uv run python scripts/run_experiments.py --config configs/smoke.yaml
    uv run python scripts/run_experiments.py --config configs/main.yaml --shard 0/2 --out results/raw
    uv run python scripts/run_experiments.py --config configs/main.yaml --shard 1/2 --resume-from results.zip
    uv run python scripts/run_experiments.py --config configs/smoke.yaml --limit 2

Env: FACECLOAK_THREADS (default 8, max 8) caps torch / ORT / BLAS threads.
"""

from __future__ import annotations

import os

_THREADS = str(max(1, min(8, int(os.environ.get("FACECLOAK_THREADS", "8")))))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse
from pathlib import Path

from facecloak.experiment import load_config, run_experiment
from facecloak.paths import RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML in configs/")
    ap.add_argument("--shard", default="0/1", help="i/N: deterministic split of SOURCES")
    ap.add_argument("--resume-from", default=None, help="results zip/dir of an earlier run to continue from")
    ap.add_argument("--out", default=str(RESULTS_DIR / "raw"))
    ap.add_argument("--cache", default=str(RESULTS_DIR / "cache" / "cloaked"), help="cloaked-PNG cache dir")
    ap.add_argument("--limit", type=int, default=None, help="only the first n sources of the shard")
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    s = run_experiment(
        cfg, out=Path(a.out), cache=Path(a.cache), shard_spec=a.shard, limit=a.limit,
        resume_from=a.resume_from, progress=not a.no_progress,
    )
    print(s)


if __name__ == "__main__":
    main()
