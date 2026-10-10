"""Merge result zips / directories from several machines into one results/raw.

Example:
    uv run python scripts/merge_results.py a.zip b.zip --out results/raw [--cache results/cache/cloaked]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from facecloak.experiment import merge_into
from facecloak.paths import RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=str(RESULTS_DIR / "raw"))
    ap.add_argument("--cache", default=str(RESULTS_DIR / "cache" / "cloaked"))
    a = ap.parse_args()
    out = Path(a.out)
    # keep what is already in --out (it comes first, so its rows win), never silently overwrite it
    inputs = ([out] if out.is_dir() else []) + list(a.inputs)
    rep = merge_into(inputs, out, Path(a.cache))
    print("\n".join(rep.lines()))
    if any(rep.duplicates.values()):
        print("WARNING: duplicate keys found (first occurrence kept); check that shards did not overlap.")


if __name__ == "__main__":
    main()
