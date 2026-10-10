"""Tables (booktabs .tex + .csv) and figures (PDF + PNG) from results/raw.

Example:
    uv run python scripts/make_figures.py --in results/raw --out results/figs --tables results/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

from facecloak.figures import make_all
from facecloak.paths import RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(RESULTS_DIR / "raw"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "figs"))
    ap.add_argument("--tables", default=str(RESULTS_DIR / "tables"))
    a = ap.parse_args()
    make_all(Path(a.inp), Path(a.out), Path(a.tables))


if __name__ == "__main__":
    main()
