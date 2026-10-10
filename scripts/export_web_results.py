"""Write web-ready results for the Next.js /results page.

Produces web/public/results/summary.json and copies PNG figures (including the qualitative grid, if made).

Example:
    uv run python scripts/export_web_results.py --raw results/raw --figs results/figs --web web/public/results
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

from facecloak.figures import build_tables, load_raw
from facecloak.paths import REPO_ROOT, RESULTS_DIR


def _clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 5)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RESULTS_DIR / "raw"))
    ap.add_argument("--figs", default=str(RESULTS_DIR / "figs"))
    ap.add_argument("--web", default=str(REPO_ROOT / "web" / "public" / "results"))
    a = ap.parse_args()
    web = Path(a.web)
    web.mkdir(parents=True, exist_ok=True)
    R = load_raw(Path(a.raw))
    T = build_tables(R) if len(R.swap) else {}
    figs = sorted(Path(a.figs).glob("*.png")) if Path(a.figs).is_dir() else []
    for f in figs:
        shutil.copyfile(f, web / f.name)
    if not figs:
        print("notice: no PNG figures found; run make_figures.py first")
    summary = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "configs": R.configs,
        "whitebox_judges": sorted(R.whitebox()),
        "tau": {c: m["runs"][-1].get("tau") for c, m in R.metas.items() if m.get("runs")},
        "tau_used": R.tau_used,
        "tau_source": R.tau_source,
        "tables": {k: _clean(json.loads(v.to_json(orient="records"))) for k, v in T.items()},
        "figures": [f.name for f in figs],
        "n_swap_rows": len(R.swap),
        "note": "PSR = share of valid swaps with id-sim below tau; CI = 95% cluster bootstrap over source ids (Wilson if degenerate); inswapper = white-box encoder, SimSwap = black-box; null = not available",
    }
    (web / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {web / 'summary.json'} ({len(T)} tables, {len(figs)} figures)")


if __name__ == "__main__":
    main()
