"""Report which (attack, source) cloaks and swap rows are still missing, and suggested shard splits.

Example:
    uv run python scripts/remaining_work.py --config configs/main.yaml --results results.zip [more.zip dir/]
"""

from __future__ import annotations

import argparse

from facecloak.data import load_manifest
from facecloak.experiment import load_config, remaining_work


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", nargs="+", required=True, help="zip(s) or dir(s) with existing CSVs")
    ap.add_argument("--teammates", type=int, default=2)
    ap.add_argument("--list", action="store_true", help="print every missing key")
    a = ap.parse_args()
    cfg = load_config(a.config)
    r = remaining_work(cfg, load_manifest(cfg.manifest), a.results, a.teammates)
    print(f"[{cfg.name}] cloaks: {len(r['missing_cloaks'])} missing of {r['cloaks_total']}; "
          f"swap rows: {len(r['missing_swaps'])} missing of {r['swaps_total']}")
    by_attack: dict[str, int] = {}
    for k in r["missing_cloaks"]:
        by_attack[k[0]] = by_attack.get(k[0], 0) + 1
    for k, v in by_attack.items():
        print(f"  cloaks missing for {k}: {v}")
    if a.list:
        for k in r["missing_cloaks"]:
            print("  cloak", *k)
        for k in r["missing_swaps"]:
            print("  swap ", *k)
    print("Recommended split (shards are over sources; --resume-from skips what is already done):")
    for spec, d in r["shards"].items():
        print(f"  --shard {spec}: {d['sources']} sources, {d['cloaks']} cloaks, {d['swaps']} swap rows left")
        print(f"    uv run python scripts/run_experiments.py --config {a.config} --shard {spec} "
              f"--resume-from <results.zip>")
    left = [d["cloaks"] for d in r["shards"].values()]
    if left and max(left) > 1.5 * max(1, min(left)):
        print("  note: remaining cloaks are unevenly spread across shards; the faster machine should take "
              "the heavier shard.")


if __name__ == "__main__":
    main()
