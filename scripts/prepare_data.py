"""Download LFW (if needed), run detection filtering and write the evaluation manifest."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from facecloak.data import DEFAULT_MANIFEST, build_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-ids", type=int, default=150)
    ap.add_argument("--n-targets", type=int, default=30)
    ap.add_argument("--targets-per-src", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lfw-root", type=Path, default=None, help="default: $FACECLOAK_LFW or data/lfw")
    ap.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    a = ap.parse_args()
    t0 = time.time()
    m = build_manifest(a.n_ids, a.n_targets, a.targets_per_src, a.seed, a.lfw_root, a.out)
    st = m["stats"]
    print(f"manifest -> {a.out}")
    print(f"sources={len(m['sources'])} targets={len(m['targets'])} pairs={len(m['pairs'])}")
    print(f"source candidates checked={st['src_candidates_checked']} "
          f"detection failures={st['src_det_failures']}")
    print(f"target candidates checked={st['tgt_candidates_checked']} "
          f"detection failures={st['tgt_det_failures']}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
