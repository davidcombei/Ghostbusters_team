#!/usr/bin/env python
"""
[B3 wave] scripts/compare_evals.py plus a paired-bootstrap CI block (b3wave/metrics_ci.py).

    python scripts/b3_compare_evals.py <baseline> <candidate> [...] [--labels A B] [--ci 2000]

Everything compare_evals.py prints is printed unchanged; with --ci N each candidate is then
bootstrapped against the baseline on the cached scores (half B, the four report sets) and the
delta's 95 % CI is printed and written to <candidate>/compare_ci.json. Both evals must hold
scores/ for the four report sets.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "SLS_setup"))

import compare_evals  # noqa: E402
from b3wave.metrics_ci import REPORT_SETS, format_ci, paired_bootstrap  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evals", nargs="+")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--split", default="half_B", choices=["half_B", "full"])
    parser.add_argument("--ci", type=int, default=0, help="bootstrap resamples (0 = off)")
    parser.add_argument("--sets", default=str(REPO_ROOT / "configs/local_eval_sets.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    passthrough = ["compare_evals.py", *args.evals, "--split", args.split]
    if args.labels:
        passthrough += ["--labels", *args.labels]
    sys.argv = passthrough
    compare_evals.main()

    if not args.ci:
        return
    labels = args.labels or [f"eval{i}" for i in range(len(args.evals))]
    base = compare_evals.resolve(args.evals[0]).parent
    half = "B" if args.split == "half_B" else None
    for cand, lab in zip(args.evals[1:], labels[1:]):
        cand_dir = compare_evals.resolve(cand).parent
        res = paired_bootstrap(base, cand_dir, args.sets, REPORT_SETS, half=half, n=args.ci, seed=args.seed)
        print()
        print(format_ci(res, labels[0], lab))
        res["labels"] = {"a": labels[0], "b": lab}
        out = cand_dir / "compare_ci.json"
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
