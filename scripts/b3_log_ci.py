#!/usr/bin/env python
"""
[B3 wave] Fill the ci_low / ci_high columns of a docs/experiment_log.csv row from a
compare_ci.json written by scripts/b3_compare_evals.py --ci.

    python scripts/b3_log_ci.py --id B3a-loud23-ep8 --ci_json SLS_setup/exp/<run>/local_eval/epoch_8/compare_ci.json

The values are the 95 % CI of the half_B WF1@0.5 delta (candidate minus baseline), in percent.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from registry import DEFAULT_REGISTRY, read_registry, write_registry  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True)
    parser.add_argument("--ci_json", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args()
    res = json.loads(Path(args.ci_json).read_text(encoding="utf-8"))
    if not res.get("wf1"):
        sys.exit("no wf1 block in the CI json (clean + matched + heldout sets needed)")
    lo, hi = 100 * res["wf1"]["ci_low"], 100 * res["wf1"]["ci_high"]
    rows = read_registry(args.registry)
    hits = [r for r in rows if r["id"] == args.id]
    if not hits:
        sys.exit(f"id {args.id} not in {args.registry}")
    hits[0]["ci_low"], hits[0]["ci_high"] = f"{lo:.2f}", f"{hi:.2f}"
    write_registry(args.registry, rows)
    print(f"{args.id}: ci_low={lo:.2f} ci_high={hi:.2f} (delta {100 * res['wf1']['delta']:+.2f} vs "
          f"{res.get('labels', {}).get('a', 'baseline')}) written to {args.registry}")


if __name__ == "__main__":
    main()
