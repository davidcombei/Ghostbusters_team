"""
Pick epochs of one training run for a weight-averaged merge, from its per-epoch local metrics
(exp/<run>/metrics.jsonl, written by main_train.py --local_eval_sets), and write a merge list
for scripts/local_eval.py --merge_list / main_eval.py --model_merging --merge_list.

    cd SLS_setup
    python ../scripts/select_topk.py --run exp/<run> --k 5                    # top-5 by local WF1
    python ../scripts/select_topk.py --run exp/<run> --epochs 1 8 13 18 23    # fixed epochs

The ranking uses the half-A selection sets only (dev_online_clean subsample, sim_matched_mini,
sim_heldout_mini), so half B stays clean for the final local_eval readout. Writes
exp/<run>/top<k>_by_<metric>/merge_list.txt or exp/<run>/epochs_<e1>_<e2>.../merge_list.txt.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLS_DIR = REPO_ROOT / "SLS_setup"

# metric -> True if higher is better (keys of the "wf1" block of a metrics.jsonl row)
METRICS = {"wf1": True, "f1_noisy": True, "wf1_oracle": True, "eer_noisy": False}


def read_metrics(run):
    """{epoch: row}; a later row for the same epoch wins (as in main_train.read_metrics)."""
    rows = {}
    with open(run / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["epoch"]] = row
    return rows


def list_path(ckpt):
    """Merge-list entry: relative to SLS_setup/ when possible (what read_merge_list expects)."""
    ckpt = Path(os.path.abspath(ckpt))          # no resolve(): keep symlinked exp/ dirs relative
    try:
        return str(ckpt.relative_to(SLS_DIR))
    except ValueError:
        return str(ckpt)


def main():
    parser = argparse.ArgumentParser(description="Select epochs of a run for a weight merge")
    parser.add_argument("--run", type=str, required=True, help="exp/<run> dir with metrics.jsonl")
    pick = parser.add_mutually_exclusive_group()
    pick.add_argument("--k", type=int, default=5, help="top-k epochs by --metric")
    pick.add_argument("--epochs", type=int, nargs="+", help="fixed epochs instead of a ranking")
    parser.add_argument("--metric", type=str, default="wf1", choices=sorted(METRICS))
    parser.add_argument("--out", type=str, default=None,
                        help="merge list path (default: <run>/<selection>/merge_list.txt)")
    args = parser.parse_args()

    run = Path(args.run)
    rows = read_metrics(run)
    if not rows:
        sys.exit(f"no rows in {run / 'metrics.jsonl'}")

    higher = METRICS[args.metric]
    ranked = sorted(rows, key=lambda e: (-rows[e]["wf1"][args.metric] if higher
                                         else rows[e]["wf1"][args.metric], e))
    if args.epochs:
        missing = [e for e in args.epochs if e not in rows]
        if missing:
            sys.exit(f"epochs {missing} have no metrics row (run has epochs "
                     f"{min(rows)}..{max(rows)})")
        chosen = list(args.epochs)
        tag = "epochs_" + "_".join(map(str, chosen))
    else:
        chosen = ranked[:args.k]
        tag = f"top{args.k}_by_{args.metric}"

    rank = {e: i + 1 for i, e in enumerate(ranked)}
    print(f"{'':2}{'epoch':>5} {'rank':>4} {'wf1':>6} {'f1_cln':>6} {'f1_nsy':>6} "
          f"{'wf1_or':>6} {'eer_ns':>6} {'dev_loss':>8}  ckpt")
    for e in sorted(rows):
        r, w = rows[e], rows[e]["wf1"]
        exists = (run / "ckpt" / r["ckpt"]).is_file()
        print(f"{'*' if e in chosen else ' ':2}{e:>5} {rank[e]:>4} {100 * w['wf1']:6.2f} "
              f"{100 * w['f1_clean']:6.2f} {100 * w['f1_noisy']:6.2f} {100 * w['wf1_oracle']:6.2f} "
              f"{100 * w['eer_noisy']:6.2f} {r['dev_loss']:8.5f}  "
              f"{r['ckpt'] if exists else '(deleted)'}")

    ckpts = [run / "ckpt" / rows[e]["ckpt"] for e in chosen]
    missing = [str(c) for c in ckpts if not c.is_file()]
    if missing:
        sys.exit("checkpoints missing (pruned or deleted):\n  " + "\n  ".join(missing))

    out = Path(args.out) if args.out else run / tag / "merge_list.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(list_path(c) + "\n" for c in ckpts), encoding="utf-8")
    mean = sum(rows[e]["wf1"][args.metric] for e in chosen) / len(chosen)
    print(f"\n{tag}: epochs {chosen} (mean per-epoch {args.metric} {100 * mean:.2f})")
    print(f"Merge list: {out}")


if __name__ == "__main__":
    main()
