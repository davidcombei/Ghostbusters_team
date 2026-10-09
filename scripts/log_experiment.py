"""
Add a run's local_eval results to the experiment registry (docs/experiment_log.csv).
Backend of the /log-experiment skill; needs no GPU and no torch.

    python scripts/log_experiment.py --run_dir SLS_setup/exp/<run> --tag best_by_wf1 \
        --id L1-level-screen --change_vs_ref "B0 + level aug (A2.7a)" --ref B0-level-screen --dry_run

    python scripts/log_experiment.py --metrics SLS_setup/exp/<run>/local_eval/<tag>/metrics.json ...

    python scripts/log_experiment.py --backfill     # fill new columns of existing rows

Metrics come from metrics.json (written by scripts/local_eval.py). git_sha, seed and gpu_h come
from the training run dir (config.yaml, logs/train.log); owner defaults to the git email handle.
"""

import argparse
import json
import sys
from pathlib import Path

from registry import (DEFAULT_REGISTRY, REGISTRY_COLUMNS, REPORT_SETS, SLS_DIR, append_row,
                      build_row, gpu_hours, read_registry, read_run_config, run_dir_for,
                      owner_handle, write_registry)


def load_report(args):
    if args.metrics:
        path = Path(args.metrics)
    else:
        path = Path(args.run_dir) / "local_eval" / args.tag / "metrics.json"
    if not path.is_file():
        tags = sorted(p.parent.name for p in Path(args.run_dir).glob("local_eval/*/metrics.json")) \
            if args.run_dir else []
        sys.exit(f"No metrics.json at {path}" + (f" (available tags: {', '.join(tags)})" if tags else
                 " - run scripts/local_eval.py with the 17 report sets first"))
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def check_sets(report):
    names = list(report["full"]["sets"])
    missing = [n for n in REPORT_SETS if n not in names]
    extra = [n for n in names if n not in REPORT_SETS]
    if missing:
        print(f"WARNING: missing report sets (not comparable with other rows): {' '.join(missing)}")
    if extra:
        print(f"WARNING: sets outside the 17 report sets were scored: {' '.join(extra)}")
    return not missing and not extra


def compare_to_ref(row, ref_id, rows):
    ref = next((r for r in rows if r["id"] == ref_id), None)
    if ref is None:
        print(f"(no row with id '{ref_id}' to compare against)")
        return

    def delta(col):
        try:
            return f"{float(row[col]) - float(ref[col]):+.2f}"
        except ValueError:
            return "n/a"

    print(f"\nvs {ref_id}:  half_B WF1 {delta('half_b_wf1')}   full WF1 {delta('wf1@0.5')}   "
          f"clean F1 {delta('clean_f1')}   matched {delta('matched_f1')}   "
          f"heldout {delta('heldout_f1')}   echo {delta('echo_f1')}")
    print("Decision rule (next_steps_plan.md): promote only if the paired-bootstrap 95% CI of "
          "dWF1 (half_B) excludes 0 and clean F1 drops by <= 0.5; otherwise 'no effect'.")


def backfill(path):
    """Fill seed / half_b_wf1 / gpu_h of existing rows from the metrics.json with the same
    config_hash (any exp/*/local_eval/*/metrics.json). Never touches other columns."""
    reports = {}
    for p in SLS_DIR.glob("exp/*/local_eval/*/metrics.json"):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        reports.setdefault(r.get("config_hash"), r)
    rows = read_registry(path)
    for row in rows:
        report = reports.get(row["config_hash"])
        run_dir = run_dir_for(report["checkpoints"][0]) if report and report.get("checkpoints") else None
        filled = []
        wb = ((report or {}).get("half_B") or {}).get("wf1")
        if not row["half_b_wf1"] and wb:
            row["half_b_wf1"] = f"{100 * wb['wf1']:.2f}"
            filled.append("half_b_wf1")
        seed = read_run_config(run_dir).get("seed")
        if not row["seed"] and seed is not None:
            row["seed"] = str(seed)
            filled.append("seed")
        if not row["gpu_h"] and run_dir is not None and gpu_hours(run_dir):
            row["gpu_h"] = gpu_hours(run_dir)
            filled.append("gpu_h")
        print(f"{row['id'] or row['model']}: " + (", ".join(filled) if filled else "nothing found"))
    write_registry(path, rows)
    print(f"Rewrote {path} with {len(REGISTRY_COLUMNS)} columns")


def main():
    parser = argparse.ArgumentParser(description="Add a run's results to docs/experiment_log.csv")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--metrics", type=str, help="a local_eval metrics.json")
    src.add_argument("--run_dir", type=str, help="exp/<run>; uses local_eval/<tag>/metrics.json")
    parser.add_argument("--tag", type=str, default="best_by_wf1")
    parser.add_argument("--registry", type=str, default=str(DEFAULT_REGISTRY))
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--owner", type=str, default=None, help="default: git user.email handle")
    parser.add_argument("--change_vs_ref", type=str, default="")
    parser.add_argument("--ref", type=str, default=None, help="registry id to print deltas against")
    parser.add_argument("--decision", type=str, default="", choices=["", "promote", "drop", "revisit"])
    parser.add_argument("--progress_score", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--ci_low", type=str, default="")
    parser.add_argument("--ci_high", type=str, default="")
    parser.add_argument("--replace", action="store_true", help="overwrite the row with the same id")
    parser.add_argument("--dry_run", action="store_true", help="print the row, write nothing")
    parser.add_argument("--backfill", action="store_true", help="fill new columns of existing rows")
    args = parser.parse_args()

    if args.backfill:
        backfill(args.registry)
        return
    if not (args.metrics or args.run_dir):
        parser.error("one of --metrics / --run_dir is required")
    if not args.id:
        parser.error("--id is required")

    report, metrics_path = load_report(args)
    print(f"metrics: {metrics_path}")
    check_sets(report)
    row = build_row(report, id=args.id, owner=args.owner or owner_handle(),
                    change_vs_ref=args.change_vs_ref, notes=args.notes, decision=args.decision,
                    progress_score=args.progress_score, ci_low=args.ci_low, ci_high=args.ci_high,
                    run_dir=Path(args.run_dir).resolve() if args.run_dir else None)
    width = max(map(len, REGISTRY_COLUMNS))
    print()
    for c in REGISTRY_COLUMNS:
        print(f"  {c:<{width}}  {row[c]}")
    if args.ref:
        compare_to_ref(row, args.ref, read_registry(args.registry))

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return
    try:
        action = append_row(args.registry, row, replace=args.replace)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"\nRow '{row['id']}' {action} in {args.registry}")


if __name__ == "__main__":
    main()
