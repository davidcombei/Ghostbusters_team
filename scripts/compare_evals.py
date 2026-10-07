"""
Side-by-side comparison of local evaluations (scripts/local_eval.py outputs) against a baseline.

    python scripts/compare_evals.py <baseline> <candidate> [<candidate> ...] [--labels B0 B1]

Each argument is a local_eval output dir (exp/<run>/local_eval/<tag>, holding metrics.json and
optionally probe_report.json), or that metrics.json itself. The first one is the baseline; every
other column shows its value and the delta against it. Paths may be relative to the repo root or
to SLS_setup/.

Reads half B by default (--split full for the full sets): half A is used for epoch selection
during training, so half B is the clean readout (docs/dev_splits.md). Reported per model:
WF1 block (wf1, wf1_oracle, f1_clean, f1_noisy, eer_noisy), F1 / EER / spoof and bonafide recall
per non-probe set, and the probe summary when probe_report.json exists (scripts/probe_report.py).
Numbers are in percent.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLS_DIR = REPO_ROOT / "SLS_setup"

WF1_ROWS = ("wf1", "wf1_oracle", "f1_clean", "f1_noisy", "eer_noisy")
LOWER_IS_BETTER = {"eer_noisy", "eer", "level_flip_bona", "level_flip_spoof"}
SET_ROWS = ("f1", "eer", "recall_spoof", "recall_bona")
# (probe, label) rows from probe_report.json; for bonafide a lower predicted-spoof share is better
PROBE_ROWS = (("sil_only", "bonafide"), ("sil_only", "spoof"),
              ("noise_only", "bonafide"), ("noise_only", "spoof"))


def resolve(path):
    p = Path(path)
    for cand in (p, REPO_ROOT / p, SLS_DIR / p):
        if cand.is_file() and cand.name == "metrics.json":
            return cand
        if (cand / "metrics.json").is_file():
            return cand / "metrics.json"
    run_dirs = [c for c in (p, REPO_ROOT / p, SLS_DIR / p) if (c / "local_eval").is_dir()]
    if run_dirs:
        tags = sorted(m.parent.name for m in (run_dirs[0] / "local_eval").glob("*/metrics.json"))
        sys.exit(f"{path} is a run dir; pass one of its local_eval/<tag> dirs: {tags}")
    sys.exit(f"no metrics.json at {path}")


def load(path):
    metrics_path = resolve(path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    probe_path = metrics_path.parent / "probe_report.json"
    probes = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else None
    return metrics_path, metrics, probes


def default_label(metrics_path):
    tag = metrics_path.parent.name
    run = metrics_path.parent.parent.parent.name
    return f"{run.split('_')[0]}/{tag}"


def fmt(value, base=None, lower=False):
    if value is None:
        return f"{'-':>8}{'':>8}"
    cell = f"{100 * value:8.2f}"
    if base is None:
        return cell + f"{'':>8}"
    delta = 100 * (value - base)
    better = delta < 0 if lower else delta > 0
    mark = "+" if abs(delta) >= 0.005 and better else ("-" if abs(delta) >= 0.005 else " ")
    return cell + f" {delta:+6.2f}{mark}"


def print_rows(title, keys, getters, labels):
    print(f"\n{title}")
    print(f"{'':28}" + "".join(f"{lab[:15]:>16}" for lab in labels))
    for key, name, lower in keys:
        values = [g(key) for g in getters]
        base = values[0]
        cells = [fmt(values[0])] + [fmt(v, base if base is not None and v is not None else None, lower)
                                    for v in values[1:]]
        print(f"{name:28}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evals", nargs="+", help="baseline first, then candidates")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--split", default="half_B", choices=["half_B", "full"])
    args = parser.parse_args()
    if len(args.evals) < 2:
        sys.exit("need a baseline and at least one candidate")

    loaded = [load(p) for p in args.evals]
    labels = args.labels or [default_label(m) for m, _, _ in loaded]
    if len(labels) != len(loaded):
        sys.exit("--labels needs one label per eval")

    for (path, metrics, _), lab in zip(loaded, labels):
        if args.split not in metrics:
            sys.exit(f"{path} has no '{args.split}' block (scored on an older set list?); rerun local_eval")
        ckpts = metrics.get("checkpoints") or [metrics.get("model")]
        print(f"{lab:16} {path.parent}  ({len(ckpts)} checkpoint{'s' if len(ckpts) != 1 else ''})")

    blocks = [m[args.split] for _, m, _ in loaded]
    print_rows(f"[{args.split}] weighted F1 (first column = baseline; +/- = better/worse)",
               [(k, k, k in LOWER_IS_BETTER) for k in WF1_ROWS], [lambda k, b=b: b["wf1"].get(k) for b in blocks], labels)

    set_names = [n for n, s in blocks[0]["sets"].items() if s.get("role") != "probe"]
    missing = {lab: [n for n in set_names if n not in b["sets"]] for lab, b in zip(labels, blocks)}
    for lab, names in missing.items():
        if names:
            print(f"\nWARNING: {lab} lacks sets {names}: its WF1 is not comparable with the baseline")
    for name in set_names:
        print_rows(f"[{args.split}] {name}", [(k, k, k in LOWER_IS_BETTER) for k in SET_ROWS],
                   [lambda k, b=b, n=name: b["sets"].get(n, {}).get(k) for b in blocks], labels)

    probes = [p for _, _, p in loaded]
    if any(probes):
        def summary(p, k):
            return p["summary"].get(k) if p else None

        def pred_spoof(p, key):
            probe, label = key
            if not p:
                return None
            rows = [r for r in p["rows"] if r["probe"] == probe and r["label"] == label]
            return rows[0]["pred_spoof"] if rows else None

        print_rows("probes (full set; lower flip = less level shortcut)",
                   [(k, k, True) for k in ("level_flip_bona", "level_flip_spoof")],
                   [lambda k, p=p: summary(p, k) for p in probes], labels)
        print_rows("probes: share predicted spoof (bona should be low, spoof high)",
                   [(k, f"{k[0]} {k[1]}", k[1] == "bonafide") for k in PROBE_ROWS],
                   [lambda k, p=p: pred_spoof(p, k) for p in probes], labels)
        absent = [lab for lab, p in zip(labels, probes) if not p]
        if absent:
            print(f"(no probe_report.json for {absent}; run scripts/probe_report.py <eval>/scores)")


if __name__ == "__main__":
    main()
