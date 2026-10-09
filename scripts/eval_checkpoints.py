"""
Score every checkpoint of one training run on the local proxy sets and rank them by WF1 -- for
runs trained before (or without) main_train.py --local_eval_sets.

    cd SLS_setup
    python ../scripts/eval_checkpoints.py exp/<run>                      # model from exp/<run>/config.yaml
    python ../scripts/eval_checkpoints.py exp/<legacy_run> --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24
    python ../scripts/eval_checkpoints.py exp/<run> --every 5            # or --epochs 8 35 40

Writes under exp/<run>/local_eval/ckpts/:

    <tag>/scores/<set>.txt|.logits   per-checkpoint score cache, same format as scripts/local_eval.py
    <tag>/metrics.json               scripts/local_eval.py's report, plus a half_A block
    summary.csv                      one row per checkpoint: {split}.wf1, {split}.f1_clean, ... and
                                     {split}.{set}.{f1,eer,...} in percent, split in full/half_A/half_B;
                                     rewritten after every checkpoint
    best.json                        top-k by --select_on (default half_A.wf1), with their half_B.wf1

The sets and the model are loaded once; only the weights change between checkpoints. Cached
scores are reused, so an interrupted run resumes where it stopped, and the summary also lists
every earlier-scored checkpoint whose cache covers the requested sets. Select on half A, report
half B: half B is never used for picking.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLS_DIR = REPO_ROOT / "SLS_setup"
sys.path.insert(0, str(SLS_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from local_eval import BREAKDOWN_KEYS, read_scores, write_scores  # noqa: E402  (scripts/local_eval.py)
from model.sls_model import ModelSLS, ssl_path  # noqa: E402
from utils import metrics  # noqa: E402
from utils.data_utils import set_random_seed  # noqa: E402
from utils.data_utils import build_loudness  # noqa: E402
from utils.local_eval import evaluate_sets, load_sets, score_dataset  # noqa: E402

DEFAULT_NAMES = ["dev_online_clean", "sim_matched_v1", "sim_heldout_v1", "sim_echo_v1"]
CKPT_RE = re.compile(r"epoch_(\d+)_dev_loss_([0-9.]+)\.pth$")
SPLITS = {"full": None, "half_A": "A", "half_B": "B"}
WF1_KEYS = ("wf1", "wf1_oracle", "f1_clean", "f1_noisy", "eer_noisy",
            "calibration_gap_noisy", "separability_gap")
SET_KEYS = ("f1", "f1_oracle", "eer", "recall_spoof", "recall_bona", "ece")
LOWER_IS_BETTER = ("dev_loss", "eer", "eer_noisy", "ece", "calibration_gap_noisy", "separability_gap")


# --------------------------------------------------------------------------- #
# run discovery
# --------------------------------------------------------------------------- #
def find_checkpoints(ckpt_dir):
    """Regular *.pth files in ckpt/, sorted by epoch; symlinks become aliases of their target."""
    ckpts, aliases = {}, {}
    for p in sorted(ckpt_dir.glob("*.pth")):
        if p.is_symlink():
            aliases.setdefault(p.resolve().name, []).append(p.stem)
            continue
        m = CKPT_RE.match(p.name)
        tag = f"epoch_{m[1]}" if m else p.stem
        if any(c["tag"] == tag for c in ckpts.values()):
            tag = p.stem
        ckpts[p.name] = {"path": p, "tag": tag, "epoch": int(m[1]) if m else None,
                         "dev_loss": float(m[2].rstrip(".")) if m else None, "aliases": []}
    for name, names in aliases.items():
        if name in ckpts:
            ckpts[name]["aliases"] = sorted(names)
    return sorted(ckpts.values(), key=lambda c: (c["epoch"] is None, c["epoch"] or 0, c["tag"]))


def resolve_model_args(args, exp):
    """CLI flags win; otherwise the run's config.yaml (written by main_train.py)."""
    cfg_path = exp / "config.yaml"
    cfg = {}
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = (yaml.safe_load(f) or {}).get("args", {}) or {}
    for key in ("ssl_name", "n_layers", "devices", "layer_split"):
        if getattr(args, key) is None:
            setattr(args, key, cfg.get(key))
    # loudness normalisation must match training: config.yaml unless --loudness_norm / --no_loudness_norm
    if args.loudness_norm is None:
        args.loudness_norm = bool(cfg.get("loudness_norm", False))
    if args.loudness_norm_lufs is None:
        args.loudness_norm_lufs = float(cfg.get("loudness_norm_lufs", -23.0))
    if args.ssl_name is None or args.n_layers is None:
        raise SystemExit(f"{exp} has no config.yaml with ssl_name / n_layers: pass --ssl_name and "
                         f"--n_layers (e.g. --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24)")
    args.ssl_name = ssl_path(args.ssl_name)


# --------------------------------------------------------------------------- #
# scoring and metrics
# --------------------------------------------------------------------------- #
def cached_scores(out, sets):
    """{set: p_spoof} for the sets whose cache matches the current protocol."""
    scores = {}
    for name, s in sets.items():
        path = out / "scores" / f"{name}.txt"
        if path.is_file():
            ids, p = read_scores(path)
            if ids == s["file_list"]:
                scores[name] = p
    return scores


def has_half(s):
    return any(r.get("half") for r in s["meta_rows"])


def evaluate_split(sets, scores, half):
    """evaluate_sets, minus the sets that carry a half column but have no rows in that half."""
    if half is not None:
        sets = {n: s for n, s in sets.items()
                if not has_half(s) or any(r.get("half") == half for r in s["meta_rows"])}
    return evaluate_sets(sets, scores, half=half)


def build_report(ckpt, sets, scores, args):
    report = {"model": str(ckpt["path"]), "checkpoints": [str(ckpt["path"])], "epoch": ckpt["epoch"],
              "dev_loss": ckpt["dev_loss"], "aliases": ckpt["aliases"], "ssl_name": args.ssl_name,
              "n_layers": args.n_layers, "sets_file": args.sets}
    for split, half in SPLITS.items():
        report[split] = evaluate_split(sets, scores, half)
    report["breakdowns"] = {}
    for name, s in sets.items():
        keys = [k for k in BREAKDOWN_KEYS if any(r.get(k) for r in s["meta_rows"])]
        if keys:
            report["breakdowns"][name] = {k: metrics.breakdown(s["meta_rows"], s["y_spoof"],
                                                               scores[name], k) for k in keys}
    return report


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return round(100 * float(x), 2)


def summary_row(ckpt, report, sets):
    row = {"ckpt": ckpt["path"].name, "epoch": "" if ckpt["epoch"] is None else ckpt["epoch"],
           "dev_loss": "" if ckpt["dev_loss"] is None else ckpt["dev_loss"],
           "aliases": " ".join(ckpt["aliases"])}
    for split in SPLITS:
        w = report[split]["wf1"]
        for k in WF1_KEYS:
            row[f"{split}.{k}"] = pct(w.get(k))
    for split in SPLITS:
        for name, s in report[split]["sets"].items():
            if split == "full" or has_half(sets[name]):
                for k in SET_KEYS:
                    row[f"{split}.{name}.{k}"] = pct(s.get(k))
    return row


def write_summary(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def rank(rows, key):
    rows = [r for r in rows if r.get(key) not in ("", None)]
    lower = key == "dev_loss" or key.rsplit(".", 1)[-1] in LOWER_IS_BETTER
    return sorted(rows, key=lambda r: r[key], reverse=not lower)


def write_best(path, rows, select_on, top_k):
    def brief(r):
        keys = ("ckpt", "epoch", "dev_loss", "aliases", select_on, "full.wf1", "half_B.wf1",
                "half_B.f1_clean", "half_B.f1_noisy")
        return {k: r.get(k, "") for k in dict.fromkeys(keys)}

    best = {"select_on": select_on, "n_checkpoints": len(rows),
            "top": [brief(r) for r in rank(rows, select_on)[:top_k]]}
    for key in ("full.wf1", "dev_loss"):
        ranked = rank(rows, key)
        best[f"best_by_{key}"] = brief(ranked[0]) if ranked else None
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def print_best(best):
    sel = best["select_on"]
    print(f"\nTop {len(best['top'])} of {best['n_checkpoints']} checkpoints by {sel}:")
    print(f"   {'ckpt':<34} {sel:>12} {'full.wf1':>9} {'half_B.wf1':>11}  aliases")
    for r in best["top"]:
        print(f"   {r['ckpt']:<34} {r[sel]:>12} {r['full.wf1']:>9} {r['half_B.wf1']:>11}  {r['aliases']}")
    for key in ("full.wf1", "dev_loss"):
        r = best[f"best_by_{key}"]
        if r:
            print(f"best by {key:<9} {r['ckpt']:<34} half_B.wf1 = {r['half_B.wf1']}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp", type=str, help="experiment folder (with a ckpt/ subfolder)")
    parser.add_argument("--sets", type=str, default=str(REPO_ROOT / "configs/local_eval_sets.yaml"))
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    parser.add_argument("--epochs", type=int, nargs="+", default=None, help="only these epochs")
    parser.add_argument("--every", type=int, default=None, help="only epochs divisible by N")
    parser.add_argument("--out", type=str, default=None, help="default: <exp>/local_eval/ckpts")
    parser.add_argument("--select_on", type=str, default="half_A.wf1",
                        help="summary.csv column to rank by (half_A.wf1, full.wf1, dev_loss, ...)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--ssl_name", type=str, default=None, help="default: from <exp>/config.yaml")
    parser.add_argument("--n_layers", type=int, default=None, help="default: from <exp>/config.yaml")
    parser.add_argument("--devices", type=int, nargs="+", default=None)
    parser.add_argument("--layer_split", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="rescore sets that are already cached")
    parser.add_argument("--loudness_norm", dest="loudness_norm", action="store_true", default=None,
                        help="read-time loudness normalisation (default: as in <exp>/config.yaml)")
    parser.add_argument("--no_loudness_norm", dest="loudness_norm", action="store_false", default=None,
                        help="force it off even if the run was trained with it")
    parser.add_argument("--loudness_norm_lufs", type=float, default=None,
                        help="target LUFS (default: from <exp>/config.yaml, else -23)")
    args = parser.parse_args()

    exp = Path(args.exp)
    if not (exp / "ckpt").is_dir() and (SLS_DIR / exp / "ckpt").is_dir():
        exp = SLS_DIR / exp
    exp = exp.resolve()
    if not (exp / "ckpt").is_dir():
        raise SystemExit(f"no ckpt/ folder under {exp}")
    minis = [n for n in args.names if n.endswith("_mini")]
    if minis:
        raise SystemExit(f"{minis} are half-A subsets of the *_v1 sets (per-epoch selection in "
                         f"main_train.py); they would double-count in the noisy WF1 -- drop them")
    resolve_model_args(args, exp)
    args.cudnn_deterministic_toggle, args.cudnn_benchmark_toggle = True, False
    set_random_seed(args.seed, args)

    ckpts = find_checkpoints(exp / "ckpt")
    if not ckpts:
        raise SystemExit(f"no checkpoints in {exp / 'ckpt'}")
    selected = [c for c in ckpts
                if (args.epochs is None or c["epoch"] in args.epochs)
                and (args.every is None or (c["epoch"] is not None and c["epoch"] % args.every == 0))]
    out_root = Path(args.out) if args.out else exp / "local_eval" / "ckpts"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{exp.name}: {len(ckpts)} checkpoints, {len(selected)} selected | {args.ssl_name}, "
          f"{args.n_layers} layers | sets {args.names} -> {out_root}", flush=True)

    sets = load_sets(args.sets, args.names, ssl_name=args.ssl_name, loudness=build_loudness(args))
    if args.select_on != "dev_loss" and args.select_on.split(".", 1)[0] not in SPLITS:
        raise SystemExit(f"--select_on {args.select_on}: expected <split>.<metric>, split in {list(SPLITS)}")

    model, device, rows = None, None, []
    try:
        for i, ckpt in enumerate(ckpts, 1):
            out = out_root / ckpt["tag"]
            scores = {} if args.force and ckpt in selected else cached_scores(out, sets)
            missing = [n for n in sets if n not in scores]
            if missing and ckpt not in selected:
                continue                      # not asked for and not fully cached
            if missing:
                if model is None:
                    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
                    model = ModelSLS(args, device)
                    device = model.input_device
                state = torch.load(ckpt["path"], map_location="cpu")
                model.load_state_dict(state)
                del state
                (out / "scores").mkdir(parents=True, exist_ok=True)
                for name in missing:
                    ids, p, d = score_dataset(model, sets[name]["dataset"], device,
                                              args.batch_size, args.num_workers)
                    write_scores(out / "scores" / f"{name}.txt", ids, p)
                    write_scores(out / "scores" / f"{name}.logits", ids, d)
                    scores[name] = p

            report = build_report(ckpt, sets, scores, args)
            with open(out / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            row = summary_row(ckpt, report, sets)
            rows.append(row)
            write_summary(out_root / "summary.csv", rows)
            print(f"[{i}/{len(ckpts)}] {ckpt['tag']:<12} WF1 full {row['full.wf1']} | A {row['half_A.wf1']} "
                  f"| B {row['half_B.wf1']}   (B: clean {row['half_B.f1_clean']}, noisy {row['half_B.f1_noisy']})"
                  f"{'  [scored ' + ','.join(missing) + ']' if missing else '  [cached]'}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; summarising the checkpoints done so far")

    if not rows:
        raise SystemExit("no checkpoint scored")
    if args.select_on not in rows[0]:
        raise SystemExit(f"--select_on {args.select_on} is not a summary.csv column")
    best = write_best(out_root / "best.json", rows, args.select_on, args.top_k)
    print_best(best)
    print(f"\nSummary: {out_root / 'summary.csv'}\nBest:    {out_root / 'best.json'}")


if __name__ == "__main__":
    main()
