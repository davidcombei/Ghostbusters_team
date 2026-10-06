"""
Full local report for one model (a checkpoint or a weight-averaged merge) on the proxy sets.

    cd SLS_setup
    python ../scripts/local_eval.py --model_path exp/<run>/ckpt/epoch_8_dev_loss_0.088333.pth \
        --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 --out exp/<run>/local_eval/epoch_8 \
        --registry ../docs/experiment_log.csv --id REF-0

    python ../scripts/local_eval.py --merge_list exp/REF-0/merge_list.txt ...

Scores are cached per set in <out>/scores/<set>.txt ("utt p_spoof") and <set>.logits
("utt d", d = z_spoof - z_bona, for later calibration and soups); rerunning only scores
missing sets unless --force. Metrics go to <out>/metrics.json and stdout.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLS_DIR = REPO_ROOT / "SLS_setup"
sys.path.insert(0, str(SLS_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from main_eval import merge_checkpoints  # noqa: E402
from model.sls_model import ModelSLS, ssl_path  # noqa: E402
from utils import metrics  # noqa: E402
from utils.data_utils import set_random_seed  # noqa: E402
from utils.local_eval import evaluate_sets, load_sets, score_dataset  # noqa: E402

BREAKDOWN_KEYS = ("family", "snr_db", "codec", "src", "lang", "aec")
REGISTRY_COLUMNS = [
    "id", "date", "owner", "git_sha", "config_hash", "model", "change_vs_ref",
    "clean_f1", "matched_f1", "heldout_f1", "echo_f1", "wf1@0.5", "wf1@oracle",
    "eer_clean", "eer_matched", "eer_heldout", "eer_echo",
    "recall_spoof_noisy", "recall_bona_noisy", "progress_score", "gpu_h", "decision", "notes",
]


def resolve_ckpt(path):
    """Checkpoint paths in merge lists are usually relative to SLS_setup/."""
    p = Path(path)
    if p.is_file():
        return p
    if (SLS_DIR / p).is_file():
        return SLS_DIR / p
    raise FileNotFoundError(path)


def read_merge_list(path):
    with open(path, encoding="utf-8") as f:
        return [str(resolve_ckpt(line.strip())) for line in f
                if line.strip() and not line.lstrip().startswith("#")]


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def read_scores(path):
    with open(path, encoding="utf-8") as f:
        rows = [line.split() for line in f if line.strip()]
    return [r[0] for r in rows], np.array([float(r[1]) for r in rows])


def write_scores(path, ids, values):
    with open(path, "w", encoding="utf-8") as f:
        for u, v in zip(ids, values):
            f.write(f"{u} {v:.10f}\n")


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "   -  "
    return f"{100 * x:6.2f}" if pct else f"{x:6.3f}"


def print_report(report):
    print(f"\n{'set':<26} {'role':<8} {'n':>6} {'F1@.5':>6} {'F1@or':>6} {'thr_or':>6} "
          f"{'EER':>6} {'ECE':>6} {'R_spf':>6} {'R_bon':>6}")
    for name, s in report["full"]["sets"].items():
        print(f"{name:<26} {s['role']:<8} {s['n']:>6} {fmt(s['f1'])} {fmt(s['f1_oracle'])} "
              f"{fmt(s['thr_oracle'], False)} {fmt(s['eer'])} {fmt(s['ece'])} "
              f"{fmt(s['recall_spoof'])} {fmt(s['recall_bona'])}")
    for tag in ("full", "half_B"):
        w = report[tag]["wf1"]
        if w:
            print(f"\n[{tag}] WF1_local@0.5 = {fmt(w['wf1'])}  (clean {fmt(w['f1_clean'])}, "
                  f"noisy {fmt(w['f1_noisy'])})   WF1@oracle = {fmt(w['wf1_oracle'])}")
            print(f"[{tag}] calibration gap (noisy) = {fmt(w['calibration_gap_noisy'])}   "
                  f"separability gap = {fmt(w['separability_gap'])}   "
                  f"mean noisy EER = {fmt(w.get('eer_noisy'))}")
    for name, by_key in report.get("breakdowns", {}).items():
        for key, groups in by_key.items():
            print(f"\n{name} by {key}:")
            for value, s in groups.items():
                print(f"   {value:<22} n={s['n']:>5} F1={fmt(s['f1'])} EER={fmt(s['eer'])} "
                      f"R_spf={fmt(s['recall_spoof'])} R_bon={fmt(s['recall_bona'])}")


def append_registry(path, report, args, model_desc, config_hash):
    sets = report["full"]["sets"]

    def by_role(role, key):
        vals = [s[key] for s in sets.values() if s["role"] == role]
        return f"{100 * vals[0]:.2f}" if vals else ""

    noisy = [s for s in sets.values() if s["role"] in ("matched", "heldout")]
    w = report["full"]["wf1"]
    row = {
        "id": args.id or "", "date": date.today().isoformat(), "owner": args.owner or "",
        "git_sha": git_sha(), "config_hash": config_hash, "model": model_desc,
        "change_vs_ref": args.change_vs_ref or "",
        "clean_f1": by_role("clean", "f1"), "matched_f1": by_role("matched", "f1"),
        "heldout_f1": by_role("heldout", "f1"), "echo_f1": by_role("echo", "f1"),
        "wf1@0.5": f"{100 * w['wf1']:.2f}" if w else "",
        "wf1@oracle": f"{100 * w['wf1_oracle']:.2f}" if w else "",
        "eer_clean": by_role("clean", "eer"), "eer_matched": by_role("matched", "eer"),
        "eer_heldout": by_role("heldout", "eer"), "eer_echo": by_role("echo", "eer"),
        "recall_spoof_noisy": f"{100 * np.mean([s['recall_spoof'] for s in noisy]):.2f}" if noisy else "",
        "recall_bona_noisy": f"{100 * np.mean([s['recall_bona'] for s in noisy]):.2f}" if noisy else "",
        "progress_score": "", "gpu_h": "", "decision": "", "notes": args.notes or "",
    }
    path = Path(path)
    new = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(f"Registry row appended to {path}")


def main():
    parser = argparse.ArgumentParser(description="Score one model on the local proxy sets")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model_path", type=str)
    src.add_argument("--merge_list", type=str, help="text file, one checkpoint path per line")
    parser.add_argument("--sets", type=str, default=str(REPO_ROOT / "configs/local_eval_sets.yaml"))
    parser.add_argument("--names", nargs="*", default=None, help="subset of sets (default: all)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ssl_name", type=str, default="facebook/wav2vec2-xls-r-2b")
    parser.add_argument("--n_layers", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=int, nargs="+", default=None)
    parser.add_argument("--layer_split", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="rescore sets that are already cached")
    parser.add_argument("--registry", type=str, default=None)
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--owner", type=str, default=None)
    parser.add_argument("--change_vs_ref", type=str, default=None)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    args.ssl_name = ssl_path(args.ssl_name)
    args.cudnn_deterministic_toggle, args.cudnn_benchmark_toggle = True, False
    set_random_seed(args.seed, args)

    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    sets = load_sets(args.sets, args.names, ssl_name=args.ssl_name)

    if args.merge_list:
        paths = read_merge_list(args.merge_list)
        (out / "merge_list.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        model_desc = f"merge:{args.merge_list}"
    else:
        paths = [str(resolve_ckpt(args.model_path))]
        model_desc = args.model_path
    config_hash = hashlib.sha1(json.dumps(
        {"ckpts": paths, "ssl": args.ssl_name, "n_layers": args.n_layers}).encode()).hexdigest()[:10]

    scores, model, device = {}, None, None
    for name, s in sets.items():
        score_path, logit_path = out / "scores" / f"{name}.txt", out / "scores" / f"{name}.logits"
        if score_path.is_file() and not args.force:
            ids, p = read_scores(score_path)
            if ids == s["file_list"]:
                print(f"[{name}] cached scores: {score_path}")
                scores[name] = p
                continue
        if model is None:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            model = ModelSLS(args, device)
            device = model.input_device
            state = merge_checkpoints(paths) if args.merge_list else torch.load(paths[0], map_location="cpu")
            model.load_state_dict(state)
            del state
        print(f"[{name}] scoring {len(s['file_list'])} files")
        ids, p, d = score_dataset(model, s["dataset"], device, args.batch_size, args.num_workers)
        write_scores(score_path, ids, p)
        write_scores(logit_path, ids, d)
        scores[name] = p

    report = {
        "model": model_desc, "checkpoints": paths, "ssl_name": args.ssl_name,
        "n_layers": args.n_layers, "config_hash": config_hash, "sets_file": args.sets,
        "full": evaluate_sets(sets, scores),
        "half_B": evaluate_sets(sets, scores, half="B"),
        "breakdowns": {},
    }
    for name, s in sets.items():
        keys = [k for k in BREAKDOWN_KEYS if any(r.get(k) for r in s["meta_rows"])]
        if keys:
            report["breakdowns"][name] = {k: metrics.breakdown(s["meta_rows"], s["y_spoof"],
                                                               scores[name], k) for k in keys}
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print_report(report)
    print(f"\nMetrics saved to: {out / 'metrics.json'}")
    if args.registry:
        append_registry(args.registry, report, args, model_desc, config_hash)


if __name__ == "__main__":
    main()
