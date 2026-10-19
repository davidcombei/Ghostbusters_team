"""
TIES-merge the checkpoints of a merge list (scripts/select_topk.py output, or any list of plain
ModelSLS state dicts) and save the result as one checkpoint, usable anywhere a single checkpoint
goes (local_eval.py --model_path, make_submission.py, /post-train-eval).

    python scripts/ties_merge.py --merge_list SLS_setup/exp/<run>/top5_by_wf1/merge_list.txt
    python scripts/ties_merge.py --merge_list <list> --k 20 --lam 1.0 --stats --out <dir> [--force]
    python scripts/ties_merge.py --merge_list <list> --base <ckpt.pth>        full-model TIES vs that checkpoint

Algorithm and defaults (Yadav et al. 2023, arXiv 2306.01708): task vectors against the pretrained
backbone (theta_init = hub Qwen3-ASR weights, read from the cached safetensors), trim to the top
k = 20% magnitudes, elect the sign of the sum, disjoint mean of the agreeing entries,
theta = theta_init + lambda * tau_m with lambda = 1. The head and the BatchNorm statistics are
plain-averaged exactly as main_eval.merge_checkpoints does (the head has no pretrained init).

Writes to --out (default <merge_list dir>/ties_<tag>/, tag = k20_lam1[...]):
    merged.pth            plain state dict (tmp + rename)
    merged.sources.txt    resolved source paths (a merge list itself)
    ties_config.json      every setting, the base, the sources, the package sha and the merge summary
    ties_stats.json/.npz  with --stats: the analysis behind the paper figures (ties_merging.stats),
                          rendered by `python -m ties_merging.plots` and the notebook
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLS_DIR = REPO_ROOT / "SLS_setup"
sys.path.insert(0, str(SLS_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from local_eval import read_merge_list  # noqa: E402
from ties_merging import __version__, module_sha256  # noqa: E402
from ties_merging import merge as M  # noqa: E402
from ties_merging.base import load_base, model_args_from_runs, runs_of  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="TIES-merge the checkpoints of a merge list into one .pth",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--merge_list", type=str, required=True, help="text file, one checkpoint path per line")
    p.add_argument("--out", type=str, default=None, help="output dir (default: <merge_list dir>/ties_<tag>)")
    p.add_argument("--force", action="store_true", help="overwrite an existing merged.pth")
    p.add_argument("--base", type=str, default="pretrained",
                   help="'pretrained' (hub backbone; head plain-averaged) or a checkpoint path (full-model TIES)")
    p.add_argument("--ssl_name", type=str, default=None, help="default: the run's config.yaml")
    p.add_argument("--n_layers", type=int, default=None, help="default: the run's config.yaml")
    p.add_argument("--base_via_model", action="store_true",
                   help="build ModelSLS for the pretrained base instead of reading the safetensors")
    p.add_argument("--k", type=float, default=20.0, help="top-k%% of |tau| kept per task vector (100 = no trim)")
    p.add_argument("--lam", type=float, default=1.0, help="lambda: theta = theta_init + lambda * tau_m")
    p.add_argument("--trim_scope", type=str, default="global", choices=M.TRIM_SCOPES,
                   help="global (paper) or per tensor")
    p.add_argument("--merge", type=str, default="dis-mean", choices=M.MERGE_MODES,
                   help="dis-mean (paper: / |A^p|) or mean_over_T (/ T, ablation)")
    p.add_argument("--no_trim", action="store_true", help="ablation: keep every entry (same as --k 100)")
    p.add_argument("--no_elect", action="store_true", help="ablation: no sign election, average every kept entry")
    p.add_argument("--stats", action="store_true", help="also write ties_stats.json/.npz (minutes, ~20 GiB RAM)")
    p.add_argument("--k_grid", type=float, nargs="+", default=None,
                   help="k values of the sign-conflict-vs-k statistic (default 5 10 20 30 50 70 100)")
    return p


def main():
    args = build_parser().parse_args()
    t0 = time.time()
    try:
        paths = read_merge_list(args.merge_list)
    except FileNotFoundError as e:
        sys.exit(f"checkpoint in {args.merge_list} not found: {e}")
    if not paths:
        sys.exit(f"no checkpoints in {args.merge_list}")
    trim, elect = not args.no_trim, not args.no_elect
    tag = M.cell_tag(args.k, args.lam, args.merge, trim, elect, args.trim_scope)
    out = Path(args.out) if args.out else Path(args.merge_list).parent / f"ties_{tag}"
    merged_path = out / "merged.pth"
    if merged_path.exists() and not args.force:
        sys.exit(f"{merged_path} exists (use --force to overwrite)")

    runs = runs_of(paths)
    ssl_name, n_layers, _, _, warnings = model_args_from_runs(runs, args.ssl_name, args.n_layers)
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"{len(paths)} checkpoints from {len(runs)} run(s); base={args.base} "
          f"(ssl_name={ssl_name}, n_layers={n_layers}); k={args.k:g} lam={args.lam:g} "
          f"trim_scope={args.trim_scope} merge={args.merge} trim={trim} elect={elect}", flush=True)
    base = load_base(args.base, ssl_name, n_layers, args.base_via_model)
    print(f"base: {base.label} ({len(base.state)} tensors) {base.meta}", flush=True)

    merger = M.Merger(paths, base)
    ing = merger.ing
    print(f"TIES keys: {len(ing.ties_keys)} ({ing.n / 1e6:.1f}M params), mean keys: {len(ing.mean_keys)}, "
          f"copied: {len(ing.copy_keys)}", flush=True)
    tau_m, summary = merger.tau(args.k, args.trim_scope, args.merge, trim, elect)
    state = merger.state(tau_m, args.lam)
    summary["lambda"] = args.lam

    out.mkdir(parents=True, exist_ok=True)
    tmp = merged_path.with_name(merged_path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, merged_path)
    (out / "merged.sources.txt").write_text("".join(os.path.abspath(p) + "\n" for p in paths), encoding="utf-8")
    config = {
        "tag": tag, "merge_list": os.path.abspath(args.merge_list), "sources": [os.path.abspath(p) for p in paths],
        "runs": [str(r) for r in runs], "base": {"spec": args.base, "kind": base.kind, "label": base.label,
                                                  "meta": base.meta},
        "ssl_name": ssl_name, "n_layers": n_layers, "k": args.k, "lambda": args.lam,
        "trim_scope": args.trim_scope, "merge": args.merge, "trim": trim, "elect": elect,
        "ties_keys": len(ing.ties_keys), "mean_keys": ing.mean_keys, "copy_keys": ing.copy_keys,
        "summary": summary, "warnings": warnings,
        "ties_merging_version": __version__, "ties_merging_sha256": module_sha256(),
    }
    with open(out / "ties_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    n_params = sum(v.numel() for v in state.values() if v.is_floating_point())
    print(f"\nkept fraction per model: {[round(x, 4) for x in summary['kept_frac']]}")
    print(f"positions kept by >= 2 models: {100 * summary['shared_frac']:.2f}%, sign conflicts among them: "
          f"{100 * summary['conflict_frac']:.3f}%, merged positions: {100 * summary['merged_frac']:.2f}%, "
          f"||tau_m|| = {summary['tau_m_norm']:.4f}")
    print(f"Merged {len(paths)} checkpoints: {len(state)} tensors, {n_params / 1e6:.1f}M float params, "
          f"{merged_path.stat().st_size / 2**30:.2f} GiB in {time.time() - t0:.0f}s")
    print(f"Merged checkpoint: {merged_path}\nConfig: {out / 'ties_config.json'}")
    del state

    if args.stats:
        from ties_merging import stats as S
        k_grid = args.k_grid or S.K_GRID
        S.analyze(merger, args.k, k_grid, out, args.lam, args.trim_scope, args.merge)
        print(f"Stats: {out / 'ties_stats.json'}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
