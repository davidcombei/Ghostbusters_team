"""
Score TIES merges of one merge list on the local proxy sets over a (k, lambda) grid, next to the
plain mean of the same checkpoints, with the model built once and only the weights swapped.

    cd SLS_setup
    python -u ../scripts/ties_eval.py --merge_list exp/<run>/top5_by_wf1/merge_list.txt \\
        --k 20 --lam 1.0 --out exp/<run>/local_eval/ties_sweep                      # default cell + mean
    python -u ../scripts/ties_eval.py --merge_list <list> --k 10 20 30 50 70 100 --lam 1.0 \\
        --names dev_online_clean sim_matched_mini sim_heldout_mini --out <dir>       # stage 1: k grid, half A
    python -u ../scripts/ties_eval.py --merge_list <list> --k 20 --lam 0.6 0.8 1.0 1.2 1.4 ...  # lambda grid
    python -u ../scripts/ties_eval.py --merge_list <list> --k 20 --lam 1.0 --ablation --single 0 ...

Cells: `mean` (plain merge_checkpoints, always unless --no_mean); one TIES cell per (k, lambda,
--merge mode); --ablation adds no-trim, no-elect and mean-over-T variants at the first k / lambda
(paper Table "ablation"); --single i adds the T = 1 trim curve of ingredient i (its own task vector
trimmed at each k, lambda = first). Writes under --out (like scripts/eval_checkpoints.py):

    <tag>/scores/<set>.txt|.logits   per-cell score cache, same format as scripts/local_eval.py
    <tag>/metrics.json               local_eval.py's report + half_A block + a `ties` block
    summary.csv                      one row per cell: kind, k, lam, merge, trim, elect, T and the
                                     {split}.{metric} columns of eval_checkpoints.py (percent)
    best.json                        top cells by --select_on (default half_A.wf1), with half_B.wf1

Select on half A, report half B: scripts/compare_evals.py and scripts/b3_compare_evals.py --ci
work on any <tag> directory. Cached cells are reused (resume after an interruption); the model
arguments (ssl_name, n_layers, loudness) come from the run's config.yaml unless given.
One 17-set pass is ~0.5 GPU-h on a 4090; the three half-A selection sets take ~5 min per cell.
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

from eval_checkpoints import (SPLITS, build_report, cached_scores, print_best, summary_row,  # noqa: E402
                              write_best, write_summary)
from local_eval import read_merge_list, write_scores  # noqa: E402
from model.sls_model import ModelSLS, ssl_path  # noqa: E402
from ties_merging import __version__, module_sha256  # noqa: E402
from ties_merging import merge as M  # noqa: E402
from ties_merging.base import load_base, model_args_from_runs, runs_of  # noqa: E402
from utils.data_utils import build_loudness, set_random_seed  # noqa: E402
from utils.local_eval import load_sets, score_dataset  # noqa: E402

DEFAULT_NAMES = ["dev_online_clean", "sim_matched_v1", "sim_heldout_v1", "sim_echo_v1"]
CELL_FIELDS = ("kind", "k", "lam", "merge", "trim", "elect", "T", "ingredient")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merge_list", type=str, required=True, help="text file, one checkpoint path per line")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--k", type=float, nargs="+", default=[20.0], help="k values (%% kept); 100 = no trim")
    p.add_argument("--lam", type=float, nargs="+", default=[1.0], help="lambda values")
    p.add_argument("--merge", type=str, nargs="+", default=["dis-mean"], choices=M.MERGE_MODES)
    p.add_argument("--trim_scope", type=str, default="global", choices=M.TRIM_SCOPES)
    p.add_argument("--ablation", action="store_true", help="add no-trim / no-elect / mean-over-T cells at k[0], lam[0]")
    p.add_argument("--single", type=int, nargs="*", default=None,
                   help="ingredient indices (0-based in the list) whose T = 1 trim curve is added at every k, lam[0]")
    p.add_argument("--no_mean", action="store_true", help="skip the plain-mean baseline cell")
    p.add_argument("--base", type=str, default="pretrained", help="'pretrained' or a checkpoint path")
    p.add_argument("--base_via_model", action="store_true")
    p.add_argument("--save", type=str, default="none", choices=["none", "best", "all"],
                   help="write <tag>/merged.pth for the best cell (by --select_on) or every cell")
    p.add_argument("--sets", type=str, default=str(REPO_ROOT / "configs/local_eval_sets.yaml"))
    p.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    p.add_argument("--select_on", type=str, default="half_A.wf1")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--ssl_name", type=str, default=None, help="default: from the run's config.yaml")
    p.add_argument("--n_layers", type=int, default=None, help="default: from the run's config.yaml")
    p.add_argument("--devices", type=int, nargs="+", default=None)
    p.add_argument("--layer_split", type=int, nargs="+", default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--force", action="store_true", help="rescore cells that are already cached")
    p.add_argument("--loudness_norm", dest="loudness_norm", action="store_true", default=None)
    p.add_argument("--no_loudness_norm", dest="loudness_norm", action="store_false", default=None)
    p.add_argument("--loudness_norm_lufs", type=float, default=None)
    return p


def plan_cells(args, T):
    """[{tag, kind, k, lam, merge, trim, elect, T, ingredient}] in scoring order (grouped by k)."""
    cells = []
    if not args.no_mean:
        cells.append({"tag": "mean", "kind": "mean", "k": "", "lam": "", "merge": "", "trim": "", "elect": "",
                      "T": T, "ingredient": ""})
    ks, lams = list(dict.fromkeys(args.k)), list(dict.fromkeys(args.lam))
    for k in ks:
        for merge in args.merge:
            for lam in lams:
                cells.append({"tag": M.cell_tag(k, lam, merge, True, True, args.trim_scope), "kind": "ties",
                              "k": k, "lam": lam, "merge": merge, "trim": True, "elect": True, "T": T,
                              "ingredient": ""})
        if args.single:
            for i in args.single:
                if not 0 <= i < T:
                    raise SystemExit(f"--single {i}: the list has {T} checkpoints")
                cells.append({"tag": f"single{i}_" + M.cell_tag(k, lams[0], "dis-mean", True, True, args.trim_scope),
                              "kind": "single", "k": k, "lam": lams[0], "merge": "dis-mean", "trim": True,
                              "elect": True, "T": 1, "ingredient": i})
    if args.ablation:
        k, lam = ks[0], lams[0]
        for merge, trim, elect in (("dis-mean", False, True), ("dis-mean", True, False), ("mean_over_T", True, True)):
            cells.append({"tag": M.cell_tag(k, lam, merge, trim, elect, args.trim_scope), "kind": "ties",
                          "k": k, "lam": lam, "merge": merge, "trim": trim, "elect": elect, "T": T, "ingredient": ""})
    seen, uniq = set(), []
    for c in cells:
        if c["tag"] not in seen:
            seen.add(c["tag"])
            uniq.append(c)
    return uniq


def cell_state(cell, merger, singles, base, paths):
    """(state_dict, merge summary or None) of one cell."""
    if cell["kind"] == "mean":
        return merger.plain_mean(), None
    if cell["kind"] == "single":
        i = cell["ingredient"]
        if i not in singles:
            singles[i] = M.Merger([paths[i]], base)
        m = singles[i]
    else:
        m = merger
    tau_m, summary = m.tau(cell["k"], cell["trim_scope"], cell["merge"], cell["trim"], cell["elect"])
    return m.state(tau_m, cell["lam"]), summary


def main():
    args = build_parser().parse_args()
    t0 = time.time()
    paths = read_merge_list(args.merge_list)
    if not paths:
        raise SystemExit(f"no checkpoints in {args.merge_list}")
    minis = {n[:-5] for n in args.names if n.endswith("_mini")}
    clash = [n for n in args.names if n.endswith("_v1") and n[:-3] in minis]
    if clash:
        raise SystemExit(f"{clash} and their _mini subsets would double-count in the noisy WF1 -- drop one side")

    runs = runs_of(paths)
    ssl_name, n_layers, loud, lufs, warnings = model_args_from_runs(runs, args.ssl_name, args.n_layers)
    for w in warnings:
        print(f"WARNING: {w}")
    if ssl_name is None or n_layers is None:
        raise SystemExit("no config.yaml behind the checkpoints: pass --ssl_name and --n_layers")
    args.ssl_name, args.n_layers = ssl_path(ssl_name), n_layers
    if args.loudness_norm is None:
        args.loudness_norm = loud
    if args.loudness_norm_lufs is None:
        args.loudness_norm_lufs = lufs
    args.cudnn_deterministic_toggle, args.cudnn_benchmark_toggle = True, False
    set_random_seed(args.seed, args)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "merge_list.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    cells = plan_cells(args, len(paths))
    for c in cells:
        c["trim_scope"] = args.trim_scope
    print(f"{len(paths)} checkpoints from {len(runs)} run(s) | {args.ssl_name}, {args.n_layers} layers, "
          f"loudness_norm={args.loudness_norm} | base={args.base} | {len(cells)} cells: "
          f"{[c['tag'] for c in cells]} | sets {args.names} -> {out_root}", flush=True)
    sets = load_sets(args.sets, args.names, ssl_name=args.ssl_name, loudness=build_loudness(args))
    if args.select_on != "dev_loss" and args.select_on.split(".", 1)[0] not in SPLITS:
        raise SystemExit(f"--select_on {args.select_on}: expected <split>.<metric>, split in {list(SPLITS)}")

    model, device, merger, base, singles, rows, states_kept = None, None, None, None, {}, [], {}
    current_k = None
    try:
        for i, cell in enumerate(cells, 1):
            out = out_root / cell["tag"]
            scores = {} if args.force else cached_scores(out, sets)
            missing = [n for n in sets if n not in scores]
            summary = None
            if missing or args.save == "all":
                if base is None:
                    base = load_base(args.base, args.ssl_name, args.n_layers, args.base_via_model)
                    merger = M.Merger(paths, base)
                    print(f"base: {base.label}; TIES keys {len(merger.ing.ties_keys)} "
                          f"({merger.ing.n / 1e6:.1f}M params), mean keys {len(merger.ing.mean_keys)}", flush=True)
                if cell["kind"] == "ties" and cell["k"] != current_k:
                    merger.clear_cache()
                    current_k = cell["k"]
                state, summary = cell_state(cell, merger, singles, base, paths)
                if args.save == "all":
                    out.mkdir(parents=True, exist_ok=True)
                    torch.save(state, out / "merged.pth")
            if missing:
                if model is None:
                    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
                    model = ModelSLS(args, device)
                    device = model.input_device
                model.load_state_dict(state)
                del state
                (out / "scores").mkdir(parents=True, exist_ok=True)
                for name in missing:
                    ids, p, d = score_dataset(model, sets[name]["dataset"], device, args.batch_size, args.num_workers)
                    write_scores(out / "scores" / f"{name}.txt", ids, p)
                    write_scores(out / "scores" / f"{name}.logits", ids, d)
                    scores[name] = p
            ckpt = {"path": Path(cell["tag"]), "tag": cell["tag"], "epoch": None, "dev_loss": None, "aliases": []}
            report = build_report(ckpt, sets, scores, args)
            report["model"] = f"ties:{args.merge_list}:{cell['tag']}"
            report["checkpoints"] = paths if cell["kind"] != "single" else [paths[cell["ingredient"]]]
            report["ties"] = {k: cell[k] for k in CELL_FIELDS}
            report["ties"].update({"tag": cell["tag"], "base": args.base, "trim_scope": args.trim_scope,
                                   "summary": summary, "ties_merging_version": __version__,
                                   "ties_merging_sha256": module_sha256()})
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            row = summary_row(ckpt, report, sets)
            row.update({k: cell[k] for k in CELL_FIELDS})
            rows.append(row)
            write_summary(out_root / "summary.csv", rows)
            print(f"[{i}/{len(cells)}] {cell['tag']:<22} WF1 full {row['full.wf1']} | A {row['half_A.wf1']} "
                  f"| B {row['half_B.wf1']}   (B: clean {row['half_B.f1_clean']}, noisy {row['half_B.f1_noisy']})"
                  f"{'  [scored ' + ','.join(missing) + ']' if missing else '  [cached]'}"
                  f"{'' if summary is None else f'  kept {summary['kept_frac'][0]:.3f} conflict {summary['conflict_frac']:.4f}'}",
                  flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; summarising the cells done so far")

    if not rows:
        raise SystemExit("no cell scored")
    if args.select_on not in rows[0]:
        raise SystemExit(f"--select_on {args.select_on} is not a summary.csv column")
    best = write_best(out_root / "best.json", rows, args.select_on, args.top_k)
    print_best(best)
    if args.save == "best" and best["top"]:
        tag = best["top"][0]["ckpt"]
        cell = next(c for c in cells if c["tag"] == tag)
        if base is None:
            base = load_base(args.base, args.ssl_name, args.n_layers, args.base_via_model)
            merger = M.Merger(paths, base)
        state, _ = cell_state(cell, merger, singles, base, paths)
        torch.save(state, out_root / tag / "merged.pth")
        print(f"best cell saved: {out_root / tag / 'merged.pth'}")
    print(f"\nSummary: {out_root / 'summary.csv'}\nBest:    {out_root / 'best.json'}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
