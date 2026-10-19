"""
Figures of a TIES merge, the analogues of the analysis figures of arXiv 2306.01708, from the
aggregates written by ties_merging.stats (ties_stats.json / .npz) and, for the performance
plots, from a scripts/ties_eval.py sweep directory (summary.csv). Shared by the notebook
(notebooks/ties_merging_analysis.ipynb) and the CLI:

    python -m ties_merging.plots --stats <dir with ties_stats.*> [--sweep <ties_eval out dir>] --out <dir>

Every function takes the loaded data and returns a matplotlib Figure; `render_all` saves PNGs.
One hue per ingredient in fixed slot order (never cycled), TIES vs mean as slots 1 / 2,
one-hue sequential ramp for the cosine matrix, hairline grid, legends whenever >= 2 series.
"""

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# validated reference palette (dataviz skill): categorical slots, chrome, blue sequential ramp
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TIES_C, MEAN_C, OTHER_C = CAT[0], CAT[1], "#898781"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                     "#184f95", "#0d366b"])
EPOCH_RE = re.compile(r"epoch_(\d+)")


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 5,
        "font.family": "sans-serif", "figure.dpi": 110,
    })


def model_label(path):
    m = EPOCH_RE.search(str(path))
    return f"epoch {int(m[1])}" if m else Path(path).stem


def model_labels(js):
    labels = [model_label(m["path"]) for m in js["models"]]
    if len(set(labels)) < len(labels):          # cross-run lists: prefix with the run name
        labels = [f"{Path(m['path']).parents[1].name[:12]}/{l}" for m, l in zip(js["models"], labels)]
    return labels


def model_color(i):
    return CAT[i] if i < len(CAT) else OTHER_C


def _k_str(js):
    k = js["k"]
    return f"{int(k)}" if float(k).is_integer() else f"{k:g}"


def _group_order(js):
    return sorted(js["groups"], key=lambda g: (0 if g == "conv" else 2 if g == "ln_post" else 3 if g == "head"
                                               else 1, g))


# --------------------------------------------------------------------------- #
# task-vector magnitudes (paper Fig. 3 / 10 motivation)
# --------------------------------------------------------------------------- #
def fig_concentration(js, npz):
    """Share of the task vector's energy (sum tau^2) held by its top-q% entries; diagonal = uniform."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    q = npz["q_grid"]
    for i, (lab, row) in enumerate(zip(model_labels(js), npz["concentration"])):
        ax.plot(q, 100 * row, color=model_color(i), label=lab)
    ax.plot(q, q, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="uniform magnitudes")
    k = js["k"]
    ax.axvline(k, color=AXIS, linewidth=1)
    y_at_k = 100 * float(np.mean([row[int(round(k)) - 1] for row in npz["concentration"]]))
    ax.annotate(f"top {_k_str(js)}% holds {y_at_k:.0f}% of the energy", (k, y_at_k), xytext=(k + 4, y_at_k - 12),
                fontsize=9, color=INK2)
    ax.set_xlabel("top-q% of the entries by |τ|")
    ax.set_ylabel("share of Σ τ² (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_title("How concentrated are the task vectors?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_magnitude_hist(js, npz):
    """Distribution of log10 |tau| per ingredient, with the trim threshold at k."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    edges = npz["mag_bins"]
    centers = 0.5 * (edges[1:] + edges[:-1])
    for i, (lab, row, m) in enumerate(zip(model_labels(js), npz["mag_hist"], js["models"])):
        dens = row / max(row.sum(), 1) / (edges[1] - edges[0])
        ax.plot(centers, dens, color=model_color(i), label=lab, drawstyle="steps-mid")
        thr = m["thresholds"].get(str(float(js["k"])))
        if thr and np.isfinite(thr) and thr > 0:
            ax.axvline(np.log10(thr), color=model_color(i), linewidth=1, alpha=0.6)
    lo = max(edges[0], min(centers[(npz["mag_hist"].sum(0) > 0)]) - 0.5)
    ax.set_xlim(lo, edges[-1])
    ax.set_xlabel("log10 |τ|   (thin lines: trim threshold at k = %s%%)" % _k_str(js))
    ax.set_ylabel("density")
    ax.set_title("Magnitude of the fine-tuning updates")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_layers(js):
    """Per layer group: relative update norm, kept fraction at k per ingredient, and the merge's shared /
    conflict / merged fractions."""
    groups = _group_order(js)
    x = np.arange(len(groups))
    labels = model_labels(js)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    ax = axes[0]
    for i, m in enumerate(js["models"]):
        ax.plot(x, [100 * m["rel_norm_by_group"][g] for g in groups], color=model_color(i), marker="o",
                markersize=3.5, label=labels[i])
    ax.set_ylabel("‖τ‖ / ‖θ_init‖ (%)")
    ax.set_title("Where the fine-tuning moved the weights, and what TIES keeps")
    ax.legend(ncol=min(4, len(labels)), loc="upper left")
    ax = axes[1]
    for i, m in enumerate(js["models"]):
        ax.plot(x, [100 * m["kept_frac_by_group"][g] for g in groups], color=model_color(i), marker="o",
                markersize=3.5)
    ax.axhline(js["k"], color=AXIS, linewidth=1)
    ax.set_ylabel(f"kept at k = {_k_str(js)}% (%)")
    ax = axes[2]
    pg = js["per_group"]
    ax.plot(x, [100 * pg[g]["merged_frac"] for g in groups], color=TIES_C, marker="o", markersize=3.5,
            label="merged (kept by ≥ 1)")
    ax.plot(x, [100 * pg[g]["shared_frac"] for g in groups], color=CAT[2], marker="o", markersize=3.5,
            label="kept by ≥ 2")
    ax.plot(x, [100 * pg[g]["conflict_frac"] for g in groups], color=CAT[7], marker="o", markersize=3.5,
            label="sign conflict (of kept by ≥ 2)")
    ax.set_ylabel("positions (%)")
    ax.legend(ncol=3, loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("layer", "L") for g in groups], rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# sign conflicts (paper Fig. 4 / 14 / 15)
# --------------------------------------------------------------------------- #
def fig_conflicts(js, npz):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    kg = npz["k_grid"]
    ax.plot(kg, 100 * npz["conflict_vs_k"], color=CAT[7], marker="o", label="sign conflict (of positions kept by ≥ 2)")
    ax.plot(kg, 100 * npz["shared_vs_k"], color=CAT[2], marker="o", label="positions kept by ≥ 2 models")
    ax.axvline(js["k"], color=AXIS, linewidth=1)
    ax.set_xlabel("k (% of entries kept per task vector)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax.set_title("Sign conflicts vs trimming")
    ax.legend(loc="upper left")
    ax = axes[1]
    Ts = [r["T"] for r in js["conflict_vs_T"]]
    ax.plot(Ts, 100 * npz["conflict_vs_T"], color=CAT[7], marker="o", label=f"sign conflict at k = {_k_str(js)}%")
    ax.plot(Ts, [100 * r["shared_frac"] for r in js["conflict_vs_T"]], color=CAT[2], marker="o",
            label="positions kept by ≥ 2 models")
    ax.set_xlabel("number of checkpoints merged (list prefix)")
    ax.set_ylabel("%")
    ax.set_xticks(Ts)
    ax.set_title("Sign conflicts vs number of models")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# interference (paper Fig. 9a / 9b)
# --------------------------------------------------------------------------- #
def _grouped_bars(ax, labels, a, b, name_a, name_b, counts=None):
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, a, w - 0.04, color=TIES_C, label=name_a)
    ax.bar(x + w / 2, b, w - 0.04, color=MEAN_C, label=name_b)
    ticks = labels if counts is None else [f"{l}\n{_fmt_n(n)}" for l, n in zip(labels, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=9)
    ax.legend(loc="upper left")


def _fmt_n(n):
    n = int(n)
    return f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.0f}k" if n >= 1e3 else str(n))


def fig_interference(js, npz):
    """Mean |theta_merged - theta_init| per position group, TIES vs plain mean: by how many models kept the
    position (redundant-value interference) and by sign agreement among them (sign interference)."""
    T = js["T"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    kc = npz["kept_count_mean_abs"]
    labels = [("kept by\nnobody" if c == 0 else f"kept by\n{c}") for c in range(T + 1)]
    _grouped_bars(ax, labels, kc[0], kc[1], f"TIES (k = {_k_str(js)}%, λ = {js['lambda']:g})", "plain mean",
                  npz["kept_count_hist"])
    ax.set_ylabel("mean |θ_merged − θ_init|")
    ax.set_title("Redundant-value interference")
    ax = axes[1]
    ag = npz["agreement_mean_abs"]
    edges = npz["agreement_edges"]
    labels = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" if i < len(edges) - 1 else "1.0" for i in range(len(edges))]
    _grouped_bars(ax, labels, ag[0], ag[1], "TIES", "plain mean", npz["agreement_hist"])
    ax.set_xlabel("sign agreement among the models keeping the position")
    ax.set_title("Sign interference (positions kept by ≥ 2)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# election and geometry
# --------------------------------------------------------------------------- #
def fig_election(js):
    labels = model_labels(js)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.bar(x, [100 * m["elect_agree_frac"] for m in js["models"]], 0.6, color=[model_color(i) for i in range(len(labels))])
    ax.set_ylabel("kept entries matching the elected sign (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Who agrees with the election")
    ax = axes[1]
    ax.bar(x, [100 * m["contribution_frac"] for m in js["models"]], 0.6, color=[model_color(i) for i in range(len(labels))])
    ax.axhline(100 / len(labels), color=AXIS, linewidth=1)
    ax.set_ylabel("share of the merged entries (%)")
    ax.set_title("Who contributes to τ_m (line: equal share)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    return fig


def fig_cosine(js, npz):
    labels = model_labels(js)
    c = npz["cosine"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(labels), 0.8 + 0.8 * len(labels)))
    im = ax.imshow(c, cmap=SEQ, vmin=min(0.0, float(c.min())), vmax=1.0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color=INK if c[i, j] < 0.75 else SURFACE)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(False)
    ax.set_title("cosine(τ_i, τ_j)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# performance from a ties_eval sweep (paper Fig. 12 / 13)
# --------------------------------------------------------------------------- #
def load_sweep(sweep_dir):
    import pandas as pd
    df = pd.read_csv(Path(sweep_dir) / "summary.csv")
    for c in ("k", "lam", "T"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("trim", "elect"):
        if c in df:
            df[c] = df[c].astype(str).str.lower().isin(("true", "1"))
    return df


def _ties_cells(df):
    return df[(df["kind"] == "ties") & df["trim"] & df["elect"] & (df["merge"] == "dis-mean")]


def fig_perf_vs_lambda(df, metric="half_B.wf1", ref_kind="mean"):
    """metric vs lambda, one line per k; dashed line = plain mean of the same checkpoints."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cells = _ties_cells(df)
    ks = sorted(cells["k"].dropna().unique())
    for i, k in enumerate(ks):
        sub = cells[cells["k"] == k].sort_values("lam")
        if len(sub) == 0:
            continue
        ax.plot(sub["lam"], sub[metric], color=model_color(i), marker="o", label=f"k = {k:g}%")
    ref = df[df["kind"] == ref_kind]
    if len(ref):
        ax.axhline(float(ref[metric].iloc[0]), color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="plain mean")
    ax.set_xlabel("λ")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs λ")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_perf_vs_k(df, metrics=("half_A.wf1", "half_B.wf1"), lam=1.0, ref_kind="mean"):
    """metric vs k at one lambda, one line per split; dashed = plain mean (half_B)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cells = _ties_cells(df)
    cells = cells[np.isclose(cells["lam"], lam)].sort_values("k")
    for i, m in enumerate(metrics):
        if m in cells:
            ax.plot(cells["k"], cells[m], color=model_color(i), marker="o", label=m)
    ref = df[df["kind"] == ref_kind]
    if len(ref) and metrics[-1] in ref:
        ax.axhline(float(ref[metrics[-1]].iloc[0]), color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
                   label=f"plain mean ({metrics[-1]})")
    ax.set_xlabel("k (% kept)")
    ax.set_ylabel("WF1 (%)")
    ax.set_title(f"WF1 vs k at λ = {lam:g}")
    ax.legend()
    fig.tight_layout()
    return fig


def ablation_table(df, metrics=("half_A.wf1", "half_B.wf1", "half_B.f1_clean", "half_B.f1_noisy")):
    """One row per cell (kind / k / lambda / variant) with the chosen metrics, for the notebook."""
    cols = [c for c in ("ckpt", "kind", "k", "lam", "merge", "trim", "elect", "T") if c in df] + \
           [m for m in metrics if m in df]
    out = df[cols].copy()
    if "kind" in out:
        out = out.sort_values(["kind", "k", "lam"], na_position="first")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
FIGS = {"concentration": fig_concentration, "magnitude_hist": fig_magnitude_hist, "layers": lambda js, npz: fig_layers(js),
        "conflicts": fig_conflicts, "interference": fig_interference, "election": lambda js, npz: fig_election(js),
        "cosine": fig_cosine}


def render_all(stats_dir, out_dir, sweep_dir=None, dpi=150):
    from .stats import load
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, npz = load(stats_dir)
    written = []
    for name, fn in FIGS.items():
        fig = fn(js, npz)
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    if sweep_dir is not None and (Path(sweep_dir) / "summary.csv").is_file():
        df = load_sweep(sweep_dir)
        if len(_ties_cells(df)):
            for name, fig in (("perf_vs_lambda", fig_perf_vs_lambda(df)), ("perf_vs_k", fig_perf_vs_k(df))):
                p = out_dir / f"{name}.png"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                written.append(p)
        ablation_table(df).to_csv(out_dir / "cells.csv", index=False)
        written.append(out_dir / "cells.csv")
    return written


def main():
    ap = argparse.ArgumentParser(description="Render the TIES analysis figures to PNG")
    ap.add_argument("--stats", type=str, required=True, help="directory with ties_stats.json / .npz")
    ap.add_argument("--sweep", type=str, default=None, help="scripts/ties_eval.py output dir (summary.csv)")
    ap.add_argument("--out", type=str, default=None, help="default: <stats>/figures")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    matplotlib.use("Agg")
    out = Path(args.out) if args.out else Path(args.stats) / "figures"
    for p in render_all(args.stats, out, args.sweep, args.dpi):
        print(p)


if __name__ == "__main__":
    main()
