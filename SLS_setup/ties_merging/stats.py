"""
Analysis statistics of a TIES merge, the quantities behind the paper's figures (Sec. 7 of
arXiv 2306.01708), gathered once per merge list and saved as small aggregates:

    <out>/ties_stats.json   scalars and tables (per model, per layer group, per k, per prefix T)
    <out>/ties_stats.npz    binned arrays (magnitude histograms, concentration curves, kept-count
                            and agreement histograms with merged magnitudes, cosine matrix)

Nothing of the size of a task vector is written, so ties_merging.plots and the notebook run in
seconds. Norms and dot products are accumulated in float64 (a float32 dot over 3e8 entries is
wrong by percent). Per model, one sort of |tau| (numpy, in place, no index) gives every k
threshold, the quantiles and the energy-concentration curve.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch

from .merge import Merger, trim_mask

K_GRID = (5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0)
Q_GRID = np.arange(1, 101)                    # top-q% of the entries, q = 1..100
LOG_LO, LOG_HI, LOG_BINS = -9.0, 0.0, 90      # log10 |tau| histogram range
AGREE_EDGES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0 + 1e-9])   # last bin = full agreement
QUANTILES = (0.5, 0.8, 0.9, 0.99, 0.999)
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def key_group(key):
    m = LAYER_RE.search(key)
    if m:
        return f"layer{int(m[1]):02d}"
    if not key.startswith("ssl_model."):
        return "head"
    if ".conv" in key:
        return "conv"
    if "ln_post" in key:
        return "ln_post"
    return "other"


def _groups(layout):
    g = {}
    for k in layout.keys:
        g.setdefault(key_group(k), []).append(k)
    return g


def _sorted_abs(flat):
    """Ascending |tau| as a float32 numpy array (own copy, sorted in place, no index tensor)."""
    a = flat.abs().numpy().copy()
    a.sort()
    return a


def _concentration(sorted_abs):
    """Share of sum(tau^2) held by the top-q% entries, q = 1..100 (float64)."""
    n = sorted_abs.size
    sq = np.square(sorted_abs[::-1].astype(np.float64))
    cs = np.cumsum(sq)
    del sq
    total = cs[-1] if cs[-1] > 0 else 1.0
    idx = np.clip(np.round(n * Q_GRID / 100.0).astype(np.int64) - 1, 0, n - 1)
    return cs[idx] / total


def _threshold_from_sorted(sorted_abs, k_pct):
    n = sorted_abs.size
    k_keep = min(n, max(0, int(round(n * k_pct / 100.0))))
    if k_keep >= n:
        return 0.0
    if k_keep <= 0:
        return float("inf")
    return float(sorted_abs[n - k_keep])          # (n - k_keep + 1)-th smallest


def _group_mean_abs(values, group_idx, n_groups):
    """mean |values| per integer group id (float64), n per group."""
    av = values.abs().to(torch.float64) if values.is_floating_point() else values.to(torch.float64)
    sums = torch.zeros(n_groups, dtype=torch.float64).index_add_(0, group_idx, av)
    cnts = torch.zeros(n_groups, dtype=torch.float64).index_add_(0, group_idx, torch.ones_like(av))
    return (sums / cnts.clamp(min=1)).numpy(), cnts.numpy().astype(np.int64)


def analyze(merger, k_pct=20.0, k_grid=K_GRID, out_dir=None, lam=1.0, trim_scope="global",
            merge="dis-mean", log=print):
    """
    Gather the statistics for `merger` (ties_merging.merge.Merger) at the merge setting (k, lam)
    plus the k grid; write ties_stats.json / .npz to `out_dir` (if given) and return the dict.
    """
    ing = merger.ing
    T, n = ing.T, ing.n
    if T > 127:
        raise ValueError("int8 counters: at most 127 ingredients")
    k_grid = sorted(set(float(k) for k in k_grid) | {float(k_pct)})
    groups = _groups(ing.layout)
    group_names = sorted(groups)
    gid = {g: i for i, g in enumerate(group_names)}
    group_idx = torch.empty(n, dtype=torch.int64)
    for k, o, m in ing.layout.slices():
        group_idx[o:o + m] = gid[key_group(k)]

    # ---- pass 1: per model, flat |tau|: histogram, quantiles, thresholds, masks per k ----
    pos = {k: torch.zeros(n, dtype=torch.int8) for k in k_grid}
    neg = {k: torch.zeros(n, dtype=torch.int8) for k in k_grid}
    s_full = torch.zeros(n, dtype=torch.float32)          # sum of untrimmed task vectors
    s_trim = torch.zeros(n, dtype=torch.float32)          # sum of trimmed task vectors at k_pct
    signed = []                                           # int8 signed mask at k_pct per model
    mag_hist = np.zeros((T, LOG_BINS), dtype=np.int64)
    concentration = np.zeros((T, Q_GRID.size), dtype=np.float64)
    models = []
    conflict_vs_T = []
    for t in range(T):
        log(f"[stats] model {t + 1}/{T}: {ing.paths[t]}")
        flat = ing.flat_tau(t)
        nz = flat != 0
        n_zero = n - int(nz.sum())
        lm = torch.log10(flat[nz].abs()).clamp_(LOG_LO, LOG_HI)
        mag_hist[t] = torch.histc(lm, bins=LOG_BINS, min=LOG_LO, max=LOG_HI).to(torch.int64).numpy()
        del lm
        sa = _sorted_abs(flat)
        concentration[t] = _concentration(sa)
        thresholds = {k: _threshold_from_sorted(sa, k) for k in k_grid}
        quantiles = {str(q): float(sa[min(sa.size - 1, int(q * sa.size))]) for q in QUANTILES}
        del sa
        s_full += flat
        kept_frac, kept_group = {}, {}
        for k in k_grid:
            thr = thresholds[k]
            mask = (flat.abs() >= thr) & nz
            sm = flat.sign().to(torch.int8)
            sm.mul_(mask.to(torch.int8))
            pos[k] += (sm > 0).to(torch.int8)
            neg[k] += (sm < 0).to(torch.int8)
            kept_frac[k] = int(mask.sum()) / n
            if k == float(k_pct):
                signed.append(sm)
                s_trim += torch.where(mask, flat, torch.zeros((), dtype=flat.dtype))
                cnt_g = torch.zeros(len(group_names), dtype=torch.float64).index_add_(
                    0, group_idx, mask.to(torch.float64))
                size_g = torch.zeros(len(group_names), dtype=torch.float64).index_add_(
                    0, group_idx, torch.ones(n, dtype=torch.float64))
                kept_group = {g: float(cnt_g[i] / size_g[i]) for g, i in gid.items()}
                c = pos[k] + neg[k]
                shared = c >= 2
                conf = shared & (pos[k] > 0) & (neg[k] > 0)
                n_shared = int(shared.sum())
                conflict_vs_T.append({"T": t + 1, "shared_frac": n_shared / n,
                                      "conflict_frac": (int(conf.sum()) / n_shared) if n_shared else 0.0})
            del mask, sm
        del flat, nz
        models.append({"index": t, "path": ing.paths[t], "n_exact_zero": n_zero,
                       "thresholds": {str(k): thresholds[k] for k in k_grid},
                       "kept_frac": {str(k): kept_frac[k] for k in k_grid},
                       "kept_frac_by_group": kept_group, "quantiles_abs": quantiles})

    # ---- election at k_pct, conflict vs k ----
    gamma = s_trim.sign().to(torch.int8)
    del s_trim
    conflict_vs_k = {}
    for k in k_grid:
        c = pos[k] + neg[k]
        shared = c >= 2
        conf = shared & (pos[k] > 0) & (neg[k] > 0)
        n_shared = int(shared.sum())
        conflict_vs_k[str(k)] = {"kept_any_frac": int((c > 0).sum()) / n, "shared_frac": n_shared / n,
                                 "conflict_frac": (int(conf.sum()) / n_shared) if n_shared else 0.0,
                                 "mean_kept_frac": float(np.mean([m["kept_frac"][str(k)] for m in models]))}
    kp, kn = pos[float(k_pct)], neg[float(k_pct)]
    kept_count = (kp + kn).to(torch.int64)
    agree_level = torch.maximum(kp, kn).to(torch.float64) / kept_count.clamp(min=1).to(torch.float64)
    for k in list(pos):
        if k != float(k_pct):
            del pos[k], neg[k]

    # per-model election statistics
    total_agree = 0
    for t in range(T):
        sm = signed[t]
        kept = int((sm != 0).sum())
        agree = int(((sm != 0) & (sm == gamma)).sum())
        models[t]["elect_agree_frac"] = (agree / kept) if kept else 0.0
        models[t]["n_agree"] = agree
        total_agree += agree
    for t in range(T):
        models[t]["contribution_frac"] = (models[t]["n_agree"] / total_agree) if total_agree else 0.0
    del signed

    # ---- merged vectors: TIES tau_m at (k, lam) and the plain mean task vector ----
    tau_m, summary = merger.tau(k_pct, trim_scope, merge)
    ties_abs = (lam * tau_m).abs()
    mean_abs = (s_full / T).abs().to(torch.float64)
    del s_full
    kc_hist = np.bincount(kept_count.numpy(), minlength=T + 1)[:T + 1]
    kc_ties, _ = _group_mean_abs(ties_abs, kept_count, T + 1)
    kc_mean, _ = _group_mean_abs(mean_abs, kept_count, T + 1)
    shared = kept_count >= 2
    abin = torch.from_numpy(np.digitize(agree_level[shared].numpy(), AGREE_EDGES[1:-1]))  # 0..5
    agreement_hist = np.bincount(abin.numpy(), minlength=6)[:6]
    ag_ties, _ = _group_mean_abs(ties_abs[shared], abin, 6)
    ag_mean, _ = _group_mean_abs(mean_abs[shared], abin, 6)
    del abin, shared, agree_level

    # per-group conflict / merged fractions and norms of the merged vectors
    n_g = len(group_names)
    ones = torch.ones(n, dtype=torch.float64)
    size_g = torch.zeros(n_g, dtype=torch.float64).index_add_(0, group_idx, ones)
    conf = ((kp > 0) & (kn > 0)).to(torch.float64)
    shared_g = torch.zeros(n_g, dtype=torch.float64).index_add_(0, group_idx, (kept_count >= 2).to(torch.float64))
    conf_g = torch.zeros(n_g, dtype=torch.float64).index_add_(0, group_idx, conf)
    merged_g = torch.zeros(n_g, dtype=torch.float64).index_add_(0, group_idx, (kept_count > 0).to(torch.float64))
    ties_sq_g = torch.zeros(n_g, dtype=torch.float64).index_add_(0, group_idx, ties_abs.square())
    mean_sq_g = torch.zeros(n_g, dtype=torch.float64).index_add_(0, group_idx, mean_abs.square())
    del conf, ones, ties_abs, mean_abs, kp, kn, kept_count

    # ---- pass 2: per key in float64: gram matrix (norms, cosines) and per-group norms ----
    log("[stats] gram matrix / per-layer norms")
    gram = torch.zeros(T, T, dtype=torch.float64)
    norm_sq_g = np.zeros((T, n_g), dtype=np.float64)
    init_sq_g = np.zeros(n_g, dtype=np.float64)
    for k, o, m in ing.layout.slices():
        gi = gid[key_group(k)]
        init_sq_g[gi] += float(ing.base.state[k].to(torch.float64).square().sum())
        taus = [ing.tau(t, k, torch.float64) for t in range(T)]
        for a in range(T):
            norm_sq_g[a, gi] += float(taus[a].dot(taus[a]))
            for b in range(a, T):
                gram[a, b] += taus[a].dot(taus[b])
        del taus
    gram = gram + gram.triu(1).T
    norms = gram.diag().sqrt()
    cosine = (gram / (norms[:, None] * norms[None, :]).clamp(min=1e-30)).numpy()
    for t in range(T):
        models[t]["norm"] = float(norms[t])
        models[t]["norm_by_group"] = {g: float(np.sqrt(norm_sq_g[t, i])) for g, i in gid.items()}
        models[t]["rel_norm_by_group"] = {g: float(np.sqrt(norm_sq_g[t, i] / max(init_sq_g[i], 1e-30)))
                                          for g, i in gid.items()}

    per_group = {g: {"n_params": int(size_g[i]), "shared_frac": float(shared_g[i] / size_g[i]),
                     "conflict_frac": float(conf_g[i] / shared_g[i]) if shared_g[i] else 0.0,
                     "merged_frac": float(merged_g[i] / size_g[i]),
                     "ties_norm": float(np.sqrt(ties_sq_g[i])), "mean_norm": float(np.sqrt(mean_sq_g[i])),
                     "init_norm": float(np.sqrt(init_sq_g[i]))} for g, i in gid.items()}
    result = {
        "n_params": n, "T": T, "k": float(k_pct), "lambda": float(lam), "trim_scope": trim_scope,
        "merge": merge, "k_grid": k_grid, "base": {"kind": ing.base.kind, "label": ing.base.label,
                                                   "meta": ing.base.meta},
        "paths": ing.paths, "groups": group_names, "models": models,
        "conflict_vs_k": conflict_vs_k, "conflict_vs_T": conflict_vs_T,
        "kept_count_groups": {str(c): {"n": int(kc_hist[c]), "mean_abs_ties": float(kc_ties[c]),
                                       "mean_abs_mean": float(kc_mean[c])} for c in range(T + 1)},
        "agreement_bins": {f"{AGREE_EDGES[i]:.1f}-{min(AGREE_EDGES[i + 1], 1.0):.1f}": {
            "n": int(agreement_hist[i]), "mean_abs_ties": float(ag_ties[i]), "mean_abs_mean": float(ag_mean[i])}
            for i in range(6)},
        "per_group": per_group, "cosine": cosine.tolist(),
        "tau_m_norm": float(tau_m.norm()), "lam_tau_m_norm": float(lam * tau_m.norm()),
        "mean_tau_norm": float(np.sqrt(mean_sq_g.sum())),
        "merge_summary": summary,
    }
    arrays = {
        "mag_hist": mag_hist, "mag_bins": np.linspace(LOG_LO, LOG_HI, LOG_BINS + 1),
        "concentration": concentration, "q_grid": Q_GRID,
        "kept_count_hist": kc_hist, "kept_count_mean_abs": np.stack([kc_ties, kc_mean]),
        "agreement_hist": agreement_hist, "agreement_mean_abs": np.stack([ag_ties, ag_mean]),
        "agreement_edges": AGREE_EDGES[:-1], "cosine": cosine,
        "k_grid": np.array(k_grid), "conflict_vs_k": np.array([conflict_vs_k[str(k)]["conflict_frac"] for k in k_grid]),
        "shared_vs_k": np.array([conflict_vs_k[str(k)]["shared_frac"] for k in k_grid]),
        "conflict_vs_T": np.array([r["conflict_frac"] for r in conflict_vs_T]),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ties_stats.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        np.savez(out_dir / "ties_stats.npz", **arrays)
        log(f"[stats] written: {out_dir / 'ties_stats.json'}, {out_dir / 'ties_stats.npz'}")
    result["_arrays"] = arrays
    return result


def load(stats_dir):
    """(json dict, npz arrays) of a stats directory."""
    stats_dir = Path(stats_dir)
    with open(stats_dir / "ties_stats.json", encoding="utf-8") as f:
        js = json.load(f)
    return js, dict(np.load(stats_dir / "ties_stats.npz"))
