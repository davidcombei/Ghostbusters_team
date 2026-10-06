"""
Local evaluation metrics for RTC-SDD.

Convention: y_spoof = 1 for spoof, 0 for bonafide (i.e. 1 - LABEL_TO_ID[label]);
p_spoof is P(spoof) = softmax(logits)[:, 0], as written by main_eval.write_scores.

The challenge score is 0.3 * MacroF1(clean) + 0.7 * MacroF1(noisy) at a fixed 0.5
threshold; wf1_local mirrors it on the local proxy sets.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_curve

CLEAN_WEIGHT = 0.3


def _arrays(y_spoof, p_spoof):
    y = np.asarray(y_spoof, dtype=int).reshape(-1)
    p = np.asarray(p_spoof, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"y and p differ in length: {y.shape} vs {p.shape}")
    return y, np.nan_to_num(p, nan=0.5)


def macro_f1(y_spoof, p_spoof, thr=0.5):
    y, p = _arrays(y_spoof, p_spoof)
    return float(f1_score(y, (p >= thr).astype(int), average="macro", labels=[0, 1], zero_division=0))


def eer(y_spoof, p_spoof):
    """Equal error rate and the threshold where it is reached."""
    y, p = _arrays(y_spoof, p_spoof)
    if y.min() == y.max():
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, p)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(min(thr[i], 1.0))


def oracle_f1(y_spoof, p_spoof):
    """Best macro-F1 over all thresholds (cut points at the unique scores), and that threshold."""
    y, p = _arrays(y_spoof, p_spoof)
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    n_pos, n = int(y.sum()), len(y)
    n_neg = n - n_pos
    # predicting spoof for the top-k scores, k = 0..n; only cut between distinct scores
    tp = np.concatenate([[0], np.cumsum(y_sorted)])
    fp = np.concatenate([[0], np.cumsum(1 - y_sorted)])
    fn = n_pos - tp
    tn = n_neg - fp
    valid = np.concatenate([[True], p_sorted[1:] != p_sorted[:-1], [True]])
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_spoof = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_bona = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    f1 = np.where(valid, (f1_spoof + f1_bona) / 2, -1.0)
    k = int(np.argmax(f1))
    if k == 0:
        thr = float(np.nextafter(p_sorted[0], np.inf)) if n else 0.5
    else:
        thr = float(p_sorted[k - 1])            # predict spoof for p >= thr
    return float(f1[k]), thr


def ece(y_spoof, p_spoof, n_bins=15):
    """Expected calibration error of P(spoof) with equal-width bins."""
    y, p = _arrays(y_spoof, p_spoof)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def recalls(y_spoof, p_spoof, thr=0.5):
    """(spoof recall, bonafide recall) at `thr`."""
    y, p = _arrays(y_spoof, p_spoof)
    pred = p >= thr
    spoof = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    bona = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return spoof, bona


def summarize(y_spoof, p_spoof, thr=0.5):
    y, p = _arrays(y_spoof, p_spoof)
    f1_o, thr_o = oracle_f1(y, p)
    e, thr_e = eer(y, p)
    rec_s, rec_b = recalls(y, p, thr)
    return {
        "n": int(len(y)),
        "n_spoof": int(y.sum()),
        "f1": macro_f1(y, p, thr),
        "f1_oracle": f1_o,
        "thr_oracle": thr_o,
        "eer": e,
        "thr_eer": thr_e,
        "ece": ece(y, p),
        "recall_spoof": rec_s,
        "recall_bona": rec_b,
    }


def breakdown(rows, y_spoof, p_spoof, key, thr=0.5, min_n=20):
    """summarize() per value of rows[i][key]; rows is a list of dicts aligned with y/p."""
    y, p = _arrays(y_spoof, p_spoof)
    values = np.array([str(r.get(key, "")) or "(none)" for r in rows])
    out = {}
    for v in sorted(set(values)):
        m = values == v
        if m.sum() >= min_n:
            out[v] = summarize(y[m], p[m], thr)
    return out


def wf1_local(clean, noisy_list, thr=0.5):
    """
    clean / each noisy item: (y_spoof, p_spoof). Returns (wf1, f1_clean, f1_noisy) with
    f1_noisy the mean macro-F1 over the noisy sets.
    """
    f_clean = macro_f1(*clean, thr)
    f_noisy = float(np.mean([macro_f1(*n, thr) for n in noisy_list]))
    return CLEAN_WEIGHT * f_clean + (1 - CLEAN_WEIGHT) * f_noisy, f_clean, f_noisy
