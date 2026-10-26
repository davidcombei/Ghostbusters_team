"""
Paired bootstrap confidence intervals over cached local-eval scores (scripts/local_eval.py and
scripts/b3_local_eval.py write <out>/scores/<set>.txt as "utt p_spoof", in protocol order).

    paired_bootstrap(eval_dir_a, eval_dir_b, sets_yaml, names, half="B", n=2000)

Resamples utterances with replacement, the same indices for both models, and recomputes
macro-F1@0.5 per set and WF1 = 0.3 * F1(clean) + 0.7 * mean(F1(matched), F1(heldout)) on every
resample. Reports the point estimates (which reproduce metrics.json exactly), the delta (b - a)
with its 2.5 / 97.5 percentiles, and the share of resamples where b does not beat a.
Half B restricts the sets that carry a `half` column in meta.csv to that half, like
utils.local_eval.evaluate_sets. No torch / model needed.
"""

import csv
import json
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SETS = ("dev_online_clean", "sim_matched_v1", "sim_heldout_v1", "sim_echo_v1")
CLEAN_WEIGHT = 0.3
NOISY_ROLES = ("matched", "heldout")
LABEL_TO_SPOOF = {"spoof": 1, "fake": 1, "bonafide": 0, "bona-fide": 0, "real": 0}


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_protocol_labels(protocol_path):
    ids, y = [], []
    with open(protocol_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 2:
                utt, label = parts
            elif len(parts) >= 5:
                utt, label = parts[1], parts[4]
            else:
                raise ValueError(f"bad protocol line: {line.rstrip()}")
            ids.append(utt)
            y.append(LABEL_TO_SPOOF[label.strip().lower()])
    return ids, np.asarray(y, dtype=int)


def read_scores(path):
    ids, p = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                ids.append(parts[0])
                p.append(float(parts[1]))
    return ids, np.asarray(p, dtype=np.float64)


def set_info(sets_yaml, name, half=None):
    """(role, ids, y_spoof, keep_mask) of one set from configs/local_eval_sets.yaml."""
    with open(sets_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    entry = cfg[name]
    ids, y = read_protocol_labels(_resolve(entry["protocol"]))
    keep = np.ones(len(ids), dtype=bool)
    meta_path = entry.get("meta")
    if half is not None and meta_path and _resolve(meta_path).is_file():
        with open(_resolve(meta_path), newline="", encoding="utf-8") as f:
            halves = {row["path"]: row.get("half", "") for row in csv.DictReader(f)}
        marks = np.array([halves.get(u, "") for u in ids])
        if (marks != "").any():
            keep = marks == half
    return entry.get("role", "noisy"), ids, y, keep


def load_eval_dir(eval_dir, sets_yaml, names=REPORT_SETS, half="B"):
    """{name: {"role", "y", "p"}} for the requested sets, restricted to `half`."""
    eval_dir = Path(eval_dir)
    out = {}
    for name in names:
        role, ids, y, keep = set_info(sets_yaml, name, half)
        score_ids, p = read_scores(eval_dir / "scores" / f"{name}.txt")
        if score_ids != ids:
            raise ValueError(f"{eval_dir}/scores/{name}.txt is not in protocol order (or a different set)")
        out[name] = {"role": role, "y": y[keep], "p": p[keep]}
    return out


def macro_f1_counts(tp, fp, fn, tn):
    """sklearn macro-F1 over labels [0, 1] with zero_division=0, from confusion counts."""
    d_spoof = 2 * tp + fp + fn
    d_bona = 2 * tn + fn + fp
    f_spoof = np.where(d_spoof > 0, 2 * tp / np.maximum(d_spoof, 1), 0.0)
    f_bona = np.where(d_bona > 0, 2 * tn / np.maximum(d_bona, 1), 0.0)
    return (f_spoof + f_bona) / 2


def _categories(y, p, thr):
    pred = (p >= thr).astype(int)
    # 0: tn (y=0,pred=0)  1: fp (y=0,pred=1)  2: fn (y=1,pred=0)  3: tp (y=1,pred=1)
    return y * 2 + pred


def _f1_from_cat(cat_counts):
    tn, fp, fn, tp = cat_counts[..., 0], cat_counts[..., 1], cat_counts[..., 2], cat_counts[..., 3]
    return macro_f1_counts(tp, fp, fn, tn)


def wf1_from_sets(f1_by_name, roles):
    clean = [f for n, f in f1_by_name.items() if roles[n] == "clean"]
    noisy = [f for n, f in f1_by_name.items() if roles[n] in NOISY_ROLES]
    if not clean or not noisy:
        return None
    return CLEAN_WEIGHT * clean[0] + (1 - CLEAN_WEIGHT) * np.mean(noisy, axis=0)


def paired_bootstrap(eval_dir_a, eval_dir_b, sets_yaml, names=REPORT_SETS, half="B", n=2000, seed=0, thr=0.5):
    a = load_eval_dir(eval_dir_a, sets_yaml, names, half)
    b = load_eval_dir(eval_dir_b, sets_yaml, names, half)
    rng = np.random.default_rng(seed)
    roles = {name: a[name]["role"] for name in names}
    point_a, point_b, boots_a, boots_b = {}, {}, {}, {}
    for name in names:
        y = a[name]["y"]
        if not np.array_equal(y, b[name]["y"]):
            raise ValueError(f"{name}: labels differ between the two evals")
        cat_a = _categories(y, a[name]["p"], thr)
        cat_b = _categories(y, b[name]["p"], thr)
        point_a[name] = float(_f1_from_cat(np.bincount(cat_a, minlength=4)))
        point_b[name] = float(_f1_from_cat(np.bincount(cat_b, minlength=4)))
        N = len(y)
        idx = rng.integers(0, N, size=(n, N))
        counts_a = np.stack([np.bincount(cat_a[row], minlength=4) for row in idx])
        counts_b = np.stack([np.bincount(cat_b[row], minlength=4) for row in idx])
        boots_a[name] = _f1_from_cat(counts_a)
        boots_b[name] = _f1_from_cat(counts_b)

    def summarise(va, vb):
        delta = vb - va
        return {"a": float(np.mean(va)), "b": float(np.mean(vb)), "delta_mean": float(np.mean(delta)),
                "ci_low": float(np.percentile(delta, 2.5)), "ci_high": float(np.percentile(delta, 97.5)),
                "p_not_better": float(np.mean(delta <= 0))}

    result = {"n": n, "half": half, "seed": seed, "thr": thr, "eval_a": str(eval_dir_a), "eval_b": str(eval_dir_b),
              "sets": {}, "wf1": None}
    for name in names:
        result["sets"][name] = {"role": roles[name], "point_a": point_a[name], "point_b": point_b[name],
                                "delta": point_b[name] - point_a[name],
                                **summarise(boots_a[name], boots_b[name])}
    wa, wb = wf1_from_sets(point_a, roles), wf1_from_sets(point_b, roles)
    if wa is not None:
        ba, bb = wf1_from_sets(boots_a, roles), wf1_from_sets(boots_b, roles)
        result["wf1"] = {"point_a": float(wa), "point_b": float(wb), "delta": float(wb - wa), **summarise(ba, bb)}
    return result


def format_ci(result, label_a="A", label_b="B"):
    lines = [f"paired bootstrap n={result['n']} half={result['half']} ({label_b} - {label_a}), 95 % CI of the delta:"]
    if result["wf1"]:
        w = result["wf1"]
        lines.append(f"  {'wf1':24} {100 * w['point_a']:7.2f} {100 * w['point_b']:7.2f} "
                     f"{100 * w['delta']:+7.2f}  [{100 * w['ci_low']:+6.2f}, {100 * w['ci_high']:+6.2f}]"
                     f"  p(not better)={w['p_not_better']:.3f}")
    for name, s in result["sets"].items():
        lines.append(f"  {name + ' f1':24} {100 * s['point_a']:7.2f} {100 * s['point_b']:7.2f} "
                     f"{100 * s['delta']:+7.2f}  [{100 * s['ci_low']:+6.2f}, {100 * s['ci_high']:+6.2f}]")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_a")
    parser.add_argument("eval_b")
    parser.add_argument("--sets", default=str(REPO_ROOT / "configs/local_eval_sets.yaml"))
    parser.add_argument("--names", nargs="+", default=list(REPORT_SETS))
    parser.add_argument("--half", default="B")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="write the result here")
    args = parser.parse_args()
    res = paired_bootstrap(args.eval_a, args.eval_b, args.sets, args.names, args.half, args.n, args.seed)
    print(format_ci(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
