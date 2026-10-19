"""
Self-checks of the ties_merging package. Run from SLS_setup/:

    python -m ties_merging.checks cpu                          synthetic state dicts, seconds
    python -m ties_merging.checks real --merge_list <txt>      first two checkpoints of a real merge
                                                               list + the pretrained base (CPU, minutes)
    python -m ties_merging.checks real --merge_list <txt> --with_model
                                                               also builds ModelSLS to compare the
                                                               safetensors base with the model's init

    single CPU groups: trim | elect | disjoint | identity | mean_equivalence | kept_fraction |
                       lambda_cache | dtypes_keys | stats_totals | module
    all               = cpu

Every check prints `PASS <name>`, `FAIL <name>: <reason>` or `SKIP <name>: <reason>`; measured
values go on indented lines below; the process exits non-zero on any FAIL.
"""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SLS_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if str(SLS_DIR) not in sys.path:
    sys.path.insert(0, str(SLS_DIR))

from ties_merging import merge as M  # noqa: E402
from ties_merging import module_sha256  # noqa: E402
from ties_merging.base import BaseWeights, checkpoint_base, pretrained_backbone  # noqa: E402

RESULTS = []


def report(name, ok, reason="", values=None):
    ok = None if ok is None else bool(ok)          # numpy / torch bools compare by identity otherwise
    tag = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
    print(f"{tag} {name}" + (f": {reason}" if reason else ""), flush=True)
    for k, v in (values or {}).items():
        print(f"    {k} = {v}", flush=True)
    RESULTS.append((name, ok))


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
SYN_KEYS = {  # ModelSLS-like key names: backbone (TIES), head (mean), BN buffers, int64 counter
    "ssl_model.model.layers.0.w": (3, 4),
    "ssl_model.model.ln_post.b": (4,),
    "fc1.weight": (2, 3),
    "first_bn.running_mean": (1,),
    "first_bn.num_batches_tracked": (),
}


def syn_state(seed, base=None, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    st = {}
    for k, shape in SYN_KEYS.items():
        if k.endswith("num_batches_tracked"):
            st[k] = torch.tensor(1000 + seed, dtype=torch.int64)
        else:
            v = torch.randn(shape, generator=g) * scale
            st[k] = (base[k] + v) if base is not None and k in base else v
    return st


def syn_base(seed=0):
    g = torch.Generator().manual_seed(seed)
    st = {k: torch.randn(shape, generator=g) for k, shape in SYN_KEYS.items() if k.startswith("ssl_model.")}
    return BaseWeights("pretrained", st, "synthetic-pretrained", {})


def save_states(states, tmp):
    paths = []
    for i, st in enumerate(states):
        p = Path(tmp) / f"syn_{i}.pth"
        torch.save(st, p)
        paths.append(str(p))
    return paths


def flat_of(st, base):
    """Flat task vector over the base keys in state order (reference implementation)."""
    return torch.cat([(st[k].double() - base.state[k].double()).reshape(-1) for k in st if k in base.state])


# --------------------------------------------------------------------------- #
# CPU checks
# --------------------------------------------------------------------------- #
def check_module():
    ok = True
    try:
        sha = module_sha256()
    except Exception as e:  # noqa: BLE001
        report("module", False, f"module_sha256 failed: {e}")
        return
    report("module", ok, values={"sha256": sha[:12], "merge_modes": M.MERGE_MODES, "trim_scopes": M.TRIM_SCOPES})


def check_trim():
    flat = torch.tensor([0.0, -5.0, 1.0, 2.0, -3.0, 4.0, 0.5, -0.25, 6.0, -7.0, 0.75, 1.5])
    # 12 entries, k = 25% -> keep 3 largest |.|: -7, 6, -5
    mask, thr = M.trim_mask(flat, 25.0)
    exp = torch.zeros(12, dtype=torch.bool)
    exp[[1, 8, 9]] = True
    ok = torch.equal(mask, exp) and thr == 5.0
    # k = 100 keeps every non-zero entry, k = 0 none
    m100, t100 = M.trim_mask(flat, 100.0)
    m0, t0 = M.trim_mask(flat, 0.0)
    ok &= torch.equal(m100, flat != 0) and t100 == 0.0 and int(m0.sum()) == 0 and t0 == math.inf
    # ties at the threshold: >= keeps both
    tie = torch.tensor([1.0, 2.0, 2.0, 3.0])
    mt, tt = M.trim_mask(tie, 50.0)        # keep 2 -> threshold 2.0 -> three entries survive
    ok &= tt == 2.0 and int(mt.sum()) == 3
    # tensor scope: per-slice thresholds
    lay = M.Layout(["a", "b"], {"a": (6,), "b": (6,)}, {"a": (0, 6), "b": (6, 6)}, 12)
    ms, ts = M.trim_mask(flat, 50.0, "tensor", lay)
    ok &= int(ms[:6].sum()) == 3 and int(ms[6:].sum()) == 3 and ts["a"] == 3.0 and ts["b"] == 1.5
    report("trim", ok, values={"threshold_k25": thr, "kept_k25": mask.nonzero().flatten().tolist(),
                               "tensor_thresholds": ts})


def _three_model_fixture(tmp):
    """Three models whose task vectors on layers.0.w (12 entries) are hand-written."""
    base = syn_base(0)
    # zero base for the hand-written tensor so the float32 checkpoints hold tau exactly
    base.state["ssl_model.model.layers.0.w"] = torch.zeros(3, 4)
    taus = torch.tensor([
        #  p0    p1    p2    p3    p4    p5    p6    p7    p8    p9   p10   p11
        [ 2.0,  1.0,  0.0,  3.0, -1.0,  0.5,  0.0,  4.0, -2.0,  0.1,  0.0,  1.0],
        [ 4.0, -1.0,  0.0,  3.0, -3.0,  0.0,  0.0,  4.0,  2.0,  0.1,  0.0,  1.0],
        [-1.0,  0.0,  0.0,  3.0, -2.0,  0.0,  0.0,  4.0,  0.0,  0.1,  0.0, -1.0],
    ], dtype=torch.float64)
    states = []
    for t in range(3):
        st = syn_state(10 + t, None)
        st["ssl_model.model.layers.0.w"] = (base.state["ssl_model.model.layers.0.w"].double()
                                            + taus[t].view(3, 4)).float()
        st["ssl_model.model.ln_post.b"] = base.state["ssl_model.model.ln_post.b"].clone()   # tau = 0
        states.append(st)
    return base, taus, states, save_states(states, tmp)


def check_elect():
    with tempfile.TemporaryDirectory() as tmp:
        base, taus, states, paths = _three_model_fixture(tmp)
        ing = M.Ingredients(paths, base)
        el = M.elect_signs(ing, 100.0)
        g = el.gamma[:12].tolist()
        # float32 round trip of base + tau may not give the exact tau, but signs and zeros are exact
        exp = [1, 0, 0, 1, -1, 1, 0, 1, 0, 1, 0, 1]      # p1: +1-1 = 0 cancel; p8: -2+2 = 0 cancel
        s = el.summary()
        ok = g == exp and s["n_gamma_zero_cancel"] == 2 and el.kept == [9, 8, 6]
        ok &= s["n_conflict"] == 4        # p0, p1, p8, p11 carry both signs among >= 2 keepers
        report("elect", ok, values={"gamma[:12]": g, "kept": el.kept, "summary": {k: s[k] for k in
                                    ("n_conflict", "n_gamma_zero_cancel", "shared_frac", "conflict_frac")}})


def check_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        base, taus, states, paths = _three_model_fixture(tmp)
        ing = M.Ingredients(paths, base)
        el = M.elect_signs(ing, 100.0)
        dis, cnt = M.disjoint_merge(ing, el, "dis-mean")
        over_t, _ = M.disjoint_merge(ing, el, "mean_over_T")
        el_noelect = M.elect_signs(ing, 100.0, elect=False)
        noel, _ = M.disjoint_merge(ing, el_noelect, "dis-mean")
        # the float32 checkpoints hold base + tau rounded; recompute the exact stored taus
        st = torch.stack([(s["ssl_model.model.layers.0.w"].double()
                           - base.state["ssl_model.model.layers.0.w"].double()).reshape(-1) for s in states])
        gamma = torch.sign(st.sum(0))
        sel = (st != 0) & (torch.sign(st) == gamma)
        exp_dis = (st * sel).sum(0) / sel.sum(0).clamp(min=1)
        exp_over = (st * sel).sum(0) / 3
        exp_noel = (st * (st != 0)).sum(0) / (st != 0).sum(0).clamp(min=1)
        ok = (torch.allclose(dis[:12], exp_dis, atol=1e-12) and torch.allclose(over_t[:12], exp_over, atol=1e-12)
              and torch.allclose(noel[:12], exp_noel, atol=1e-12) and torch.equal(cnt[:12], sel.sum(0).to(cnt.dtype))
              and bool((dis[12:] == 0).all()))
        # hand values: p0 -> (2+4)/2 = 3, p3 -> 3, p4 -> -2, p7 -> 4, p11 -> (1+1)/2 = 1, p1/p8 cancel -> 0
        hand = {0: 3.0, 3: 3.0, 4: -2.0, 7: 4.0, 11: 1.0, 1: 0.0, 8: 0.0}
        ok &= all(abs(dis[i].item() - v) < 1e-6 for i, v in hand.items())
        report("disjoint", ok, values={"dis_mean[:12]": [round(x, 4) for x in dis[:12].tolist()],
                                       "mean_over_T[:12]": [round(x, 4) for x in over_t[:12].tolist()],
                                       "count[:12]": cnt[:12].tolist()})


def check_identity():
    with tempfile.TemporaryDirectory() as tmp:
        base = syn_base(0)
        st = syn_state(3, base.state)
        (p,) = save_states([st], tmp)
        out, summ = M.ties_merge([p], base, 100.0, 1.0)
        ok = list(out) == list(st) and all(torch.equal(out[k], st[k]) for k in st)
        ok &= all(out[k].dtype == st[k].dtype for k in st)
        # checkpoint base, T = 1: tau = 0 everywhere -> identity too, head included
        out2, _ = M.ties_merge([p], checkpoint_base(p), 20.0, 1.0)
        ok &= all(torch.equal(out2[k], st[k]) for k in st)
        report("identity", ok, values={"kept_frac": summ["kept_frac"], "n_ties_keys": summ["n_ties_keys"]})


def check_mean_equivalence():
    sys.path.insert(0, str(SLS_DIR))
    from main_eval import merge_checkpoints
    with tempfile.TemporaryDirectory() as tmp:
        base = syn_base(0)
        # all task vectors share one sign pattern -> no conflicts; k = 100 -> dis-mean == plain mean
        sign = torch.sign(torch.randn(3, 4, generator=torch.Generator().manual_seed(7)))
        states = []
        for t in range(3):
            st = syn_state(20 + t, base.state)
            st["ssl_model.model.layers.0.w"] = base.state["ssl_model.model.layers.0.w"] + sign * (1 + t) * 0.1
            st["ssl_model.model.ln_post.b"] = base.state["ssl_model.model.ln_post.b"] + 0.01 * (t + 1)
            states.append(st)
        paths = save_states(states, tmp)
        out, summ = M.ties_merge(paths, base, 100.0, 1.0)
        ref = merge_checkpoints(paths)
        ties_keys = [k for k in st if k.startswith("ssl_model.")]
        other = [k for k in st if k not in ties_keys]
        max_diff = max((out[k].double() - ref[k].double()).abs().max().item() for k in ties_keys)
        ok = max_diff < 1e-6 and all(torch.equal(out[k], ref[k]) for k in other)
        ok &= summ["n_conflict"] == 0
        m = M.Merger(paths, base)
        pm = m.plain_mean()
        ok &= all(torch.equal(pm[k], ref[k]) for k in ref)
        report("mean_equivalence", ok, values={"max_abs_diff_ties_keys": max_diff,
                                               "head_and_buffers_bit_exact": all(torch.equal(out[k], ref[k]) for k in other),
                                               "plain_mean_bit_exact": all(torch.equal(pm[k], ref[k]) for k in ref)})


def check_kept_fraction():
    g = torch.Generator().manual_seed(1)
    flat = torch.randn(200_000, generator=g)
    oks, vals = [], {}
    for k in (5.0, 20.0, 50.0):
        mask, thr = M.trim_mask(flat, k)
        frac = int(mask.sum()) / flat.numel()
        oks.append(abs(frac - k / 100) < 1e-5)
        vals[f"k{int(k)}"] = (frac, thr)
    report("kept_fraction", all(oks), values=vals)


def check_lambda_cache():
    with tempfile.TemporaryDirectory() as tmp:
        base = syn_base(0)
        states = [syn_state(30 + t, base.state) for t in range(4)]
        paths = save_states(states, tmp)
        m = M.Merger(paths, base)
        tau, s1 = m.tau(20.0)
        tau2, s2 = m.tau(20.0)
        ok = tau is tau2
        st05 = m.state(tau, 0.5)
        st0 = m.state(tau, 0.0)
        fresh, _ = M.ties_merge(paths, base, 20.0, 0.5)
        ok &= all(torch.equal(st05[k], fresh[k]) for k in fresh)
        ok &= all(torch.equal(st0[k], base.state[k]) for k in base.state)
        # mean keys are shared across cells
        ok &= st05["fc1.weight"] is st0["fc1.weight"]
        report("lambda_cache", ok, values={"kept_frac": [round(x, 3) for x in s1["kept_frac"]],
                                           "tau_m_norm": s1["tau_m_norm"]})


def check_dtypes_keys():
    with tempfile.TemporaryDirectory() as tmp:
        base = syn_base(0)
        states = [syn_state(40 + t, base.state) for t in range(3)]
        paths = save_states(states, tmp)
        out, _ = M.ties_merge(paths, base, 20.0, 1.0)
        st = states[0]
        ok = list(out) == list(st)
        ok &= all(out[k].dtype == st[k].dtype and out[k].shape == st[k].shape for k in st)
        nbt = out["first_bn.num_batches_tracked"]
        exp_nbt = (sum(s["first_bn.num_batches_tracked"].double() for s in states) / 3).to(torch.int64)
        ok &= nbt.dtype == torch.int64 and torch.equal(nbt, exp_nbt)
        ok &= all(out[k].is_contiguous() for k in st)
        report("dtypes_keys", ok, values={"num_batches_tracked": int(nbt), "keys": len(out)})


def check_stats_totals():
    try:
        from ties_merging import stats as S
    except ImportError as e:
        report("stats_totals", None, f"stats module not importable: {e}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        base = syn_base(0)
        states = [syn_state(50 + t, base.state) for t in range(3)]
        paths = save_states(states, tmp)
        m = M.Merger(paths, base)
        res = S.analyze(m, k_pct=50.0, k_grid=(20.0, 50.0, 100.0), out_dir=Path(tmp), lam=1.0, log=lambda *a: None)
        js = json.loads((Path(tmp) / "ties_stats.json").read_text())
        npz = np.load(Path(tmp) / "ties_stats.npz")
        n = m.ing.n
        ok = js["n_params"] == n and len(js["models"]) == 3
        ok &= all(int(npz["mag_hist"][t].sum()) == n - js["models"][t]["n_exact_zero"] for t in range(3))
        ok &= int(npz["kept_count_hist"].sum()) == n
        ok &= abs(npz["concentration"][0][-1] - 1.0) < 1e-9
        ok &= int(npz["agreement_hist"].sum()) == int(npz["kept_count_hist"][2:].sum())
        report("stats_totals", ok, values={"n": n, "keys": sorted(npz.files)[:8],
                                           "conflict_vs_k": js["conflict_vs_k"]})


CPU_GROUPS = ["module", "trim", "elect", "disjoint", "identity", "mean_equivalence", "kept_fraction",
              "lambda_cache", "dtypes_keys", "stats_totals"]


# --------------------------------------------------------------------------- #
# real checkpoints
# --------------------------------------------------------------------------- #
def check_real(merge_list, with_model=False):
    sys.path.insert(0, str(SLS_DIR.parent / "scripts"))
    from local_eval import read_merge_list
    from main_eval import merge_checkpoints
    from ties_merging.base import model_args_from_runs, runs_of
    paths = read_merge_list(merge_list)
    if len(paths) < 2:
        report("real", None, "merge list has fewer than two checkpoints")
        return
    p0, p1 = paths[0], paths[1]
    runs = runs_of(paths)
    ssl, nl, _, _, warns = model_args_from_runs(runs)
    base = pretrained_backbone(ssl, nl)
    s0 = M.load_state(p0)
    covered = [k for k in s0 if k in base.state]
    report("real_base_coverage", len(covered) == sum(k.startswith("ssl_model.") for k in s0),
           values={"covered": len(covered), "backbone_keys": sum(k.startswith("ssl_model.") for k in s0),
                   "base": base.meta, "warnings": warns})

    # 1. duplicate list, k = 100, lambda = 1: bit-exact checkpoint and bit-exact merge_checkpoints
    out, summ = M.ties_merge([p0, p0], base, 100.0, 1.0)
    ref = merge_checkpoints([p0, p0])
    ok = all(torch.equal(out[k], s0[k]) for k in s0) and all(torch.equal(out[k], ref[k]) for k in ref)
    report("real_duplicate_identity", ok, values={"kept_frac": [round(x, 6) for x in summ["kept_frac"]],
                                                  "n_conflict": summ["n_conflict"]})
    del out, ref

    # 2. two epochs, k = 100, lambda = 1: head bit-exact vs merge_checkpoints, backbone equal where no conflict
    m = M.Merger([p0, p1], base)
    el = m.election(100.0)
    tau_m, summ = m.tau(100.0)
    out = m.state(tau_m, 1.0)
    ref = merge_checkpoints([p0, p1])
    head = [k for k in s0 if k not in m.ing.layout.offsets]
    head_ok = all(torch.equal(out[k], ref[k]) for k in head)
    # positions kept by both with one sign: dis-mean == plain mean; a position kept by one model only
    # gets its full value from TIES (|A^p| = 1) and half of it from the mean, so those are excluded
    both_agree = ((el.pos + el.neg) == 2) & ~((el.pos > 0) & (el.neg > 0))
    max_diff, n_cmp = 0.0, 0
    for k, o, mm in m.ing.layout.slices():
        d = (out[k].double() - ref[k].double()).reshape(-1).abs()
        d = d[both_agree[o:o + mm]]
        n_cmp += d.numel()
        if d.numel():
            max_diff = max(max_diff, d.max().item())
    report("real_k100_vs_mean", head_ok and max_diff < 2e-6,
           values={"head_bit_exact": head_ok, "backbone_max_abs_diff_both_agree": max_diff,
                   "positions_compared_frac": n_cmp / m.ing.n,
                   "conflict_frac_k100": summ["conflict_frac"], "shared_frac": summ["shared_frac"]})
    del out, ref, el, tau_m, m

    # 3. default setting on the two epochs: kept fraction, conflict fraction, finite output
    out, summ = M.ties_merge([p0, p1], base, 20.0, 1.0)
    finite = all(torch.isfinite(v).all() for v in out.values() if v.is_floating_point())
    report("real_k20_default", finite and all(abs(f - 0.2) < 1e-4 for f in summ["kept_frac"]),
           values={"kept_frac": summ["kept_frac"], "conflict_frac": summ["conflict_frac"],
                   "merged_frac": summ["merged_frac"], "tau_m_norm": summ["tau_m_norm"],
                   "thresholds": summ["thresholds"]})
    del out

    # 4. checkpoint base (head included): [p0] vs base p0 -> tau = 0 -> p0 at any k;
    #    [p1] vs base p0 at k = 100, lambda = 1 -> p0 + (p1 - p0) = p1 bit-exact (float64 path)
    out_a, summ_a = M.ties_merge([p0], checkpoint_base(p0), 20.0, 1.0)
    out_b, _ = M.ties_merge([p1], checkpoint_base(p0), 100.0, 1.0)
    s1 = M.load_state(p1)
    ok_a = all(torch.equal(out_a[k], s0[k]) for k in s0)
    ok_b = all(torch.equal(out_b[k], s1[k]) for k in s1)
    report("real_checkpoint_base_identity", ok_a and ok_b,
           values={"same_ckpt_as_base": ok_a, "k100_other_ckpt": ok_b, "n_ties_keys": summ_a["n_ties_keys"],
                   "kept_frac_zero_tau": summ_a["kept_frac"]})
    del out_a, out_b, s1

    # 5. safetensors base == ModelSLS init (optional: builds the model)
    if with_model:
        from ties_merging.base import pretrained_backbone_from_model
        bm = pretrained_backbone_from_model(ssl, nl)
        same_keys = set(bm.state) == set(base.state)
        exact = same_keys and all(torch.equal(bm.state[k], base.state[k]) for k in base.state)
        report("real_base_matches_model", exact, values={"same_keys": same_keys, "n": len(base.state)})
    else:
        report("real_base_matches_model", None, "pass --with_model to build ModelSLS and compare")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="cpu", help="cpu | real | one group name")
    ap.add_argument("--merge_list", type=str, default=None)
    ap.add_argument("--with_model", action="store_true")
    args = ap.parse_args()
    os.chdir(SLS_DIR)
    torch.manual_seed(0)
    if args.what == "cpu":
        for g in CPU_GROUPS:
            globals()[f"check_{g}"]()
    elif args.what == "real":
        if not args.merge_list:
            sys.exit("real needs --merge_list <txt>")
        check_real(args.merge_list, args.with_model)
    elif f"check_{args.what}" in globals():
        globals()[f"check_{args.what}"]()
    else:
        sys.exit(f"unknown target {args.what}; groups: {CPU_GROUPS}")
    n_fail = sum(1 for _, ok in RESULTS if ok is False)
    n_skip = sum(1 for _, ok in RESULTS if ok is None)
    print(f"\n{len(RESULTS) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
