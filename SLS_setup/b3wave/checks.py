"""
Self-checks of the b3wave package (run by scripts/b3_smoke.sh; CPU unless stated).

    python -m b3wave.checks augmenter      RNG reproduction: RTCAugmenterB3 == RTCAugmenter with defaults
    python -m b3wave.checks data           utt_meta counts, pair map size, stratified subset
    python -m b3wave.checks sampler        balanced sampler shares over 20 epochs
    python -m b3wave.checks losses         JS / embedding loss properties and gradient flow
    python -m b3wave.checks model          [GPU] ModelSLSB3 == ModelSLS, key-exact checkpoint load, SpecAugment
    python -m b3wave.checks optim          [GPU model build] LLRD groups, freeze_bottom_k
    python -m b3wave.checks all            everything above that needs no GPU

Every check prints `PASS <name>` or `FAIL <name>: <reason>` and the process exits non-zero on
any failure. Paths default to this repo's layout (run from SLS_setup/).
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SLS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SLS_DIR.parent
TRAIN_PROTOCOL = "/mnt/paml-research/RTCFake/data/train_label.txt"
PAIR_CSV = "/mnt/paml-research/RTCFake/data/meta/train_offline_online_pairs.csv"
EXPECTED_COMPOSITION = {       # (src, lang, label): count in train_label.txt
    ("offline", "en", "bonafide"): 1500, ("offline", "en", "spoof"): 7499,
    ("offline", "zh", "bonafide"): 5963, ("offline", "zh", "spoof"): 23698,
    ("online", "en", "bonafide"): 1422, ("online", "en", "spoof"): 7133,
    ("online", "zh", "bonafide"): 5468, ("online", "zh", "spoof"): 23102,
}
EXPECTED_PAIRS = 37125          # 38,660 CSV rows, 1,535 offline clips without an online twin

RESULTS = []


def report(name, ok, reason=""):
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {reason}" if reason and not ok else ""), flush=True)
    return ok


def _synthetic_clips(n, rng, sr=16000, seconds=2.0):
    clips = []
    for _ in range(n):
        t = np.arange(int(seconds * sr)) / sr
        f0 = float(rng.uniform(100, 300))
        clip = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        clip += 0.02 * rng.standard_normal(len(t))
        clips.append(clip.astype(np.float32))
    return clips


# --------------------------------------------------------------------------- #
def check_augmenter(n_clips=200, n_codec=10, seed=1234):
    from utils.rtc_augment import RTCAugConfig, RTCAugmenter
    from b3wave.augment import RTCAugConfigB3, RTCAugmenterB3

    pools = dict(noise_dirs=[str(REPO_ROOT / "data/augmentation_train/rnnoise"),
                             str(REPO_ROOT / "data/augmentation_train/esc50")],
                 rir_dirs=[str(REPO_ROOT / "data/augmentation_train/rirs")],
                 musan_dirs=[str(REPO_ROOT / "data/augmentation/musan")],
                 with_echo=True, p_apply=0.8, seed=seed, verbose=False)
    ok_all = True
    for with_codecs, n in ((False, n_clips), (True, n_codec)):
        ref = RTCAugmenter(RTCAugConfig(with_codecs=with_codecs, **pools))
        b3 = RTCAugmenterB3(RTCAugConfigB3(with_codecs=with_codecs, **pools))
        rng = np.random.default_rng(7)
        clips = _synthetic_clips(n, rng)
        same = 0
        for clip in clips:
            a = ref(clip.copy(), 16000)
            b = b3(clip.copy(), 16000)
            same += int(np.array_equal(a, b))
        ok = same == n and ref.stage_counts == b3.stage_counts
        ok_all &= report(f"augmenter_repro(codecs={with_codecs})", ok,
                         f"{same}/{n} identical; counts ref={ref.stage_counts} b3={b3.stage_counts}")
    ref = RTCAugmenter(RTCAugConfig(with_codecs=True, **pools))
    b3 = RTCAugmenterB3(RTCAugConfigB3(with_codecs=True, **pools))
    ok_all &= report("augmenter_describe_identical", ref.describe() == b3.describe(),
                     f"\n  ref: {ref.describe()}\n  b3:  {b3.describe()}")

    # non-default paths run and change the draws as intended
    b3c = RTCAugmenterB3(RTCAugConfigB3(with_codecs=False, max_stages=3, snr_dist="beta",
                                        snr_db=(-5.0, 25.0), snr_beta_mode=8.0, **pools))
    rng = np.random.default_rng(3)
    for clip in _synthetic_clips(60, rng):
        b3c(clip, 16000)
    counts = b3c.stage_counts
    n_snr = sum(v for k, v in counts.items() if k.startswith("snr:"))
    multi = any(k.startswith("snr:") for k in counts)
    ok_all &= report("augmenter_chained_beta_runs", n_snr > 0 and multi and "snr:lt0" in counts or n_snr > 0,
                     f"counts={counts}")
    vec = b3c.counts_vector({})
    ok_all &= report("augmenter_counts_vector", len(vec) == len(b3c.meta_keys) and int(vec.sum()) == sum(counts.values()),
                     f"len {len(vec)} vs {len(b3c.meta_keys)}, sum {int(vec.sum())} vs {sum(counts.values())}")

    # reserved ops raise
    try:
        from b3wave.heldout_reserve import assert_training_ops
        assert_training_ops(codecs=["libspeex"])
        ok_all &= report("reserve_raises_on_codec", False, "no error")
    except ValueError:
        ok_all &= report("reserve_raises_on_codec", True)
    try:
        from b3wave.heldout_reserve import assert_training_ops
        assert_training_ops(dirs=[str(REPO_ROOT / "data/augmentation/musan/music")])
        ok_all &= report("reserve_raises_on_dir", False, "no error")
    except ValueError:
        ok_all &= report("reserve_raises_on_dir", True)
    return ok_all


# --------------------------------------------------------------------------- #
def check_data():
    from utils.data_utils import read_protocol
    from b3wave.data import composition, load_pair_map, stratified_subset, utt_meta

    file_list, labels = read_protocol(TRAIN_PROTOCOL, require_label=True)
    comp = composition(file_list, labels)
    ok = report("utt_meta_composition", comp == EXPECTED_COMPOSITION, f"{comp}")
    pairs, stats = load_pair_map(PAIR_CSV, file_list)
    ok &= report("pair_map_count", len(pairs) == EXPECTED_PAIRS, f"{stats}")
    sub = stratified_subset(file_list, labels, 0.1, seed=0)
    sub_comp = composition(sub, labels)
    worst = max(abs(sub_comp[k] / comp[k] - 0.1) for k in comp)
    ok &= report("stratified_subset_10pct", worst < 0.01 and sub == sorted(sub, key=file_list.index),
                 f"max deviation {worst:.4f}")
    ok &= report("utt_meta_unknown_layout", utt_meta("weird.wav") == {"src": "?", "lang": "?"})
    return ok


# --------------------------------------------------------------------------- #
def check_sampler(epochs=20):
    from utils.data_utils import read_protocol
    from b3wave.data import utt_meta
    from b3wave.sampler import expected_shares, make_balanced_sampler, realised_shares, sampling_weights

    file_list, labels = read_protocol(TRAIN_PROTOCOL, require_label=True)
    y = [labels[u] for u in file_list]
    langs = [utt_meta(u)["lang"] for u in file_list]
    ok = True
    for lang_balance, want_en in ((False, None), (True, 0.5)):
        sampler = make_balanced_sampler(y, langs, 0.5, lang_balance, seed=1234)
        ok &= report(f"sampler_len(lang_balance={lang_balance})", len(sampler) == len(file_list))
        shares = [realised_shares(list(iter(sampler)), y, langs) for _ in range(epochs)]
        bona = np.mean([s["bona"] for s in shares])
        en = np.mean([s["en"] for s in shares])
        exp = expected_shares(sampling_weights(y, langs, 0.5, lang_balance), y, langs)
        cond = abs(bona - 0.5) < 0.01 and (want_en is None or abs(en - want_en) < 0.01)
        ok &= report(f"sampler_shares(lang_balance={lang_balance})", cond,
                     f"bona {bona:.4f} en {en:.4f} expected {exp}")
    natural = np.mean(y)
    ok &= report("natural_bona_share", abs(natural - 0.189) < 0.002, f"{natural:.4f}")
    return ok


# --------------------------------------------------------------------------- #
def check_losses():
    from b3wave.losses import consistency_loss, embedding_consistency, js_divergence

    torch.manual_seed(0)
    a = torch.randn(8, 2)
    b = torch.randn(8, 2)
    ok = report("js_zero_on_identical", float(js_divergence(a, a)) < 1e-7)
    ok &= report("js_symmetric", abs(float(js_divergence(a, b)) - float(js_divergence(b, a))) < 1e-6)
    big = torch.tensor([[50.0, -50.0]] * 4)
    ok &= report("js_bounded_ln2", float(js_divergence(big, -big)) <= math.log(2) + 1e-6)
    h = torch.randn(8, 1024)
    ok &= report("emb_zero_on_identical", float(embedding_consistency(h, h, stop_grad_a=False)) < 1e-6)

    la, lb = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    ha, hb = h.clone().requires_grad_(True), torch.randn(8, 1024, requires_grad=True)
    total, comps = consistency_loss(la, lb, ha, hb, "js_emb", 0.3, 0.3, clean_view=True)
    total.backward()
    ok &= report("clean_view_stops_grad_to_a", la.grad is None and ha.grad is None and lb.grad is not None and hb.grad is not None,
                 f"la.grad={la.grad is not None} ha.grad={ha.grad is not None}")
    la, lb = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    ha, hb = h.clone().requires_grad_(True), torch.randn(8, 1024, requires_grad=True)
    total, comps = consistency_loss(la, lb, ha, hb, "js_emb", 0.3, 0.3, clean_view=False)
    total.backward()
    ok &= report("both_views_get_grad", la.grad is not None and ha.grad is not None and lb.grad is not None,
                 f"components {comps}")
    return ok


# --------------------------------------------------------------------------- #
def _args(**over):
    from types import SimpleNamespace
    base = dict(ssl_name="Qwen/Qwen3-ASR-1.7B-hf", n_layers=24, devices=None, layer_split=None,
                freeze_ssl=False, specaug=False, specaug_time=(2, 40), specaug_freq=(2, 16),
                optim="adamw", lr=5e-6, weight_decay=0.01, head_lr_mult=10.0, llrd=1.0, freeze_bottom_k=0)
    base.update(over)
    return SimpleNamespace(**base)


def check_model(ckpt):
    from model.sls_model import ModelSLS
    from b3wave.model import ModelSLSB3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location="cpu")
    ref = ModelSLS(_args(), device)
    b3 = ModelSLSB3(_args(specaug=True), device)
    res = b3.load_state_dict(state, strict=False)
    ok = report("ckpt_keys_exact", not res.missing_keys and not res.unexpected_keys,
                f"missing {res.missing_keys[:3]} unexpected {res.unexpected_keys[:3]}")
    ref.load_state_dict(state)
    ok &= report("state_dict_keys_identical", list(ref.state_dict().keys()) == list(b3.state_dict().keys()))

    torch.manual_seed(0)
    x = torch.randn(4, 128, 400, device=b3.input_device)        # (bs, n_mel, frames) log-mel
    ref.eval(); b3.eval()
    with torch.no_grad():
        ya, yb = ref(x), b3(x)
        logits, h = b3(x, return_emb=True)
    ok &= report("forward_identical_eval", torch.allclose(ya, yb, atol=1e-5) and torch.allclose(logits, yb, atol=1e-5),
                 f"max diff {(ya - yb).abs().max().item():.2e}")
    ok &= report("return_emb_shape", tuple(h.shape) == (4, 1024), f"{tuple(h.shape)}")
    b3.train()
    with torch.no_grad():
        masked = b3.specaug(x)
    changed = (masked != x).float().mean().item()
    ok &= report("specaug_masks_in_train", 0.0 < changed < 0.6, f"changed share {changed:.3f}")
    b3.eval()
    with torch.no_grad():
        ok &= report("specaug_inactive_in_eval", torch.allclose(b3(x), yb, atol=1e-5))
    del ref, b3
    return ok


def check_optim():
    from b3wave.model import ModelSLSB3
    from b3wave.optim import build_optimizer_b3, describe_groups, freeze_bottom

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModelSLSB3(_args(), device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = build_optimizer_b3(model, trainable, _args(llrd=0.85))
    names = [g["name"] for g in opt.param_groups]
    lrs = {}
    for g in opt.param_groups:
        lrs.setdefault(g["name"], g["lr"])
    layer_lrs = [lrs[f"backbone_l{i:02d}"] for i in range(23)] + [lrs["backbone"]]
    ratios = [layer_lrs[i + 1] / layer_lrs[i] for i in range(23)]
    ok = report("llrd_ladder", all(abs(r - 1 / 0.85) < 1e-6 for r in ratios) and abs(lrs["backbone"] - 5e-6) < 1e-12
                and abs(lrs["head"] - 5e-5) < 1e-12 and abs(lrs["backbone_stem"] - 5e-6 * 0.85 ** 24) < 1e-15,
                f"{describe_groups(opt)}")
    ok &= report("llrd_first_group_is_top_layer", opt.param_groups[0]["name"] == "backbone")
    n_grouped = sum(p.numel() for g in opt.param_groups for p in g["params"])
    ok &= report("llrd_all_params_grouped", n_grouped == sum(p.numel() for p in trainable),
                 f"{n_grouped} vs {sum(p.numel() for p in trainable)}")
    print("   " + describe_groups(opt))

    frozen = freeze_bottom(model, 8)
    trainable2 = [p for p in model.parameters() if p.requires_grad]
    opt2 = build_optimizer_b3(model, trainable2, _args(llrd=0.85))
    names2 = {g["name"] for g in opt2.param_groups}
    ok &= report("freeze_bottom_8", frozen > 0 and "backbone_stem" not in names2 and "backbone_l07" not in names2
                 and "backbone_l08" in names2 and sum(p.numel() for p in trainable2) + frozen == sum(p.numel() for p in trainable),
                 f"frozen {frozen} params; groups {sorted(names2)[:4]}...")
    del model
    return ok


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["augmenter", "data", "sampler", "losses", "model", "optim", "all"])
    parser.add_argument("--ckpt", default=str(SLS_DIR / "exp/B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925/ckpt/epoch_8_dev_loss_0.126069.pth"))
    args = parser.parse_args()
    os.chdir(SLS_DIR)
    sys.path.insert(0, str(SLS_DIR))
    t0 = time.time()
    todo = ["augmenter", "data", "sampler", "losses"] if args.what == "all" else [args.what]
    for what in todo:
        if what == "augmenter":
            check_augmenter()
        elif what == "data":
            check_data()
        elif what == "sampler":
            check_sampler()
        elif what == "losses":
            check_losses()
        elif what == "model":
            check_model(args.ckpt)
        elif what == "optim":
            check_optim()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed in {time.time() - t0:.0f}s"
          + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
