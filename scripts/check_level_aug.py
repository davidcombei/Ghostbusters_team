"""
CPU check of the level augmentation before spending GPU time (plan step 5).

    python scripts/check_level_aug.py --n 3000 --workers 48

Sends training clips through the real training augmentation path (SpoofAudioDataset._load ->
_augment, built by build_dataset_from_protocol(mode="train")) in three settings -- no
augmentation, RTC augmentation only, RTC + level augmentation -- and reports:

  * single-feature AUC (bonafide vs spoof) of active RMS and peak per (src, lang) cell:
    the level cue must be gone with --use_level_aug (|AUC - 0.5| <= 0.05)
  * the level stage chosen per clip vs label (chi-square): class independence
  * augmentation time per clip, RTC only vs RTC + level
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "SLS_setup"))

from utils.data_utils import build_dataset_from_protocol  # noqa: E402

DATA = Path("/mnt/paml-research/RTCFake/data")
SETTINGS = ("none", "rtc", "rtc+level")
DS = {}


LEVEL_KW = {}


def make_args(setting, seed):
    a = argparse.Namespace(seed=seed, use_rawboost=False, ssl_name=None, use_rtc_aug=setting != "none",
                           musan=False, aug_noise_dirs=[str(REPO / "data/augmentation_train/rnnoise"),
                                                        str(REPO / "data/augmentation_train/esc50")],
                           aug_rir_dirs=[str(REPO / "data/augmentation_train/rirs")], aug_music_dirs=[],
                           aug_p_apply=0.8, aug_max_stages=1, no_codec_aug=False, aug_codec_p=1.0,
                           use_level_aug=setting == "rtc+level", **LEVEL_KW)
    return a


def init(protocol, seed, level_kw):
    LEVEL_KW.update(level_kw)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        for s in SETTINGS:
            DS[s] = build_dataset_from_protocol(protocol, DATA / "wav" / "train", mode="train",
                                                args=make_args(s, seed))[0]


def auc_se(y, score):
    """AUC and its Hanley-McNeil standard error."""
    auc = roc_auc_score(y, score)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    return auc, se


def level_feats(audio):
    n = 160
    f = np.mean(audio[:len(audio) // n * n].reshape(-1, n) ** 2, axis=1)
    d = 10 * np.log10(f + 1e-10)
    return 10 * np.log10(np.mean(f[d > d.max() - 35]) + 1e-10), 20 * np.log10(max(np.abs(audio).max(), 1e-6))


def work(i):
    out = {}
    for s in SETTINGS:
        ds = DS[s]
        utt = ds.file_list[i]
        audio, sr = ds._load(ds.base_dir / utt)
        level = ds.augmenter.augmenters[-1] if hasattr(ds.augmenter, "augmenters") else None
        before = dict(level.stage_counts) if level is not None else {}
        t = time.perf_counter()
        aug = ds._augment(audio, sr)
        out[f"t_{s}"] = time.perf_counter() - t
        out[f"rms_{s}"], out[f"peak_{s}"] = level_feats(aug)
        if level is not None:
            changed = [k for k, v in level.stage_counts.items() if v != before.get(k, 0) and not k.startswith("overshoot")]
            out["stage"] = changed[0] if changed else "skipped"
        out["label"], out["cell"] = ds.labels[utt], "/".join(utt.split("/")[:2])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--level_gain_mode", default="absolute", choices=["absolute", "relative"])
    parser.add_argument("--level_p_apply", type=float, default=0.8)
    args = parser.parse_args()
    level_kw = {"level_gain_mode": args.level_gain_mode, "level_p_apply": args.level_p_apply}

    lines = [l for l in open(DATA / "train_label.txt", encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(args.seed)
    sub = [lines[i] for i in sorted(rng.choice(len(lines), args.n, replace=False))]
    proto = REPO / "data" / "audit" / f"level_check_{args.n}.txt"
    proto.parent.mkdir(parents=True, exist_ok=True)
    proto.write_text("".join(sub), encoding="utf-8")

    with Pool(args.workers, initializer=init, initargs=(str(proto), args.seed, level_kw)) as pool:
        rows = pool.map(work, range(len(sub)), chunksize=8)

    y = np.array([1 - r["label"] for r in rows])          # 1 = spoof
    cells = np.array([r["cell"] for r in rows])
    ok = True
    print(f"\nsingle-feature AUC (spoof) +- 1 SE (Hanley-McNeil), n={len(rows)}; "
          f"pass: |AUC-0.5| <= 0.05 + 2 SE for rtc+level")
    print(f"{'cell':<14}{'n_bona':>7}" + "".join(f"{f + ':' + s:>20}" for f in ("rms", "peak") for s in SETTINGS))
    for cell in sorted(set(cells)) + ["pooled"]:
        m = np.ones_like(y, bool) if cell == "pooled" else cells == cell
        vals = []
        for f in ("rms", "peak"):
            for s in SETTINGS:
                x = np.array([r[f"{f}_{s}"] for r in rows])
                if cell == "pooled":             # stratified: centre each cell, so cell means can't leak
                    x = x.copy()
                    for c in set(cells):
                        x[cells == c] -= np.median(x[cells == c])
                auc, se = auc_se(y[m], x[m])
                vals.append((auc, se))
                if s == "rtc+level" and abs(auc - 0.5) > 0.05 + 2 * se:
                    ok = False
        print(f"{cell:<14}{int((y[m] == 0).sum()):>7}" + "".join(f"{a:>13.3f} +-{e:.3f}" for a, e in vals))

    stages = sorted({r["stage"] for r in rows})
    table = np.array([[sum(r["stage"] == st and r["label"] == lab for r in rows) for lab in (0, 1)] for st in stages])
    p = chi2_contingency(table)[1]
    print(f"\nlevel stage x label (spoof, bonafide): {dict(zip(stages, table.tolist()))}  chi2 p={p:.3f}")
    t_rtc = np.mean([r["t_rtc"] for r in rows])
    t_lvl = np.mean([r["t_rtc+level"] for r in rows])
    print(f"augment time per clip: rtc {1000 * t_rtc:.1f} ms, rtc+level {1000 * t_lvl:.1f} ms "
          f"(+{100 * (t_lvl - t_rtc) / t_rtc:.1f} %)")
    print("\nPASS" if ok and p > 0.01 else "\nFAIL: level cue not removed or stage choice depends on label")


if __name__ == "__main__":
    main()
