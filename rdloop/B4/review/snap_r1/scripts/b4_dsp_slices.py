#!/usr/bin/env python3
"""
b4_dsp_slices.py: the mechanism readout of rdloop cycle B4 (plan_agreed.md section 1.1 and section 3,
owner notes 2026-10-17 13:48 = critic O3.1-O3.3). CPU only; reads `scores/*.{txt,logits}`, `metrics.json`,
`probe_report.json` of local_eval readouts, the sim `meta.csv` files, the dev source wavs (tertiles) and the
arm's `logs/train.log` / `config.yaml`. Never reads progress-split files.

    python scripts/b4_dsp_slices.py --run SLS_setup/exp/<arm>/local_eval/epoch_8 \
        [--ref <B2f same readout>] [--ref_pair <B2f merge> <B2f final>] [--run_pair <arm merge> <arm final>] \
        [--null_panel <final readout dirs>] [--null_panel_merge <merge readout dirs>] \
        [--placebo <other B4 arms' readout dirs, same readout as --run>] [--lever apm_ns|dtx_gate|opus_plc|none] \
        [--summary] [--attribution_control <control merge> <control final>] [--halves AB|B] [--json_out f.json]

Statistic (per contrast c, readout r): d_i = clip(logit_i(arm), +-8) - clip(logit_i(B2f, r), +-8) over the bona
clips of the slice; DiD = mean_T(d) - mean_C(d) with a 2,000x clip-bootstrap 95 % CI (negative = the treated slice
moved toward bona). Calibrated on the pinned null panel (9 earlier runs that never saw NS / VAD / PLC; B2a printed,
flagged "failed run, excluded"): DiD* = DiD - mu, z = DiD* / sd, bar = min(mu - 2 sd, panel min, mu - 0.4 G).
Verdicts (shown / directional / inconclusive / refuted), placebo arms, the run-anomaly flag (O3.1), the heldout-F1
outcome table with the panel SD and clearing counts (O3.2) and the missing-placebo rule (O3.3): see
SLS_setup/b4wave/README.md "Mechanism readout and verdict rules".

`--run` is the readout the verdict is issued on (primary: the final epoch); condition (d) and the outcome table
use both readouts of the arm (`--run_pair`, default: the arm's local_eval/merge_top5 and its highest epoch_N).
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "SLS_setup" / "exp"
SIM = REPO / "data" / "dev_noisy_sim"
DEV_WAV = Path("/mnt/paml-research/RTCFake/data/wav/dev")
B2F = EXP / "B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623"
PANEL = [  # (label, run dir name, final readout); merge side = local_eval/merge_top5
    ("B2b", "B2b_qwen17_rtc_musan_lvl_cos8_lr1e5_s1234_epoch8_bs16_20261008040736", "epoch_8"),
    ("B2c", "B2c_qwen17_rtc_musan_lvl_cos8_lr5e6_s1234_epoch8_bs16_20261008085023", "epoch_8"),
    ("B2d", "B2d_qwen17_rtc_musan_lvl_cos10_lr5e6_s1234_epoch10_bs16_20261008132750", "epoch_10"),
    ("B2e", "B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925", "epoch_8"),
    ("B3a", "B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846", "epoch_8"),
    ("B3b", "B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234_epoch8_bs8_20261010145615", "epoch_8"),
    ("B3c", "B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234_epoch8_bs8_20261010234200", "epoch_8"),
    ("B3d", "B3d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_codecrand_s1234_epoch8_bs8_20261011081946", "epoch_8"),
    ("B3e", "B3e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_snr0to20_s1234_epoch8_bs8_20261011170031", "epoch_8"),
]
EXCLUDED = ("B2a", "B2a_qwen17_rtc_musan_lvl_cos8_s1234_epoch8_bs16_20261007231952", "epoch_8")
CLIP = 8.0
N_BOOT = 2000
SAT = 0.99                      # saturated false alarm: bona with P(spoof) >= 0.99
THR = 0.5                       # decision threshold (utils/metrics.py: spoof if p >= 0.5)
LEVER_TARGET = {"apm_ns": "NS", "dtx_gate": "VAD", "opus_plc": "PLC_online"}
PRIMARY = ("NS", "VAD", "PLC_online")
# tertiles of the clean source's non-speech share (pinned, plan section 1.1 / O2.4)
N_WIN, FRAME, N_FRAMES, LOW_MAX, MID_MAX = 64600, 320, 201, 32, 50
# shared falsifiers / guards (plan section 2.0; B2f values read from its train.log / metrics.json)
UNDERFIT_ACC_DROP, UNDERFIT_LOSS_MULT, CLEAN_DROP = 2.0, 1.5, 0.5
HELDOUT_THRESHOLDS = (0.5, 0.8)
# opus_plc decision lines (plan 2.3 predicted signature / falsifiers; O3.2)
OPUS_APP_GUARD, OPUS_APP_DROP = ("opus:qq", "opus:wechat"), 6.0     # matched bona recall per app: down <= 6 points
MATCHED_DROP, WF1_FALSIFIER, EP1_ACC_BAND, ATTRIBUTION_SHARE = 0.5, -0.5, 2.0, 0.5


def _has(chain, key, value=None):
    return any(key in st and (value is None or st[key] == value) for st in chain)


# name -> (set, src, treated, control, kind); predicates on the parsed dsp_chain
CONTRASTS = {
    "NS": ("heldout", "offline", lambda c: _has(c, "ns"), lambda c: not _has(c, "ns"), "primary"),
    "NS_afftdn_vs_anlmdn": ("heldout", "offline", lambda c: _has(c, "ns", "afftdn"), lambda c: _has(c, "ns", "anlmdn"), "secondary"),
    "VAD": ("heldout", "offline", lambda c: _has(c, "vad"), lambda c: not _has(c, "vad"), "primary"),
    "VAD_zero": ("heldout", "offline", lambda c: _has(c, "vad", "zero"), lambda c: not _has(c, "vad"), "report"),
    "VAD_cng": ("heldout", "offline", lambda c: _has(c, "vad", "cng"), lambda c: not _has(c, "vad"), "report"),
    "PLC_online": ("heldout", "online", lambda c: _has(c, "plc"), lambda c: not _has(c, "plc"), "primary"),
    "PLC_offline": ("heldout", "offline", lambda c: _has(c, "plc"), lambda c: not _has(c, "plc"), "report"),
}


# ------------------------------------------------------------------------------------------- loading
_META = {}


def meta(name):
    """{path: row} of data/dev_noisy_sim/<name>_v1/meta.csv with the parsed dsp_chain."""
    if name not in _META:
        rows = {}
        with open(SIM / f"{name}_v1" / "meta.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["chain"] = json.loads(r["dsp_chain"] or "[]")
                rows[r["path"]] = r
        _META[name] = rows
    return _META[name]


class Readout:
    """One local_eval readout dir: P(spoof) and logits per set, metrics.json, probe_report.json."""

    def __init__(self, path, label=None):
        self.path = Path(path)
        if not (self.path / "scores").is_dir():
            raise FileNotFoundError(f"no scores/ in {self.path}")
        self.label = label or f"{self.path.parents[1].name.split('_')[0]}:{self.path.name}"
        self._cache = {}

    def scores(self, name):
        if name not in self._cache:
            out = {}
            for kind in ("txt", "logits"):
                with open(self.path / "scores" / f"sim_{name}_v1.{kind}", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            out.setdefault(parts[0], [None, None])[0 if kind == "txt" else 1] = float(parts[1])
            self._cache[name] = out
        return self._cache[name]

    def metrics(self):
        p = self.path / "metrics.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def probes(self):
        p = self.path / "probe_report.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    @property
    def run_dir(self):
        return self.path.parents[1]


def readout_kind(path):
    return "merge" if Path(path).name.startswith("merge") else "final"


def final_readout(run_dir):
    eps = sorted((int(m.group(1)), p) for p in (Path(run_dir) / "local_eval").glob("epoch_*")
                 if (m := re.fullmatch(r"epoch_(\d+)", p.name)) and (p / "scores").is_dir())
    return eps[-1][1] if eps else None


# ------------------------------------------------------------------------------------------- slices
def slice_ids(contrast, halves="AB"):
    """(treated ids, control ids) over bona clips of the contrast's set and source (half B only if halves == 'B')."""
    name, src, t_pred, c_pred, _ = CONTRASTS[contrast]
    t, c = [], []
    for path, r in meta(name).items():
        if r["label"] != "bonafide" or r["src"] != src or (halves == "B" and r["half"] != "B"):
            continue
        if t_pred(r["chain"]):
            t.append(path)
        elif c_pred(r["chain"]):
            c.append(path)
    return sorted(t), sorted(c)


def _arr(ro, name, ids, col):
    s = ro.scores(name)
    return np.array([s[i][col] for i in ids], dtype=np.float64)


def logits(ro, name, ids):
    return np.clip(_arr(ro, name, ids, 1), -CLIP, CLIP)


def bona_ok(ro, name, ids):
    return _arr(ro, name, ids, 0) < THR


def gap(ro, contrast, halves="AB"):
    """Reference-side numbers: recall T / C, logit gap G = mean_T - mean_C (clipped), deficit D = (rec_C - rec_T) n_T."""
    name = CONTRASTS[contrast][0]
    t, c = slice_ids(contrast, halves)
    rt, rc = bona_ok(ro, name, t).mean(), bona_ok(ro, name, c).mean()
    return {"n_T": len(t), "n_C": len(c), "recall_T": 100 * rt, "recall_C": 100 * rc,
            "G": float(logits(ro, name, t).mean() - logits(ro, name, c).mean()), "D": float((rc - rt) * len(t))}


def did(arm, ref, contrast, halves="AB", boot=False, seed=0):
    name = CONTRASTS[contrast][0]
    t, c = slice_ids(contrast, halves)
    dt = logits(arm, name, t) - logits(ref, name, t)
    dc = logits(arm, name, c) - logits(ref, name, c)
    out = {"did": float(dt.mean() - dc.mean()), "n_T": len(t), "n_C": len(c)}
    at, ac, rt, rc = bona_ok(arm, name, t), bona_ok(arm, name, c), bona_ok(ref, name, t), bona_ok(ref, name, c)
    out["clip_did"] = float(((at.mean() - rt.mean()) - (ac.mean() - rc.mean())) * len(t))
    out["fixed"] = int(np.sum(at & ~rt))
    out["broken"] = int(np.sum(~at & rt))
    k, n = out["fixed"], out["fixed"] + out["broken"]
    out["sign_p"] = float(min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)) if n else 1.0
    out["recall_T"], out["recall_C"] = 100 * float(at.mean()), 100 * float(ac.mean())
    if boot:
        rng = np.random.default_rng(seed)
        bt = dt[rng.integers(0, len(dt), (N_BOOT, len(dt)))].mean(axis=1)
        bc = dc[rng.integers(0, len(dc), (N_BOOT, len(dc)))].mean(axis=1)
        lo, hi = np.percentile(bt - bc, [2.5, 97.5])
        out["ci"] = [float(lo), float(hi)]
    return out


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mu": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


# ------------------------------------------------------------------------------------------- tertiles
_TERT = {}


def source_tertile(source_utt):
    """'low' | 'mid' | 'high' by k = frames > 30 dB below the clip's 95th-percentile frame energy (pinned)."""
    if source_utt not in _TERT:
        import soundfile as sf
        x, _ = sf.read(str(DEV_WAV / source_utt), dtype="float64", always_2d=True)
        x = x[:, 0]
        if len(x) < N_WIN:                              # utils/data_utils.pad_audio, random_start=False: tile
            x = np.tile(x, int(N_WIN // max(len(x), 1)) + 1)
        x = x[:N_WIN]
        frames = x[: N_FRAMES * FRAME].reshape(N_FRAMES, FRAME)
        e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
        k = int(np.sum(e < np.percentile(e, 95) - 30))
        _TERT[source_utt] = "low" if k <= LOW_MAX else ("mid" if k <= MID_MAX else "high")
    return _TERT[source_utt]


def tertile_counts(ro, halves="AB"):
    """{tertile: {"n", "fa", "sat"}} over the bona clips of matched + heldout."""
    out = {t: {"n": 0, "fa": 0, "sat": 0} for t in ("low", "mid", "high")}
    for name in ("matched", "heldout"):
        s = ro.scores(name)
        for path, r in meta(name).items():
            if r["label"] != "bonafide" or (halves == "B" and r["half"] != "B"):
                continue
            d = out[source_tertile(r["source_utt"])]
            p = s[path][0]
            d["n"] += 1
            d["fa"] += int(p >= THR)
            d["sat"] += int(p >= SAT)
    return out


def crit2(ref, arm, halves):
    a, b = tertile_counts(ref, halves), tertile_counts(arm, halves)
    return (a["high"]["sat"] - b["high"]["sat"]) - (a["low"]["sat"] - b["low"]["sat"])


# ------------------------------------------------------------------------------------------- guards
def spoof_misses(ro):
    """Spoof predicted bona (p < 0.5) in the plan section 2.0 guard slices."""
    out = {"NS_offline": 0, "VAD_offline": 0, "PLC": 0, "halfB_noisy": 0}
    for name in ("matched", "heldout"):
        s = ro.scores(name)
        for path, r in meta(name).items():
            if r["label"] != "spoof":
                continue
            miss = int(s[path][0] < THR)
            if r["half"] == "B":
                out["halfB_noisy"] += miss
            if name == "heldout":
                c = r["chain"]
                if r["src"] == "offline" and _has(c, "ns"):
                    out["NS_offline"] += miss
                if r["src"] == "offline" and _has(c, "vad"):
                    out["VAD_offline"] += miss
                if _has(c, "plc"):
                    out["PLC"] += miss
    return out


def poisson_upper(k, q=0.99):
    from scipy.stats import poisson
    return int(poisson.ppf(q, k))


def loss_bins(arm, ref):
    rows = {}
    for path, r in meta("heldout").items():
        if r["label"] != "bonafide":
            continue
        plc = [st for st in r["chain"] if "plc" in st]
        loss = plc[0]["loss"] if plc else None
        b = "none" if loss is None else ("<0.05" if loss < 0.05 else ("0.05-0.10" if loss < 0.10 else ">=0.10"))
        rows.setdefault((r["src"], b), []).append(path)
    out = {}
    for (src, b), ids in sorted(rows.items()):
        out[f"{src}:{b}"] = {"n": len(ids), "arm_logit": float(logits(arm, "heldout", ids).mean()),
                             "ref_logit": float(logits(ref, "heldout", ids).mean()),
                             "arm_recall": 100 * float(bona_ok(arm, "heldout", ids).mean()),
                             "ref_recall": 100 * float(bona_ok(ref, "heldout", ids).mean())}
    return out


def exclusive_stages(arm, ref):
    """Offline heldout bona with exactly one DSP stage, by stage type: n and false alarms (arm / ref)."""
    out = {}
    for path, r in meta("heldout").items():
        if r["label"] != "bonafide" or r["src"] != "offline" or len(r["chain"]) != 1:
            continue
        st = r["chain"][0]
        key = next(k for k in ("ns", "agc", "vad", "plc", "bw") if k in st)
        key = f"{key}:{st[key]}" if key != "bw" else "bw"
        d = out.setdefault(key, {"n": 0, "fa_arm": 0, "fa_ref": 0})
        d["n"] += 1
        d["fa_arm"] += int(arm.scores("heldout")[path][0] >= THR)
        d["fa_ref"] += int(ref.scores("heldout")[path][0] >= THR)
    return dict(sorted(out.items()))


def heldout_f1(ro):
    return 100 * ro.metrics()["half_B"]["sets"]["sim_heldout_v1"]["f1"]


def half_b(ro, key):
    m = ro.metrics().get("half_B", {})
    if key == "wf1":
        return 100 * m["wf1"]["wf1"]
    return 100 * m["sets"][key]["f1"]


def train_log_rows(run_dir):
    rows = {}
    p = Path(run_dir) / "logs" / "train.log"
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"Epoch (\d+)/(\d+) TrainLoss=([\d.]+) TrainAcc=([\d.]+)%", line)
            if m:
                rows[int(m.group(1))] = {"n": int(m.group(2)), "loss": float(m.group(3)), "acc": float(m.group(4))}
    return rows


def probe_rows(ro):
    out = {}
    for r in ro.probes().get("rows", []):
        if r.get("label") == "bonafide" and r.get("probe") in ("sil_only", "noise_only", "padded", "trimmed"):
            out[r["probe"]] = {"n": r["n"], "mean_dP": r["mean_dP"], "flip_rate": 100 * r["flip_rate"]}
    return out


# ------------------------------------------------------------------------------------------- analysis
def excluded_run(ref_f, final_dir=None):
    """B2a (failed run): printed and flagged, never part of the panel statistics."""
    p = Path(final_dir) if final_dir else EXP / EXCLUDED[1] / "local_eval" / EXCLUDED[2]
    if not (p / "scores").is_dir():
        return None
    ro = Readout(p, EXCLUDED[0])
    return {"label": EXCLUDED[0], "dids": {c: did(ro, ref_f, c)["did"] for c in CONTRASTS},
            "crit2_full_final": crit2(ref_f, ro, "AB")}


def panel_tables(ref_pair, panel_final, panel_merge):
    """Null-panel numbers per contrast: DiD stats (final), half-B means per readout, clip-DiD mean; criterion (ii)."""
    ref_m, ref_f = ref_pair
    out = {"contrasts": {}, "crit2": {}, "runs": [r.label for r in panel_final]}
    for c in CONTRASTS:
        dids = [did(r, ref_f, c)["did"] for r in panel_final]
        clips = [did(r, ref_f, c)["clip_did"] for r in panel_final]
        hb_f = [did(r, ref_f, c, "B")["did"] for r in panel_final]
        hb_m = [did(r, ref_m, c, "B")["did"] for r in panel_merge] if panel_merge else []
        g = gap(ref_f, c)
        st = stats(dids)
        bar = min(st["mu"] - 2 * st["sd"], st["min"], st["mu"] - 0.4 * g["G"])
        out["contrasts"][c] = {**st, "per_run": dids, "clip_mu": float(np.mean(clips)), "halfB_mu_final": float(np.mean(hb_f)),
                               "halfB_mu_merge": float(np.mean(hb_m)) if hb_m else None, "bar": bar,
                               "typical_gap": g["G"] + st["mu"], "z_close_typical": -(g["G"] + st["mu"]) / st["sd"]}
    for key, ref, runs, halves in (("full_final", ref_f, panel_final, "AB"), ("halfB_final", ref_f, panel_final, "B"),
                                   ("halfB_merge", ref_m, panel_merge, "B")):
        if runs:
            vals = [crit2(ref, r, halves) for r in runs]
            out["crit2"][key] = {**stats(vals), "per_run": vals}
    return out


def contrast_verdict(c, arm_d, panel, halfb_star, placebo_dids, summary):
    """shown / directional / inconclusive / refuted (+ the condition flags) for one contrast on the verdict readout."""
    p = panel["contrasts"][c]
    star = arm_d["did"] - p["mu"]
    z = star / p["sd"]
    cond = {"a": arm_d["did"] <= p["bar"],
            "b": arm_d.get("ci", [0, math.inf])[1] < p["mu"],
            "c": arm_d["clip_did"] - p["clip_mu"] >= 0.4 * arm_d["D_ref_final"],
            "d": len(halfb_star) == 2 and all(v is not None and v < 0 for v in halfb_star)}
    n_pl = len(placebo_dids)
    cond["e"] = (all(arm_d["did"] < v for v in placebo_dids) if n_pl else None)
    if star >= 0:
        v = "refuted"
    elif n_pl == 0:                                                     # O3.3: no placebo arm -> capped
        v = "directional"
    elif not cond["e"]:
        v = "inconclusive"
    elif all(cond[k] for k in "abcd"):
        v = "shown"
    else:
        v = "directional"
    return {"verdict": v, "provisional": not summary, "n_placebo": n_pl, "did_star": star, "z": z, "conditions": cond}


def analyse(args):
    run = Readout(args.run)
    kind = readout_kind(args.run)
    ref_m = Readout(args.ref_pair[0], "B2f:merge_top5")
    ref_f = Readout(args.ref_pair[1], "B2f:final")
    ref = Readout(args.ref, "B2f:" + kind) if args.ref else (ref_m if kind == "merge" else ref_f)
    if args.run_pair:
        arm_m, arm_f = Readout(args.run_pair[0]), Readout(args.run_pair[1])
    else:
        mp, fp = run.run_dir / "local_eval" / "merge_top5", final_readout(run.run_dir)
        arm_m = Readout(mp) if (mp / "scores").is_dir() else None
        arm_f = Readout(fp) if fp is not None else None
    arm_by_kind = {"merge": arm_m, "final": arm_f}
    arm_by_kind[kind] = run
    ref_by_kind = {"merge": ref_m, "final": ref_f}
    panel_final = [Readout(p, Path(p).parents[1].name.split("_")[0]) for p in args.null_panel]
    panel_merge = [Readout(p, Path(p).parents[1].name.split("_")[0]) for p in args.null_panel_merge]
    placebos = [Readout(p) for p in args.placebo]
    lever = args.lever
    if lever is None:
        cfg = run.run_dir / "config.yaml"
        levs = []
        if cfg.is_file():
            import yaml
            levs = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("b4_levers") or []
        lever = levs[0] if len(levs) == 1 else "none"

    panel = panel_tables((ref_m, ref_f), panel_final, panel_merge)
    if kind == "merge":                     # re-centre a merge-readout verdict on the merge panel
        panel_v = panel_tables((ref_m, ref_m), panel_merge, panel_merge) if panel_merge else panel
    else:
        panel_v = panel
    res = {"run": str(run.path), "readout": kind, "lever": lever, "summary_mode": bool(args.summary),
           "panel_runs": panel["runs"], "excluded": excluded_run(ref_f), "contrasts": {}, "panel": panel}

    for c in CONTRASTS:
        g_f, g_m = gap(ref_f, c), gap(ref_m, c)
        d = did(run, ref, c, boot=True)
        d["D_ref_final"] = g_f["D"] if kind == "final" else g_m["D"]
        hb = {}
        for k in ("merge", "final"):
            a = arm_by_kind[k]
            if a is not None:
                hd = did(a, ref_by_kind[k], c, "B")["did"]
                mu = panel["contrasts"][c]["halfB_mu_final" if k == "final" else "halfB_mu_merge"]
                hb[k] = {"did": hd, "did_star": hd - mu if mu is not None else None}
        pl = [did(p, ref, c)["did"] for p in placebos]
        v = contrast_verdict(c, d, panel_v, [hb.get(k, {}).get("did_star") for k in ("merge", "final") if k in hb]
                             if len(hb) == 2 else [], pl, args.summary)
        res["contrasts"][c] = {"ref_final": g_f, "ref_merge": g_m, "within_run": did(ref_m, ref_f, c)["did"],
                               "arm": d, "halfB": hb, "placebo": dict(zip([p.label for p in placebos], pl)), **v}

    # pinned tertiles + criterion (ii)
    tert = {}
    for k, ro in (("ref_merge", ref_m), ("ref_final", ref_f), ("arm_merge", arm_m), ("arm_final", arm_f)):
        if ro is not None:
            tert[k] = {"full": tertile_counts(ro), "halfB": tertile_counts(ro, "B")}
    sources = {"low": 0, "mid": 0, "high": 0}
    for path, r in meta("heldout").items():
        if r["label"] == "bonafide":
            sources[source_tertile(r["source_utt"])] += 1
    c2 = {}
    if arm_f is not None:
        c2["full_final"] = crit2(ref_f, arm_f, "AB")
        c2["halfB_final"] = crit2(ref_f, arm_f, "B")
    if arm_m is not None:
        c2["halfB_merge"] = crit2(ref_m, arm_m, "B")
    res["tertiles"] = {"sources": sources, "counts": tert, "crit2_arm": c2}

    # guards, outcome, anomaly
    guards = {}
    for k in ("merge", "final"):
        if arm_by_kind[k] is not None:
            base = spoof_misses(ref_by_kind[k])
            guards[k] = {"arm": spoof_misses(arm_by_kind[k]), "ref": base, "bound": {g: poisson_upper(v) for g, v in base.items()}}
    res["spoof_guards"] = guards
    outcome = {"panel": {}, "arm": {}}
    for k, runs, refr in (("final", panel_final, ref_f), ("merge", panel_merge, ref_m)):
        if runs:
            vals = [heldout_f1(r) for r in runs]
            outcome["panel"][k] = {"ref": heldout_f1(refr), "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                                   "delta": [v - heldout_f1(refr) for v in vals],
                                   "clearing": {str(t): int(sum(v - heldout_f1(refr) >= t for v in vals)) for t in HELDOUT_THRESHOLDS}}
        a = arm_by_kind[k]
        if a is not None:
            outcome["arm"][k] = {"heldout_f1": heldout_f1(a), "delta": heldout_f1(a) - heldout_f1(refr),
                                 "wf1_delta": half_b(a, "wf1") - half_b(refr, "wf1"),
                                 "clean_delta": half_b(a, "dev_online_clean") - half_b(refr, "dev_online_clean"),
                                 "matched_delta": half_b(a, "sim_matched_v1") - half_b(refr, "sim_matched_v1")}
    escape = len(outcome["arm"]) == 2 and all(outcome["arm"][k]["delta"] >= 0.5 for k in ("merge", "final"))
    outcome["escape_falsifier1"] = escape
    res["outcome"] = outcome

    b2f_rows = train_log_rows(B2F)
    arm_rows = train_log_rows(run.run_dir)
    acc_thr = b2f_rows[1]["acc"] - UNDERFIT_ACC_DROP if 1 in b2f_rows else None
    loss_thr = b2f_rows[max(b2f_rows)]["loss"] * UNDERFIT_LOSS_MULT if b2f_rows else None
    underfit, recall_down = False, False
    if arm_rows and acc_thr is not None:
        fit_bad = arm_rows.get(1, {}).get("acc", 100.0) < acc_thr or arm_rows[max(arm_rows)]["loss"] > loss_thr
        if arm_f is not None:
            recall_down = (arm_f.metrics()["half_B"]["sets"]["sim_heldout_v1"]["recall_bona"]
                           < ref_f.metrics()["half_B"]["sets"]["sim_heldout_v1"]["recall_bona"])
        underfit = fit_bad and recall_down
    clean_bad = any(o["clean_delta"] < -CLEAN_DROP for o in outcome["arm"].values())
    target = LEVER_TARGET.get(lever)
    z_bad = {c: res["contrasts"][c]["z"] for c in PRIMARY if c != target and abs(res["contrasts"][c]["z"]) > 3}
    anomalous = bool(z_bad) or underfit or clean_bad
    res["anomaly"] = {"anomalous": anomalous, "z_gt3_nontarget": z_bad, "underfit": underfit, "clean_guard": clean_bad,
                      "underfit_thresholds": {"ep1_acc_lt": acc_thr, "last_loss_gt": loss_thr}, "heldout_recall_down": recall_down}
    res["lever_verdict"] = lever_verdict(lever, res, panel)
    if lever == "dtx_gate":                  # O2.3 report: the VAD contrast split by fill (not a verdict rule)
        res["vad_fill_reading"] = vad_fill_reading(res["contrasts"], panel_v)
    if lever == "opus_plc":                  # plan 2.3 decision lines (falsifiers, guards, attribution control)
        ctrl = [Readout(x) for x in args.attribution_control] if getattr(args, "attribution_control", None) else None
        res["opus_plc_reading"] = opus_plc_reading(res, arm_by_kind, ref_by_kind, run.run_dir, ctrl)
    if args.halves == "B":                  # extra half-B view (the verdict stays on the full set)
        res["halfB_view"] = {c: {"ref": gap(ref, c, "B"), "arm": did(run, ref, c, "B")} for c in CONTRASTS}
    res["loss_bins"] = loss_bins(run, ref)
    res["exclusive"] = exclusive_stages(run, ref)
    res["probes"] = {"arm": probe_rows(run), "ref": probe_rows(ref)}
    return res


def vad_fill_reading(contrasts, panel):
    """
    dtx_gate report (plan 1.1 "Heldout VAD split by fill", 2.2, O2.3): a VAD gain confined to vad:zero (zero-fill
    DiD* <= -sd_zero, CNG-fill DiD* >= 0) is near-in-distribution (dtx_gate's zero fill is the heldout zero fill);
    transfer to the unseen CNG colour is claimable only with CNG-fill DiD* < 0. DiD* on the verdict readout's panel.
    """
    z, c = contrasts["VAD_zero"]["did_star"], contrasts["VAD_cng"]["did_star"]
    sd0 = panel["contrasts"]["VAD_zero"]["sd"]
    if c < 0:
        reading = "transfer claimable (CNG-fill DiD* < 0)"
    elif z <= -sd0:
        reading = "near-in-distribution (gain confined to vad:zero)"
    else:
        reading = "no fill-specific gain"
    return {"zero_did": contrasts["VAD_zero"]["arm"]["did"], "zero_did_star": z, "zero_bar": -sd0,
            "cng_did": contrasts["VAD_cng"]["arm"]["did"], "cng_did_star": c, "reading": reading}


def matched_app_recall(ro, app):
    """Matched bona recall (%) of one Opus app profile (metrics.json breakdowns, full set); None if absent."""
    v = ro.metrics().get("breakdowns", {}).get("sim_matched_v1", {}).get("codec", {}).get(app)
    return None if v is None else 100 * v["recall_bona"]


def attribution(arm_gain, ctrl_gain, share=ATTRIBUTION_SHARE):
    """Falsifier (3) of opus_plc: the zero-loss control's WF1 gain vs the arm's, same readout, both vs B2f."""
    if arm_gain is None or ctrl_gain is None:
        return "n/a"
    if arm_gain <= 0:
        return "n/a (arm WF1 gain <= 0)"
    return "credited to the decoder path" if ctrl_gain >= share * arm_gain else "credited to PLC"


def opus_plc_reading(res, arm_by_kind, ref_by_kind, run_dir, control=None):
    """
    opus_plc decision lines (plan 2.3; O3.2): falsifier (1) = online PLC DiD* >= 0 (verdict readout) AND no outcome
    escape (heldout F1 delta >= +0.5 on merge_top5 AND final); falsifier (2) = WF1 delta < -0.5 on a readout with the
    shared under-fit signature; the must-not-move guards per readout; falsifier (3), the attribution control, when its
    readouts (merge, final) are given. A report: the lever verdict stays capped at 'directional'.
    """
    c, o = res["contrasts"]["PLC_online"], res["outcome"]
    f1 = bool(c["did_star"] >= 0 and not o["escape_falsifier1"])
    wf1_low = {k: v["wf1_delta"] for k, v in o["arm"].items() if v["wf1_delta"] < WF1_FALSIFIER}
    f2 = bool(wf1_low) and bool(res["anomaly"]["underfit"])
    guards = {}
    for k in ("merge", "final"):
        a, r = arm_by_kind[k], ref_by_kind[k]
        if a is None or k not in o["arm"]:
            continue
        apps = {}
        for app in OPUS_APP_GUARD:
            va, vr = matched_app_recall(a, app), matched_app_recall(r, app)
            d = None if va is None or vr is None else va - vr
            apps[app] = {"arm": va, "ref": vr, "delta": d, "ok": d is not None and d >= -OPUS_APP_DROP}
        sg = res["spoof_guards"].get(k, {})
        plc_m, plc_b = sg.get("arm", {}).get("PLC"), sg.get("bound", {}).get("PLC")
        ox = o["arm"][k]
        guards[k] = {"matched_f1_delta": ox["matched_delta"], "matched_ok": ox["matched_delta"] >= -MATCHED_DROP,
                     "clean_delta": ox["clean_delta"], "clean_ok": abs(ox["clean_delta"]) <= CLEAN_DROP,
                     "app_recall": apps, "plc_spoof_misses": plc_m, "plc_bound": plc_b,
                     "plc_ok": plc_m is not None and plc_b is not None and plc_m <= plc_b}
    b2f_rows, arm_rows = train_log_rows(B2F), train_log_rows(run_dir)
    ep1 = (arm_rows[1]["acc"] - b2f_rows[1]["acc"]) if 1 in arm_rows and 1 in b2f_rows else None
    held = all(g["matched_ok"] and g["clean_ok"] and g["plc_ok"] and all(x["ok"] for x in g["app_recall"].values())
               for g in guards.values()) and ep1 is not None and abs(ep1) <= EP1_ACC_BAND
    ctrl = None
    if control:
        ctrl = {}
        for k, ro in zip(("merge", "final"), control):
            ref = ref_by_kind[k]
            cg = half_b(ro, "wf1") - half_b(ref, "wf1")
            ag = o["arm"].get(k, {}).get("wf1_delta")
            ctrl[k] = {"control": str(ro.path), "control_wf1_delta": cg, "arm_wf1_delta": ag, "reading": attribution(ag, cg)}
    if f1:
        decision = "DROP: falsifier (1) fires (libopus PLC does not carry over)"
    elif f2:
        decision = "DROP: falsifier (2) fires (codec-side under-fit)"
    else:
        decision = ("no falsifier fires; decide on the wave's WF1 CI rule + the guards; if positive on WF1, run the "
                    "attribution control '--aug_opus_plc_p 0.3 --aug_opus_loss 0 0' before promotion")
    return {"verdict_readout": res["readout"], "plc_did": c["arm"]["did"], "plc_did_star": c["did_star"],
            "escape_readouts": "merge_top5 AND final", "escape": bool(o["escape_falsifier1"]),
            "falsifier1": f1, "falsifier2": f2, "wf1_below": wf1_low, "underfit": bool(res["anomaly"]["underfit"]),
            "guards": guards, "ep1_acc_delta": ep1, "guards_hold": bool(held), "attribution_control": ctrl,
            "decision": decision}


def lever_verdict(lever, res, panel):
    if lever not in LEVER_TARGET:
        return {"verdict": "n/a (no B4 lever named)", "n_placebo": None}
    c = res["contrasts"]
    anomalous = res["anomaly"]["anomalous"]
    if lever == "apm_ns":
        cv = c["NS"]
        v = "mechanism shown" if cv["verdict"] == "shown" else ("refuted" if cv["verdict"] == "refuted" else cv["verdict"])
    elif lever == "dtx_gate":
        c2 = res["tertiles"]["crit2_arm"]
        ii = (c2.get("full_final", -99) >= 5 and c2.get("halfB_final", -99) >= 0 and c2.get("halfB_merge", -99) >= 0)
        vad = c["VAD"]["verdict"]
        mu2 = panel["crit2"]["full_final"]["mu"]
        if vad == "refuted" and c2.get("full_final", 99) <= mu2:
            v = "refuted"
        elif ii and vad in ("shown", "directional"):
            v = "mechanism shown (strong)" if vad == "shown" else "mechanism shown"
        else:
            v = f"not shown (criterion ii {'holds' if ii else 'fails'}, VAD {vad})"
    else:
        cv = c["PLC_online"]
        v = "refuted" if cv["verdict"] == "refuted" else ("directional" if cv["verdict"] in ("shown", "directional") else cv["verdict"])
    if anomalous and v.startswith("mechanism shown"):
        v = "confounded (run anomaly)"
    target = LEVER_TARGET[lever]
    return {"verdict": v, "n_placebo": c[target]["n_placebo"], "provisional": c[target]["provisional"],
            "lower_p_retry_allowed": lever == "apm_ns" and v == "mechanism shown"}


# ------------------------------------------------------------------------------------------- printing
def fmt(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def print_report(res):
    P = res["panel"]
    print(f"B4 mechanism readout: {res['run']}  (readout {res['readout']}, lever {res['lever']}"
          f"{', SUMMARY' if res['summary_mode'] else ''})")
    ex = res["excluded"]
    print(f"null panel ({len(P['runs'])}): {', '.join(P['runs'])}; {EXCLUDED[0]} printed only, flagged 'failed run, excluded'"
          + ("" if ex else " (its readout is missing)"))
    print("\n## Contrasts (full set; arm vs B2f same readout; DiD on clipped logits, CI 2,000x bootstrap)")
    for c, r in res["contrasts"].items():
        p, a, rf, rm = P["contrasts"][c], r["arm"], r["ref_final"], r["ref_merge"]
        print(f"\n[{c}] ({CONTRASTS[c][4]}) n_T/n_C {rf['n_T']}/{rf['n_C']} | B2f final recall T/C {rf['recall_T']:.1f}/{rf['recall_C']:.1f}"
              f" G {fmt(rf['G'])} D merge/final {rm['D']:.1f}/{rf['D']:.1f} | within-run (B2f merge-final) {fmt(r['within_run'])}")
        print(f"  panel DiD mu/sd/min/max {fmt(p['mu'])}/{fmt(p['sd'], sign=False)}/{fmt(p['min'])}/{fmt(p['max'])}"
              f"  half-B mu final/merge {fmt(p['halfB_mu_final'])}/{fmt(p['halfB_mu_merge'])}  clip mu {fmt(p['clip_mu'], 1)}"
              f"  typical gap {fmt(p['typical_gap'])}  bar {fmt(p['bar'])}")
        print(f"  panel per-run DiD: " + ", ".join(f"{lb} {fmt(v)}" for lb, v in zip(P["runs"], p["per_run"]))
              + (f" | {ex['label']} {fmt(ex['dids'][c])} (failed run, excluded)" if ex else ""))
        print(f"  arm recall T/C {a['recall_T']:.1f}/{a['recall_C']:.1f}  DiD {fmt(a['did'])} CI [{fmt(a['ci'][0])}, {fmt(a['ci'][1])}]"
              f"  DiD* {fmt(r['did_star'])}  z {fmt(r['z'])}  clip DiD {fmt(a['clip_did'], 1)} (clip DiD* {fmt(a['clip_did'] - p['clip_mu'], 1)})"
              f"  fixed/broken {a['fixed']}/{a['broken']} p={a['sign_p']:.3g}")
        hb = "  ".join(f"{k} DiD {fmt(v['did'])} DiD* {fmt(v['did_star'])}" for k, v in r["halfB"].items())
        pl = ", ".join(f"{k} {fmt(v)}" for k, v in r["placebo"].items()) or "none"
        rank = 1 + sum(v < a["did"] for v in r["placebo"].values())
        cond = " ".join(f"({k}){'Y' if v else ('-' if v is None else 'n')}" for k, v in r["conditions"].items())
        print(f"  half-B: {hb or 'n/a'} | placebo DiDs: {pl} (arm rank {rank}/{len(r['placebo']) + 1})")
        print(f"  verdict: {r['verdict'].upper()}{' (provisional)' if r['provisional'] else ''}  [placebos {r['n_placebo']}]  {cond}")
    t = res["tertiles"]
    print(f"\n## Pinned tertiles (k of {N_FRAMES} frames: low <= {LOW_MAX}, mid {LOW_MAX + 1}-{MID_MAX}, high >= {MID_MAX + 1};"
          f" sources {t['sources']['low']}/{t['sources']['mid']}/{t['sources']['high']}; each source counted twice)")
    for k, v in t["counts"].items():
        f, h = v["full"], v["halfB"]
        print(f"  {k:10s} full n {'/'.join(str(f[x]['n']) for x in f)}  FA {'/'.join(str(f[x]['fa']) for x in f)}"
              f"  sat {'/'.join(str(f[x]['sat']) for x in f)} | half-B n {'/'.join(str(h[x]['n']) for x in h)}"
              f"  sat {'/'.join(str(h[x]['sat']) for x in h)}")
    for k, v in t["crit2_arm"].items():
        p = P["crit2"].get(k)
        z = (v - p["mu"]) / p["sd"] if p else float("nan")
        need = 5 if k == "full_final" else 0
        print(f"  criterion (ii) drop_high - drop_low {k}: {v:+d} (need >= {need:+d}); panel mu/sd/max "
              f"{fmt(p['mu'], 1)}/{fmt(p['sd'], 1, False)}/{p['max']:+.0f}, z {fmt(z)}; per run "
              + ", ".join(f"{lb} {x:+d}" for lb, x in zip(P["runs"], p["per_run"]))
              + (f" | {ex['label']} {ex['crit2_full_final']:+d} (excluded)" if ex and k == "full_final" else "")
              if p else f"  {k}: {v:+d}")
    print("\n## Outcome (half B, heldout F1; O3.2: panel SD and panel runs clearing each threshold)")
    o = res["outcome"]
    for k, p in o["panel"].items():
        clear = ", ".join(f">= +{t}: {n}/{len(p['delta'])}" for t, n in p["clearing"].items())
        a = o["arm"].get(k)
        arm = (f"arm {a['heldout_f1']:.2f} (delta {fmt(a['delta'])}; WF1 {fmt(a['wf1_delta'])}, clean {fmt(a['clean_delta'])},"
               f" matched {fmt(a['matched_delta'])})" if a else "arm n/a")
        print(f"  {k:5s} B2f {p['ref']:.2f}; panel mean {p['mean']:.2f} sd {p['sd']:.2f}; clearing {clear} | {arm}")
    print(f"  falsifier (1) outcome escape (heldout F1 delta >= +0.5 on merge_top5 AND final): {o['escape_falsifier1']}."
          " 'Positive outcome' for the section 4 recommendation = the wave's WF1 CI rule, not heldout F1.")
    g = res["spoof_guards"]
    print("\n## Spoof-miss guards (Poisson 99 % upper bound of the B2f count)")
    for k, v in g.items():
        print(f"  {k:5s} " + "  ".join(f"{n} {v['arm'][n]} (B2f {v['ref'][n]}, <= {v['bound'][n]}"
                                      f"{', BROKEN' if v['arm'][n] > v['bound'][n] else ''})" for n in v["arm"]))
    an = res["anomaly"]
    print(f"\n## Run anomaly (O3.1): {'ANOMALOUS' if an['anomalous'] else 'none'} | |z|>3 on non-target primary "
          f"{ {k: round(v, 2) for k, v in an['z_gt3_nontarget'].items()} } | under-fit {an['underfit']} "
          f"(ep1 TrainAcc < {fmt(an['underfit_thresholds']['ep1_acc_lt'], 2, False)} or last TrainLoss > "
          f"{fmt(an['underfit_thresholds']['last_loss_gt'], 4, False)}, with heldout bona recall down {an['heldout_recall_down']})"
          f" | clean-F1 guard {an['clean_guard']}")
    lv = res["lever_verdict"]
    print(f"\n## Lever verdict ({res['lever']}): {lv['verdict'].upper()}"
          f"{' (provisional)' if lv.get('provisional') else ''}  [placebos {lv['n_placebo']}]"
          + (f"  lower-p retry allowed: {lv['lower_p_retry_allowed']}" if res["lever"] == "apm_ns" else ""))
    if "vad_fill_reading" in res:
        f = res["vad_fill_reading"]
        print(f"## Report (O2.3, not a verdict rule): VAD by fill -- vad:zero DiD {fmt(f['zero_did'])} DiD* {fmt(f['zero_did_star'])}"
              f" (near-in-distribution bar <= {fmt(f['zero_bar'])}); vad:cng DiD {fmt(f['cng_did'])} DiD* {fmt(f['cng_did_star'])}"
              f" (transfer needs < 0) -> {f['reading']}")
    if "opus_plc_reading" in res:
        r = res["opus_plc_reading"]
        print(f"## opus_plc decision lines (plan 2.3, O3.2; report, the verdict stays capped at 'directional'): online PLC "
              f"DiD {fmt(r['plc_did'])} DiD* {fmt(r['plc_did_star'])} ({r['verdict_readout']} readout)")
        print(f"  falsifier (1) [PLC DiD* >= 0 AND NOT heldout F1 delta >= +0.5 on {r['escape_readouts']}]: "
              f"{'FIRES' if r['falsifier1'] else 'does not fire'} (escape {r['escape']})")
        print(f"  falsifier (2) [WF1 delta < {WF1_FALSIFIER:+.1f} with the shared under-fit signature]: "
              f"{'FIRES' if r['falsifier2'] else 'does not fire'} (WF1 below {({k: round(v, 2) for k, v in r['wf1_below'].items()})}, "
              f"under-fit {r['underfit']})")
        for k, g in r["guards"].items():
            apps = ", ".join(f"{a.split(':')[1]} {fmt(x['arm'], 2, False)} (B2f {fmt(x['ref'], 2, False)}, delta {fmt(x['delta'])}"
                             f"{'' if x['ok'] else ', BROKEN'})" for a, x in g["app_recall"].items())
            print(f"  must not move, {k}: matched F1 delta {fmt(g['matched_f1_delta'])} (>= -{MATCHED_DROP}"
                  f"{'' if g['matched_ok'] else ', BROKEN'}); matched bona recall {apps} (>= -{OPUS_APP_DROP:g}); PLC spoof misses "
                  f"{g['plc_spoof_misses']} (<= {g['plc_bound']}{'' if g['plc_ok'] else ', BROKEN'}); clean F1 delta "
                  f"{fmt(g['clean_delta'])} (within {CLEAN_DROP}{'' if g['clean_ok'] else ', BROKEN'})")
        print(f"  ep1 TrainAcc delta {fmt(r['ep1_acc_delta'])} (within {EP1_ACC_BAND:g}); guards hold: {r['guards_hold']}")
        if r["attribution_control"]:
            for k, a in r["attribution_control"].items():
                print(f"  attribution control (3), {k}: control WF1 delta {fmt(a['control_wf1_delta'])} vs arm "
                      f"{fmt(a['arm_wf1_delta'])} -> {a['reading']} (decoder path if control >= {ATTRIBUTION_SHARE:g} x arm)")
        else:
            print("  attribution control (3): not given (--attribution_control <control merge> <control final>)")
        print(f"  decision: {r['decision']}")
    if "halfB_view" in res:
        print("\n## Half-B view (B2f same readout; not used for the verdict): " + "; ".join(
            f"{c} n {v['ref']['n_T']}/{v['ref']['n_C']} B2f recall {v['ref']['recall_T']:.1f}/{v['ref']['recall_C']:.1f}"
            f" G {fmt(v['ref']['G'])} arm DiD {fmt(v['arm']['did'])}" for c, v in res["halfB_view"].items()))
    print("\n## Report: bona by realised PLC loss (heldout): " + "; ".join(
        f"{k} n={v['n']} logit {fmt(v['arm_logit'])} (B2f {fmt(v['ref_logit'])}) recall {v['arm_recall']:.1f} (B2f {v['ref_recall']:.1f})"
        for k, v in res["loss_bins"].items()))
    print("## Report: exclusive single-stage offline bona (n, FA arm/B2f): " + "; ".join(
        f"{k} {v['n']} {v['fa_arm']}/{v['fa_ref']}" for k, v in res["exclusive"].items()))
    pr = res["probes"]
    print("## Report: probes (bona mean dP, flip %): " + "; ".join(
        f"{k} {fmt(v['mean_dP'], 4)} / {v['flip_rate']:.2f}% (B2f {fmt(pr['ref'].get(k, {}).get('mean_dP'), 4)} / "
        f"{pr['ref'].get(k, {}).get('flip_rate', float('nan')):.2f}%)" for k, v in pr["arm"].items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="arm readout dir (<run>/local_eval/<epoch_N|merge_top5>)")
    ap.add_argument("--ref", default=None, help="B2f readout of the same kind (default: from --ref_pair)")
    ap.add_argument("--ref_pair", nargs=2, default=[str(B2F / "local_eval/merge_top5"), str(B2F / "local_eval/epoch_8")],
                    metavar=("MERGE", "FINAL"))
    ap.add_argument("--run_pair", nargs=2, default=None, metavar=("MERGE", "FINAL"),
                    help="the arm's two readouts (default: <run>/local_eval/merge_top5 and its highest epoch_N)")
    ap.add_argument("--null_panel", nargs="*", default=[str(EXP / d / "local_eval" / f) for _, d, f in PANEL])
    ap.add_argument("--null_panel_merge", nargs="*", default=[str(EXP / d / "local_eval/merge_top5") for _, d, _ in PANEL])
    ap.add_argument("--placebo", nargs="*", default=[], help="other B4 arms' readouts (same readout as --run)")
    ap.add_argument("--lever", default=None, choices=list(LEVER_TARGET) + ["none"],
                    help="the arm's lever (default: config.yaml b4_levers when it names exactly one)")
    ap.add_argument("--summary", action="store_true", help="wave SUMMARY: verdicts are final, not provisional")
    ap.add_argument("--attribution_control", nargs=2, default=None, metavar=("MERGE", "FINAL"),
                    help="opus_plc: the zero-loss control arm's readouts (--aug_opus_plc_p 0.3 --aug_opus_loss 0 0), "
                         "for falsifier (3); also pass them to --placebo")
    ap.add_argument("--halves", default="AB", choices=["AB", "B"], help="AB = full set primary (default); B = print the half-B view only")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args(argv)
    res = analyse(args)
    print_report(res)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
