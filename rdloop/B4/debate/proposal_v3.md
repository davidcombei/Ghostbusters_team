# rdloop B4 - proposal v3

> research-architect, REVISE v3 (attempt a0), 2026-10-17. Base **B2f**, entry module `b3wave`, cycle focus
> **augmentation**, levers 2-4. Owner notes: none (`rdloop/B4/user_notes.md` holds only the header).
> This version answers critique v2 (`debate/critique_v2.md`; ledger O2.1 major, O2.2-O2.4 minor) and its two
> alternatives. §5 has the point-by-point answers and §6 the change list.
>
> RUN = `SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`.
> A pair written "merge / final" is always `local_eval/merge_top5` first, then the final epoch (`epoch_8`;
> `epoch_10` for B2d).
>
> Provenance of the numbers:
> - **copied** from a named file: AUD = `rdloop/B4/01_audit.md`, SOTA = `rdloop/B4/02_sota.md`, CRIT2 =
>   `debate/critique_v2.md`, PV2 = `debate/proposal_v2.md`, RUN `logs/train.log`, `exploop/B3/wave.json`,
>   `SLS_setup/exp/<run>/local_eval/<readout>/metrics.json`;
> - **(computed)**: read-only in-memory `python3 -I` joins, run in this session, of
>   `SLS_setup/exp/<run>/local_eval/<readout>/scores/sim_{heldout,matched}_v1.{txt,logits}` with
>   `data/dev_noisy_sim/{heldout,matched}_v1/meta.csv`, plus the clean dev source wavs
>   (`/mnt/paml-research/RTCFake/data/wav/dev`) for the non-speech tertiles, plus `scipy.signal` filter responses.
>   They reproduce every null-panel number of CRIT2 O2.1 exactly (NS mean +0.32 / SD 0.27, VAD −0.57 / 0.24,
>   PLC −0.57 / 0.41) and the v2 B2f tables;
> - **(measured)**: v2's in-memory CPU-second measurements, unchanged (PV2 §1.2).
>
> Nothing was written except this file. No progress-split file was read.

## 0. Plan in one paragraph

B2f trains on one acoustic stage, then Opus, and nothing else: "No DSP stage of any kind has ever been in
training" (AUD G1). RTCFake's training split contains Zoom / QQ online clips, i.e. platform DSP on *clean* input
only; so **B2f never saw noisy speech through a suppressor, and never a simulated DSP stage.** The plan keeps
the **three single-component, CPU-side, default-off levers** of v1 / v2, unchanged in mechanism, flags, hooks,
RNG isolation and cost: `apm_ns` (WebRTC NS only, pywebrtc-audio standalone `NoiseSuppressor`), `dtx_gate`
(webrtcvad DTX gating, zero / LPC-CNG fill) and `opus_plc` (in-process libopus, Gilbert-Elliott loss, decoder
PLC, 48 kHz decode). What v3 changes is **how their mechanism is read** (O2.1): every contrast is now calibrated
on a **null panel of 9 earlier runs** that never saw NS / VAD / PLC, re-centred on the panel mean, required to
beat the panel's spread and minimum, and required to beat the **other B4 arms** (placebo arms). The power of
each contrast under this calibration is stated honestly: `apm_ns`'s NS contrast can show a mechanism; `dtx_gate`
rests on its calibrated tertile criterion; `opus_plc`'s contrast cannot show a mechanism in one arm (its full
typical gap is < 1 null SD), so its verdict is capped at "directional" and its decision rests on the outcome,
the guards and its pre-registered attribution control. The `apm_ns` selftest is made robust (O2.2), the heldout
VAD stage is relabelled and split by fill (O2.3), and the tertile measure is pinned in frame counts and
reproduces the v2 table exactly (O2.4).

## 1. Summary

| lever | flags (default = B2f) | wave arm | gap | primary mechanism readout (full set, final readout, vs B2f; computed) | est LOC | VRAM | epoch × (saturated loader) | smoke min | new deps (import name) |
|---|---|---|---|---|---|---|---|---|---|
| `apm_ns` | `--aug_apm_p 0.0`, `--aug_apm_ns_levels 6 12 18 21` | `--aug_apm_p 0.3` | G1 | NS contrast (offline with NS vs without, 276 / 106): "shown" needs DiD ≤ **−0.22** (null μ +0.32, σ 0.27) + placebo rank; refuted if DiD ≥ +0.32 | 200 | 0 | ≤ 1.06 (estimate; stop rule 1.15) | 4.5 | `pywebrtc_audio` |
| `dtx_gate` | `--aug_dtx_p 0.0`, `--aug_dtx_fill mix` | `--aug_dtx_p 0.3` | G3, G1 | tertile criterion (ii): drop_high − drop_low ≥ 5 (null μ −10.2, σ 5.8, max 0) and ≥ 0 on half B; plus VAD contrast (136 / 246) DiD < −0.57 + placebo rank | 210 | 0 | ≤ 1.02 | 4.5 | `webrtcvad` |
| `opus_plc` | `--aug_opus_plc_p 0.0`, `--aug_opus_loss 0.01 0.20`, `--aug_opus_burst 1.0 4.0` | `--aug_opus_plc_p 0.3` | G1 | online PLC contrast (114 / 254): capped at "directional" (DiD < −0.57 + placebo rank); refuted if DiD ≥ −0.57. Decision on outcome + guards + zero-loss control | 280 | 0 | 0.85-0.87 projected; 1.0 declared | 4.5 | none (system libopus 1.3.1) |

**Smoke budget (LINT rule).** S0 4.5 + eval path 2.0 + 3 × 4.5 = **20.0 min ≤ 25** (computed). No lever adds
parameters, so there is no overfit job.

### 1.1 The mechanism readout, v3: calibrated on a null panel and on placebo arms (O2.1)

**Why v2's rule was miscalibrated** (CRIT2 O2.1, reproduced). v2 compared each arm with B2f and set the bar at
−0.4·G with a clip-bootstrap CI, i.e. against *within-run* noise only. Unrelated retrains move the same-source
contrasts by 0.3-1.1 logit, and B2f sits at the edge of the panel on them: the NS DiD vs B2f has panel mean +0.32
(B2f's NS gap, 0.85, is the third smallest of the 10 runs; a typical null run has 1.17), and the VAD and PLC DiDs
have mean −0.57 (B2f's VAD gap is the largest of the 10 runs and its PLC gap the second largest). Any retrain
regresses toward the panel, so v2's rule would call a null `opus_plc` "shown" in 3 of 10 runs and a real
`apm_ns` "refuted" in most.

**Statistic (unchanged).** For an arm R, contrast c and readout r: Δᵢ = clip(logitᵢ(R)) − clip(logitᵢ(B2f, r)),
clipped to ±8, and **DiD = mean_T(Δ) − mean_C(Δ)**, with a 2,000× clip-bootstrap 95 % CI. Negative = the treated
slice moved toward bona relative to its same-source control. Secondary: clip DiD = (Δrecall_T − Δrecall_C) · n_T;
paired fixed / broken counts with an exact sign test; exclusive-stage counts.

**Null panel** (`--null_panel`, pinned in the script; CRIT2 O2.1 (1)). The 9 earlier runs with heldout logits
and no NS / VAD / PLC exposure (B3d's `-packet_loss` was only the encoder hint, PV2 §2.3):

| run | config vs B2f (from the run name) | final readout |
|---|---|---|
| B2b | lvl, lr 1e-5 | `epoch_8` |
| B2c | lvl, lr 5e-6 | `epoch_8` |
| B2d | lvl, cos10, lr 5e-6 | `epoch_10` |
| B2e | lvl (one flag off B2f) | `epoch_8` |
| B3a | cjs 0.3, bs 8 | `epoch_8` |
| B3b | cjs 0.3 + balanced sampler | `epoch_8` |
| B3c | cjs 1.0 | `epoch_8` |
| B3d | cjs 0.3 + random Opus profiles | `epoch_8` |
| B3e | cjs 0.3 + SNR 0-20 | `epoch_8` |

B2a is printed and **flagged "failed run, excluded"** (half-B WF1 85.18 final, heldout F1 79.85; its
`metrics.json`). The merge side uses the same runs' `merge_top5` against B2f `merge_top5`. Including B2f itself
(DiD 0 by construction) would move μ by ≤ 0.06 (computed); it is the reference, so it is left out, as CRIT2
required.

**Panel statistics and the bars they imply** (computed; full set, final readout, DiD vs B2f `epoch_8`; σ with
ddof 1; "typical gap" = G + μ, the gap of an average null run):

| contrast (full set) | n_T / n_C | B2f final recall T / C | G (B2f) | D merge / final (clips) | within-run: B2f merge − final | **panel DiD μ / σ / min / max** | half-B panel μ (final / merge) | panel clip-DiD μ | typical gap | **"shown" bar on DiD** | z of closing the whole typical gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **NS**: offline with NS vs without | 276 / 106 | 69.2 / 75.5 | 0.85 | 30.1 / 17.3 | +0.22 | **+0.32 / 0.27 / −0.03 / +0.68** | +0.57 / +0.23 | −8.2 | 1.17 | **−0.22** | −4.29 |
| NS secondary: `afftdn` vs `anlmdn` (offline) | 139 / 137 | 62.6 / 75.9 | 2.01 | 21.5 / 18.5 | +0.02 | −0.84 / 0.43 / −1.66 / −0.23 | −0.61 / – | +8.4 | 1.17 | −1.70 ("strong") | −2.72 |
| **VAD**: offline with VAD vs without | 136 / 246 | 66.2 / 73.6 | 1.05 | 13.6 / 10.1 | −0.05 | **−0.57 / 0.24 / −0.81 / −0.15** | −0.23 / −0.34 | +5.4 | 0.48 | **−1.05** | −2.00 |
| VAD, zero fill only (report, O2.3) | 75 / 246 | 68.0 / 73.6 | 0.87 | 5.5 / 4.2 | −0.26 | −0.72 / 0.42 / −1.14 / +0.28 | −0.34 / – | +3.5 | 0.15 | – | – |
| VAD, CNG fill only (report, O2.3) | 61 / 246 | 63.9 / 73.6 | 1.27 | 8.1 / 5.9 | +0.21 | −0.38 / 0.35 / −0.75 / +0.45 | −0.11 / – | +1.9 | 0.89 | – | – |
| **PLC**: online with PLC vs without | 114 / 254 | 67.5 / 75.6 | 0.95 | 11.1 / 9.2 | +0.08 | **−0.57 / 0.41 / −1.11 / +0.03** | −0.67 / −0.59 | +6.2 | 0.38 | **−1.39** (unreachable) | −0.93 |

Per-run DiDs (computed, B2a first, then the 9 panel runs in table order):
- NS: −1.42 | −0.02, −0.03, +0.39, +0.36, +0.49, +0.41, +0.62, +0.68, +0.01;
- VAD: +0.67 | −0.15, −0.17, −0.68, −0.74, −0.51, −0.70, −0.67, −0.65, −0.81;
- PLC online: −0.79 | −0.42, −1.08, −1.11, −0.73, −0.54, −0.09, −0.33, −0.88, +0.03.

The **"shown" bar** is min(μ − 2σ, panel min, μ − 0.4·G): NS min(−0.22, −0.03, −0.02) = −0.22; VAD
min(−1.05, −0.81, −0.98) = −1.05; PLC min(−1.39, −1.11, −0.95) = −1.39 (computed from unrounded values). In all
three, μ − 2σ binds. The within-run row (|merge − final| ≤ 0.22) is kept as the within-run noise reference; the
cross-run spread (σ 0.24-0.41) is what the verdict is calibrated on.

**Pre-registered contrast verdict** (full set, final readout primary; DiD* = DiD − μ, z = DiD* / σ):
- **shown** — all of:
  - (a) DiD ≤ the bar above, i.e. z ≤ −2, below every null run, and DiD* ≤ −0.4·G;
  - (b) the bootstrap 95 % CI upper bound of DiD < μ (the CI of DiD* excludes 0);
  - (c) clip DiD* = clip DiD − panel clip μ ≥ +0.4·D_final;
  - (d) half-B DiD* < 0 on **both** readouts, each re-centred on its own readout's half-B panel mean;
  - (e) **placebo arms** (CRIT2 O2.1 (3)): the arm's DiD on its own contrast is below the DiD of **every other
    B4 arm** on that contrast, same readout. Each B4 arm is one flag off B2f with the same code and seed, so the
    other arms are the tightest null for a contrast they do not target.
- **directional**: DiD* < 0 and (e), but not all of (a)-(d).
- **inconclusive**: DiD* < 0 without (e).
- **refuted**: DiD* ≥ 0 (no better than an average retrain that never saw the stage).

The verdict is **provisional** until the other B4 arms have readouts ((e) = "pending"); the wave SUMMARY
re-issues every verdict once all arms are read. No panel run reaches (a) (every bar is below the panel minimum);
the placebo rank adds a further 1-in-3 factor under the null. Under a real effect exactly at the bar (z = −2),
P(refuted) ≈ 2 %.

**Power, stated per lever** (from the table): the bar asks for 46 % of the typical NS gap (reachable; closing all
of it is z −4.29), 100 % of the typical VAD gap (borderline; z −2.00) and 216 % of the typical online-PLC gap
(unreachable; z −0.93). The lever verdicts below follow from that:

| lever | lever-level "mechanism shown" | "refuted" |
|---|---|---|
| `apm_ns` | NS contrast **shown** | NS DiD* ≥ 0 (DiD ≥ +0.32) |
| `dtx_gate` | criterion (ii) holds **and** the VAD contrast is at least **directional**; "strong" if the VAD contrast is shown | VAD DiD* ≥ 0 **and** (drop_high − drop_low) ≤ the panel mean (−10.2) |
| `opus_plc` | **not attainable in one arm**; capped at **directional** | PLC DiD* ≥ 0 (DiD ≥ −0.57) |

**Outcome readouts are not re-centred.** B2f is not an outlier on the outcome: heldout F1 88.27 final vs the panel
mean 88.50 (SD 1.35), 87.92 merge vs 88.25 (computed from the 9 runs' `metrics.json`). The heldout-F1 / WF1 /
clean / matched thresholds therefore stay B2f-referenced, and the WF1 decision stays the wave's CI rule on half B.

**Saturated-core readout for `dtx_gate`, pinned (O2.4).** The bona clips of matched + heldout are split by the
non-speech share of the **clean source clip** in the scored window. Definition, pinned in the script:
1. read the source wav (first channel), take the first 64,600 samples, tiled if shorter (`utils/data_utils.pad_audio`,
   `random_start=False`);
2. 320-sample (20 ms) non-overlapping frames: **201 full frames**, the last 280 samples dropped;
3. frame energy e = 10·log10(mean(x²) + 1e-12) in float64; threshold = `numpy.percentile(e, 95)` (linear) − 30 dB;
4. k = number of frames with e < threshold;
5. **low: k ≤ 32; mid: 33 ≤ k ≤ 50; high: k ≥ 51** (cuts as frame counts; 32 / 201 = 0.1592, 50 / 201 = 0.2488).

With these cuts (computed): 259 / 250 / 241 of the 750 bona sources. **Matched and heldout share the same 750
bona sources, so every n counts each source twice** (effective n = 750).

| tertile (k of 201) | full set n (sources × 2) | full set FA merge / final | **full set saturated FA (p ≥ 0.99)** merge / final | half-B n | half-B saturated FA merge / final |
|---|---|---|---|---|---|
| low (k ≤ 32) | 518 | 102 / 97 | **63 / 66** | 270 | 43 / 44 |
| mid (33-50) | 500 | 130 / 126 | **73 / 76** | 268 | 47 / 47 |
| high (k ≥ 51) | 482 | 159 / 154 | **109 / 107** | 244 | 46 / 42 |

This is exactly the v2 table. CRIT2's 484 / 534 / 482 came from the float cut "≤ 0.159", which is below
32 / 201 = 0.15920 and moves the 17 sources (34 clips) with k = 32 from low to mid (computed: 17 ties at k = 32,
15 at k = 50). `checks dsp_slices` must reproduce this table.

**Null panel of criterion (ii)** (computed, pinned cuts, drop = B2f − run, same readout):

| | B2a (excluded) | B2b | B2c | B2d | B2e | B3a | B3b | B3c | B3d | B3e | panel μ / σ / max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| drop_high − drop_low, full set, final | +14 | −12 | −12 | −21 | −5 | −12 | −12 | −8 | −10 | 0 | −10.2 / 5.8 / 0 |
| half B, final | −11 | −17 | −14 | −16 | −14 | −14 | −14 | −12 | −15 | −5 | −13.4 / 3.5 / −5 |
| half B, merge | −4 | −2 | −4 | −10 | −12 | −6 | −7 | −11 | −6 | −3 | −6.8 / 3.6 / −2 |

Criterion (ii) — **≥ +5 on the full set (final) and ≥ 0 on half B for both readouts** — is beyond every panel run
on all three rows (z = +2.6 on the full set); no null run passes it, B2a included. It stays as in v2, as CRIT2
§5 recommended.

**Heldout VAD split by fill (O2.3).** The reserved `sim_ops.vad_gate` fills 50 / 50 with exact zeros or white
noise at 0.5 × the gated floor (`utils/sim_ops.py` l.103-110); offline bona full set: zero 75 / CNG 61. The
script prints the VAD contrast for `vad:zero` and `vad:cng` separately (rows above). Two facts from the panel:
the typical deficit sits almost entirely in the **CNG** half (typical gap 0.89 vs 0.15 for zero fill), and the
zero fill is the same as `dtx_gate`'s zero fill. Pre-registered reading: a VAD gain confined to `vad:zero`
(zero-fill DiD* ≤ −σ_zero = −0.42 while CNG-fill DiD* ≥ 0) is **near-in-distribution** and is not read as
transfer; transfer to the unseen detector / CNG colour is claimed only if the CNG-fill DiD* < 0 as well.

**Promotion arithmetic** (v1, unchanged). On half B one bona FA turned correct is worth +0.037 WF1, so the ±0.5
band (`exploop/B3/wave.json` `rules.band`) needs about 14 fewer noisy bona FA. The half-B stage deficits are 10.8
(NS), 3.9 (VAD) and 2.4 + 3.3 (PLC) clips on the final readout (PV2 §1.1). **No lever can clear the band through
its own slice alone**; every WF1 claim is a spill-over claim.

### 1.2 Cost in CPU-seconds (unchanged from v2, measured there)

- B2f loader path: **0.350 CPU-s per clip** (load + loudness 0.0268, augmenter 0.0128, ffmpeg children 0.3059,
  featurize 0.0044). B2f runs at 40.5-41.1 clips/s, i.e. 14.2-14.4 of the container's 16-CPU quota
  (`/sys/fs/cgroup/cpu.max` = `1600000 100000`). The loader is treated as saturated: a stage costing c CPU-s per
  processed clip at rate p projects to (0.350 + p·c) / 0.350; the 1.3 cap is c = 0.35 at p = 0.3.
- `opus_plc`: in-process libopus with 48 kHz decode + `resample_poly` 0.109 vs the ffmpeg path 0.258 CPU-s per
  codec call (paired, 24 clips) → projected **0.85**; declared 1.0.
- `dtx_gate`: CNG 0.0012 + framing 0.00025 CPU-s measured; webrtcvad estimated ≤ 0.01 → ≤ 1.01; declared 1.02.
- `apm_ns`: not installed, estimated 0.02-0.07 CPU-s → 1.02-1.06; declared 1.06; **stop rule**: if
  `checks timing` projects > 1.15, the implementer reports it and the lever returns to the owner before smoke.

## 2. Levers

### 2.0 Rules shared by all three levers (unchanged from v2 unless marked)

**Hook and order.** `RTCAugmenter.__call__` (`SLS_setup/utils/rtc_augment.py` l.355-368, re-read this session)
runs `_stage`, then the codec (`codecs_augm`, then `finalize`). B4 overrides `__call__` in
`RTCAugmenterB4(RTCAugmenterB3)` (`b3wave/augment.py` l.104) and inserts `_post_dsp` between them. Fixed order:
1. the parent `_stage`; 2. `wrtc_ns`; 3. `wrtc_dtx`; 4. the codec — the original `codecs_augm`, or the libopus PLC
path for clips `opus_plc` picks. This is noise → front-end → codec, the order of RTCFake / TFCL and of the heldout
builder ("faithful order: noise -> platform DSP -> codec", `scripts/build_dev_noisy_sim.py` l.189). Augmentation
runs on the whole clip before the random 4.04 s crop.

**Separate RNG stream.** Every B4 draw comes from `rng4 = np.random.default_rng([base_seed, os.getpid(), 0xB4])`,
created lazily and re-created on a pid change like `RTCAugmenter._get_rng` (the loader has
`persistent_workers=True`, `b3wave/train.py` l.55). `base_seed` is `cfg.seed` (1234 in B2f); with `None` it is a
random 31-bit draw, as in `make_rng`. The parent `rng` therefore makes exactly the B2f stage and codec draws.

**Lever off = base.** With every B4 flag at default, `build_rtc_augmenter_b4(args)` returns
`build_rtc_augmenter_b3(args)` unchanged (`b3wave/augment.py` l.223). No package or `ctypes` handle is loaded;
`describe()`, `meta_keys` and the `config.yaml` augmenter keys are b3wave's; no lever line is printed. Only the
scaffold's `b4wave: <version>` line replaces `b3wave: <version>`, which the wave's early check expects.

**B3 lesson 4** (AUD §6): every input-side change that made training harder cut the late fit. Guards: dose 0.3 on a
separate stream; NS and DTX *remove* energy; PLC touches ≤ 20 % of packets on 30 % of clips; app profiles fixed.

**Shared under-fit falsifier** (every lever): ep1 TrainAcc < 85.36 (B2f 87.36 − 2) **or** ep8 TrainLoss > 0.0752
(1.5 × 0.050114), **together with** heldout bona recall down. Source: RUN `logs/train.log` l.1 "Epoch 1/8
TrainLoss=0.355795 TrainAcc=87.36%" and l.15 "Epoch 8/8 TrainLoss=0.050114" (verified by CRIT2 §1).

**Shared class telemetry.** `b4wave/train.py` sums each batch's `meta` vector separately for bona and spoof and,
when a lever is on, logs `Epoch N B4 aug-by-class bona: … | spoof: …` and writes `metrics.jsonl["b4"]["aug_by_class"]`.

**Spoof-miss guards** (Poisson 99 % upper bound of each B2f baseline count; computed in v2):

| guard | n | B2f misses merge / final | bound |
|---|---|---|---|
| full set, offline spoof with NS | 1,135 | 18 / 18 | ≤ 29 |
| full set, offline spoof with VAD | 493 | 15 / 13 | ≤ 25 / 22 |
| full set, spoof with PLC (offline 512 + online 504) | 1,016 | 16 / 15 | ≤ 26 / 25 |
| half B, noisy spoof misses (matched + heldout; AUD §3.2) | | 24 / 23 | ≤ 36 / 35 |

**G6 labelling** (O1.4, **O2.3 corrected**). After B4, heldout slices must be reported as:

| heldout slice | label |
|---|---|
| `ns:*` | "NS family seen (WebRTC NS); implementation unseen (`afftdn` / `anlmdn`)" |
| `vad:*` | "**gating family and zero fill seen; detector (energy VAD, p10 + margin, hangover) and white-noise CNG colour unseen** (CNG level overlaps: heldout 0.5 × floor = −6 dB, `dtx_gate` −6..0 dB)". Read `vad:zero` as near-in-distribution. |
| `plc` | "loss-pattern family seen (Gilbert-Elliott); codec + concealment unseen (repeat-PLC after G.722 / G.726 / GSM)" |
| no-DSP, `agc:*`, `bw`, noise families, codecs | fully unseen |

---

### 2.1 `apm_ns`: WebRTC noise suppression (NS only) between the noise stage and the codec

**Motivation.**
- *Audit evidence (computed).* Within offline sources, NS-treated heldout bona trails untreated bona, 69.2 vs 75.5
  (full set, final; D 17.3 clips; logit gap 0.85); the deficit is `afftdn` (62.6 vs `anlmdn` 75.9). **Against the
  null panel this deficit is robust and larger than on B2f**: the typical NS gap of a null run is 1.17, and B2f's
  0.85 is the third smallest of the 10 runs, after B2c 0.82 and B2b 0.83 (computed). It is the best-powered
  contrast of the plan (§1.1).
- *Training exposure.* B2f saw platform NS only on clean online clips (Zoom / QQ), never on noisy speech, never a
  simulated suppressor; the DeepFilterNet `suppress` stage was never enabled (AUD G1).
- *Literature.* TFCL simulates the front end with "the standard implementations provided by WebRTC"; Mix training
  takes XLSR-AASIST's full-chain EER from 22.20 to 17.39 (clean 0.23 → 0.52): "The Mix strategy partially
  alleviates the degradation." (https://arxiv.org/html/2607.17761). RTCFake names "built-in speech enhancement
  modules of communication platforms, such as noise suppression, echo cancellation, and automatic gain control"
  (https://arxiv.org/html/2604.23742v1). WCR: "detectors can often identify that speech has been processed"
  (https://arxiv.org/abs/2603.14033). PYWRTC: standalone NS at levels 0-3 = 6 / 12 / 18 / 21 dB, HPF only as an
  `AudioProcessor` option, `process()` "Internally splits into 10ms frames", wheels for Python 3.10-3.14
  (https://github.com/strands-labs/pywebrtc-audio; CRIT2 §4).

**Mechanism** (unchanged). With probability `--aug_apm_p` (from `rng4`), the post-stage audio goes through one
pass of `pywebrtc_audio`'s **standalone `NoiseSuppressor`** (16 kHz mono; **no HPF, no AGC, no AEC**). The target
is drawn uniformly from `--aug_apm_ns_levels` in dB (6 / 12 / 18 / 21 → backend levels 0-3). The pass is
class-independent, independent of the acoustic stage, and also runs on clips without a noise stage. Both classes
then carry suppressor artefacts (residual / musical noise, spectral holes, onset attenuation). One component: NS.

**Flags.**
- `--aug_apm_p` (float, default **0.0**): per-clip probability of the NS pass.
- `--aug_apm_ns_levels` (int list, default `6 12 18 21`, choices 6 / 12 / 18 / 21): suppression targets in dB.
  Read only when p > 0.

**Default equivalence.** With `--aug_apm_p 0.0`: `build_rtc_augmenter_b4` returns the B3 object; no `rng4`
draws; `pywebrtc_audio` never imported; the levels flag unread; `describe()`, `meta_keys`, header and config keys
are b3wave's; the model is untouched.

**Implementation sketch.**
- New `SLS_setup/b4wave/dsp_apm.py` (~90 LOC):
  - `LEVEL_DB = {6: 0, 12: 1, 18: 2, 21: 3}`.
  - `wrtc_ns(audio, level_db, sr=16000) -> np.float32`: clip to [−1, 1], int16, pad to a multiple of 160; a
    **fresh** `NoiseSuppressor(sample_rate=16000, num_channels=1, level=LEVEL_DB[level_db])` per clip; prime on the
    first min(0.5 s, len) and discard that output; run the whole clip from the start; back to float32; trim the
    pad. The binding's frame call lives in one `_process_frame()` helper; the import is lazy.
  - **`selftest()`, rebuilt for O2.2.** Every assert has a name; a failure raises
    `AssertionError("apm selftest [<name>] level=<dB>: <measured values> (<tolerance>)")`, and `checks apm` prints
    the measured values of every assert even when they pass. Test signals are deterministic
    (`default_rng(0)`), 16 kHz; every level runs on a fresh suppressor primed on the signal's first 0.5 s, and all
    measurements skip that primed segment:
    1. **`[shape_finite]`**: on all signals below, the output is finite and has the input's length.
    2. **`[no_gain]`, one-sided, on a speech-like non-stationary input**: 6 s of a 150 Hz harmonic complex (20
       harmonics, amplitudes 1/k) under a 4 Hz on / off syllabic envelope (50 % duty, 10 ms raised-cosine ramps),
       scaled to −30 dBFS RMS. Assert **output RMS ≤ input RMS + 0.5 dB** at every level. Attenuation is allowed
       (an NS may learn a stationary or periodic component as noise); only a gain stage fails it.
    3. **`[monotone]`**: on 6 s of white noise at −30 dBFS, the energy removed is non-decreasing in the level
       within 1.0 dB, and removed(21 dB) > removed(6 dB). This catches a binding that ignores the level.
    4. **`[no_hpf]`**: on the same 6 s white-noise run (5.5 s after priming), Welch PSDs of input and output with
       **512-point** Hann frames, 50 % overlap, averaged over all frames; the attenuation in bins 1-3
       (**31.25-93.75 Hz, DC excluded**) must be within **3 dB** of the attenuation in the 1-4 kHz band, at every
       level. *Why not CRIT2's 62.5-187.5 Hz band* (computed with `scipy.signal`): a 2nd-order 80 Hz high-pass
       attenuates 62.5 / 125 / 187.5 Hz by 5.66 / 0.67 / 0.14 dB, i.e. only **1.56 dB** averaged over that band
       (1.38 dB measured on 5 s of white noise), so it would pass a 3 dB tolerance. In bins 1-3 at 512 points
       (31.25 / 62.5 / 93.75 Hz: 16.43 / 5.66 / 1.85 dB) the same HPF reads **5.00 dB** (4.90 measured), and a
       100 Hz HPF 7.15 dB. With ~340 averaged frames the band estimate is precise to about ±0.2 dB, which answers
       the "two noisy bins on a short excerpt" concern.
- `SLS_setup/b4wave/augment.py`: `RTCAugConfigB4` gains `apm_p` and `apm_ns_levels`; in `_post_dsp`, when
  `apm_p > 0 and rng4.random() < apm_p`: `lvl = levels[rng4.integers(len(levels))]`, then
  `out = finalize(wrtc_ns(out, lvl), out)`; counts `wrtc_ns{6|12|18|21}`; on an exception the clip passes through
  and `fail_wrtc_ns` is counted; at start-up `assert_training_ops(filters=["wrtc_ns"])` plus `dsp_apm.selftest()`
  when the lever is on.
- `train.py` header when on: `B4 APM NS: on p=0.3 levels_db=[6, 12, 18, 21]
  backend=pywebrtc_audio.NoiseSuppressor hpf=off agc=off`.
- `checks.py`: **`check_apm`** (the selftest with its printed values; rejects a level outside {6, 12, 18, 21}) and
  **`apm_balance`** (v2, unchanged: 200 + 200 training clips, 25 per class × src × lang cell; conditions no stage /
  forced noise U(5, 20) dB; realised removed dB per class × condition × level over the whole clip and over
  low-energy frames; `APM_BALANCE WARN` at a class difference > 2 dB with noise / > 4 dB without). v2's proxy:
  median floor depth p95 − p10 is bona 40.8 / 44.4 dB vs spoof 35.1 / 41.2 dB (offline / online), so NS likely
  removes more from spoof; both directions are guarded.
- The selftest runs first in the implement step's `checks apm`, before any smoke; a legitimate low-band tilt of the
  suppressor itself would therefore surface there, with the per-band values printed, and go to the owner instead
  of stopping a wave arm.

**Hypothesis.** B2f reads suppressor artefacts on noisy speech as synthesis. Training both classes through WebRTC
NS on 30 % of clips makes those artefacts class-neutral, and the effect transfers to the unseen `afftdn` /
`anlmdn` suppressors.

**Predicted signature** (`scripts/b4_dsp_slices.py`, arm vs B2f same readout, panel-calibrated, §1.1):
- **Mechanism shown** (NS contrast, full set, final): (a) DiD ≤ **−0.22** (z ≤ −2 on μ +0.32 / σ 0.27, below the
  panel minimum −0.03, DiD* ≤ −0.34); (b) CI upper bound < +0.32; (c) clip DiD ≥ **−1.3** (panel clip μ −8.2 +
  0.4 × 17.3); (d) half-B DiD < +0.57 (final) and < +0.23 (merge); (e) NS DiD below the `dtx_gate` and
  `opus_plc` arms' NS DiD. Secondary: `afftdn` vs `anlmdn` DiD* < 0 (DiD < −0.84); "strong" if DiD ≤ −1.70.
- **Outcome:** heldout F1 up ≥ 0.8 from 87.92 / 88.27 (AUD §3.1); expected WF1 Δ **0 to +0.6** (spill-over needed).
- **Must not move:** clean F1 within 0.5 of 97.60 / 97.69; matched F1 down ≤ 0.5 from 91.22 / 91.24; R_spoof within
  0.5 on every set; NS spoof misses ≤ 29; ep1 TrainAcc within 2 of 87.36.

**Falsifier.**
- (1) **Mechanism refuted** — NS DiD* ≥ 0, i.e. **DiD ≥ +0.32** (no better than an average null retrain) — **and**
  heldout F1 Δ < +0.5 on both readouts: WebRTC NS exposure does not transfer. Drop.
- (2) WF1 Δ < −0.5 with matched F1 down > 1.0: losing raw-noise exposure costs more than it buys (42 % instead of
  60 % of clips keep un-suppressed noise). Drop; one lower-p retry only if the mechanism was **shown** (§1.1 rule).
- (3) NS spoof misses > 29 on either readout: the lever taught "NS ⇒ bona". Drop.
- (4) The shared under-fit falsifier.

**Equivalence test** (`python -m b4wave.checks augmenter`, CPU): (1) defaults → `build_rtc_augmenter_b4` returns an
`RTCAugmenterB3`; 200 synthetic clips without the codec + 10 with it are bit-identical to `build_rtc_augmenter_b3`
with identical `stage_counts` and `describe()`; (2) a forced `RTCAugmenterB4` with default fields is bit-identical
too; (3) with `apm_p = 1.0` and an identity stub NS, the parent `stage:*` counts over 200 clips equal the base's
(`rng4` leaves the parent stream untouched); (4) `'pywebrtc_audio' not in sys.modules` after the levers-off build.
The supervisor regression adds a re-score of the B2f checkpoint (tol_logit 1e-4) and the levers-off micro-run vs
the b3wave micro-run.

**Cost.** ~200 LOC (`dsp_apm.py` 90 incl. the rebuilt selftest, augment glue 30, checks incl. `apm_balance` 80).
VRAM 0. Epoch ≤ 1.06 (estimate) with the 1.15 stop rule.

**Smoke** (4.5 min): header `^B4 APM NS: on p=`; Σ `wrtc_ns*` / Σ `codec:*` within [0.25, 0.35] and
`fail_wrtc_ns` = 0; per-class B4 rate within 0.3 ± 0.12; selftest passed with all four named asserts
(`shape_finite`, `no_gain`, `monotone`, `no_hpf`) and their values printed; `checks timing` ≤ 1.15; finite losses;
rate ≤ 1.3 × S0 (coarse); peak VRAM ≤ S0 + 0.1 GiB.

**Risks.**
- The binding's frame-call API is known from its README only; contained in `_process_frame` + the selftest. If the
  import name is not `pywebrtc_audio`, the owner notes it at the gate.
- Transfer WebRTC NS → `afftdn` is unproven (TFCL trains and tests on WebRTC).
- Matched has raw noise and no NS: 42 % instead of 60 % of clips keep un-suppressed noise (falsifier (2)).
- Class-asymmetric realised effect: measured by `apm_balance`, killed by the NS spoof-miss guard.
- If the suppressor itself tilts the < 100 Hz band on white noise, `[no_hpf]` fails at the implement step (values
  printed) and the owner decides; it cannot stop a running arm unannounced.

---

### 2.2 `dtx_gate`: webrtcvad DTX gating with zero or LPC comfort-noise fill (mechanism unchanged)

**Motivation.**
- *Audit evidence.* `sil_only` / `noise_only` bona probes are 100 % spoof, mean dP +0.8929 / +0.8985 and
  +0.8870 / +0.8965 (AUD §3.5). 136 / 133 of the 213 / 210 half-B noisy bona FA sit at p ≥ 0.99 (AUD §3.2); no B2 or
  B3 lever moved that core (AUD G3). On the full set the saturated core concentrates in pause-rich source clips:
  107 in the high non-speech tertile vs 66 in the low one (final, pinned cuts, §1.1) — and this concentration is
  **even stronger in every null run** (criterion (ii) panel μ −10.2), so it is a property of the model family,
  not of B2f. Offline VAD-gated bona trails ungated offline bona, 66.2 vs 73.6 (full set, final; logit gap 1.05;
  typical gap of a null run 0.48, mostly in the CNG half, §1.1).
- *Literature.* SIL: "removing the silence of test speech through Voice Activity Detection (VAD) can severely
  degrade performance"; Sil-mask training lowers ASVspoof 2021 DF EER for 3 / 3 models (Table VI), LA mixed
  (https://arxiv.org/html/2309.11827, https://arxiv.org/abs/2309.11827). SSG: "EER increases from 3.6% (with
  silence) to 15.5% (trimmed silence)" (https://arxiv.org/abs/2106.12914). TFCL applies WebRTC AEC → NS → AGC →
  VAD, with VAD as the **last** front-end stage; its Table 1 columns are cumulative, so VAD's own damage is not
  isolated there (corrected wording, CRIT2 §4; https://arxiv.org/html/2607.17761). WVAD: "A frame must be either
  10, 20, or 30 ms in duration"; mode 3 is "the most aggressive" (https://github.com/wiseman/py-webrtcvad).

**Mechanism** (unchanged). With probability `--aug_dtx_p` (from `rng4`), after the NS slot and before the codec:
1. `webrtcvad.Vad(mode)` per frame, mode ∈ {1, 2, 3} and frame ∈ {10, 20, 30} ms drawn per clip;
2. forward hangover U(40, 200) ms on the speech mask;
3. non-speech frames filled with exact zeros or order-10 LPC-shaped comfort noise (RMS = gated content RMS ×
   10^(U(−6, 0)/20), floor −70 dBFS); under `mix` the fill is drawn 50 / 50 per clip;
4. 5 ms raised-cosine cross-fades; length unchanged.

Both classes are gated the same way. One component: gating. The op name `wrtc_dtx` is off the reserve
(`b3wave/heldout_reserve.py` l.34 reserves `vad_gate` by name). Relation to the reserved `sim_ops.vad_gate`
(O2.3, corrected): **the gating family and the zero fill are shared; the detector (energy VAD, p10 + U(6, 15) dB
margin, 2-10 frame hangover) and the white-noise CNG colour are not** (`utils/sim_ops.py` l.95-110).

**Flags.** `--aug_dtx_p` (float, default **0.0**); `--aug_dtx_fill` (choice `mix | zero | cng`, default `mix`;
read only when p > 0).

**Default equivalence.** As §2.1: the B3 object, no `rng4` draws, `webrtcvad` never imported, the fill flag unread,
the model untouched.

**Implementation sketch** (unchanged). `SLS_setup/b4wave/dsp_dtx.py` (~110 LOC): `wrtc_dtx(audio, rng4, sr, fill)
-> (out, gated_frac, fill_used)` (LPC autocorrelation for lags 0-10, `solve_toeplitz` + `lfilter`) and
`selftest()`. `augment.py`: `dtx_p`, `dtx_fill`; counts `wrtc_dtx_{zero|cng}`, gated-fraction bins
`dtx_g0 / dtx_g1_20 / dtx_g20_40 / dtx_g40p`, `fail_wrtc_dtx`; `assert_training_ops(filters=["wrtc_dtx"])`.
Header: `B4 DTX gate: on p=0.3 fill=mix modes=[1, 2, 3] frames_ms=[10, 20, 30] hangover_ms=40..200`. `checks.py`:
`check_dtx` (selftest) and `dtx_balance` (200 + 200 training clips, realised gated fraction per class;
`DTX_BALANCE WARN` if the class means differ by > 0.05).

**Hypothesis.** Part of the saturated core is "gated or non-speech content ⇒ spoof". Class-independent DTX gating on
30 % of clips makes gated pauses class-neutral, so VAD-gated bona and the pause-rich saturated FA shrink.

**Predicted signature** (panel-calibrated, §1.1).
- **Mechanism shown, both of:**
  - (ii) **anchor, unchanged**: drop_high − drop_low ≥ **+5** clips (full set, final; pinned cuts; baseline
    107 vs 66) and ≥ 0 on half B for both readouts. Null panel: μ −10.2 / σ 5.8 / max 0 (full), max −5 / −2 (half
    B final / merge); no null run passes.
  - (i′) the VAD contrast is at least **directional**: DiD* < 0, i.e. **DiD < −0.57**, and the VAD DiD is below the
    `apm_ns` and `opus_plc` arms' VAD DiD (placebo). "Strong" if the VAD contrast is **shown** (DiD ≤ −1.05, CI
    upper bound < −0.57, clip DiD ≥ +9.4, half-B DiD < −0.23 final / < −0.34 merge, placebo) — which needs the whole
    typical VAD gap closed (z −2.00), so it is not required.
- **Report (O2.3):** the VAD contrast split by fill. A gain confined to `vad:zero` (zero-fill DiD* ≤ −0.42 with
  CNG-fill DiD* ≥ 0) is read as near-in-distribution; transfer is claimed only with CNG-fill DiD* < 0 (DiD < −0.38).
- **Outcome** (direction only): `sil_only` bona mean dP below +0.8929 / +0.8985; half-B noisy saturated FA down from
  136 / 133; expected WF1 Δ **0 to +0.5**.
- **Must not move:** clean F1 within 0.5; noisy spoof misses ≤ 36 / 35; VAD spoof misses ≤ 25 / 22;
  `probe_padded` bona flip up by no more than 2 clips (1.07 points) from 3.74 / 5.35 %; R_spoof within 0.5.

**Falsifier.**
- (1) **Mechanism refuted**: VAD DiD* ≥ 0 (DiD ≥ −0.57) **and** drop_high − drop_low ≤ −10.2 (no better than the
  panel mean) on the full set, final, with WF1 Δ < +0.5: gating is not what drives the core. Drop.
- (2) Clean F1 down > 0.5, or a spoof-miss guard broken: the gating pattern carries class information. Drop.
- (3) The shared under-fit falsifier.

**Equivalence test.** Items 1, 2 and 4 of §2.1 (with `webrtcvad` not imported), plus item 3 with `dtx_p = 1.0`.
`check_dtx` asserts that the length is unchanged and zero-filled frames are exactly 0, CNG within ±3 dB of target,
an all-tone clip unchanged.

**Cost.** ~210 LOC. VRAM 0. Epoch ≤ 1.02.

**Smoke** (4.5 min): header `^B4 DTX gate: on p=`; Σ `wrtc_dtx_*` / Σ `codec:*` within [0.25, 0.35] and
`fail_wrtc_dtx` = 0; at least one non-`dtx_g0` bin > 0; both classes in the gated bins; `checks timing` ≤ 1.05;
finite losses; rate ≤ 1.3 × S0.

**Risks.**
- Class-dependent gating: the v1 proxy share of frames > 30 dB below p95 is bona 0.205 / 0.165 vs spoof 0.153 /
  0.155 (offline / online); guarded by `dtx_balance`, per-class telemetry and the spoof-miss guards.
- Sil-mask is mixed on LA19 unseen attacks.
- Modes 2-3 gate some low-SNR speech (counted in `dtx_g40p`).
- The typical VAD deficit is in the CNG half (white-noise colour, not trained); the zero half is near-seen and has
  little typical deficit (0.15). A gain on `vad:zero` alone proves little (O2.3 reading).
- Dependency: `webrtcvad-wheels`.

---

### 2.3 `opus_plc`: in-process libopus with bursty packet loss and decoder PLC (mechanism unchanged; readout capped)

**Motivation.**
- *Audit evidence (computed).* Online PLC-treated bona trails online untreated bona on B2f, 67.5 vs 75.6 (full set,
  final; D 9.2; logit gap 0.95). **Against the null panel this deficit is small and not robust**: the typical gap
  of a null run is 0.38 (10-run mean incl. B2f 0.44, SD 0.43; range −0.16 to +0.98), and it is **not
  dose-dependent**: over all sources, bona with realised loss ≥ 0.10 is *more* bona-ward than bona with loss < 0.05
  in B2f (gap −0.48) and in 8 of the 10 runs (10-run mean −0.52); online only, the 10-run mean recall by loss bin is
  67.7 / 70.0 / 71.4 (< 0.05 / 0.05-0.10 / ≥ 0.10) against 73.0 without PLC (§5, alternative 2). B2f's codec never
  drops a packet; B3d's `-packet_loss` was only the encoder hint, and ffmpeg 4.2 has no FEC / DTX options (SOTA
  §2.3). "Packet loss + PLC" is on the never-run list (AUD §2). Heldout applies PLC to 30 % of all heldout clips,
  after every codec (`build_dev_noisy_sim.py` l.199-201).
- *Literature.* ADD-C degrades six codecs at **packet-loss rates 0 / 1 / 5 / 10 / 20 %**; trained on clean data, its
  models degrade by 5.30 (baselines) and 2.33 (theirs) EER; trained on codec + packet loss: "The EER remains
  unchanged, with a slight improvement of 0.003% from C0 to C1" (https://arxiv.org/html/2504.12423v2). RTCFake
  lists "codec encoding and decoding, and packet loss" among the "key distortion sources"
  (https://arxiv.org/html/2604.23742v1). UR21 (https://arxiv.org/abs/2107.12018). OPUSLIB is the `ctypes`
  pattern (https://github.com/orion-labs/opuslib, https://github.com/orion-labs/opuslib/blob/master/opuslib/classes.py).
- *Why it stays an arm.* The external evidence (ADD-C) is direct, the lever is the only one that lowers the
  loader cost (0.85 projected), and packet loss is a real RTC distortion the base has never seen. Its local
  mechanism readout cannot carry the decision, so the plan says so and decides on outcome + guards + the
  attribution control.

**Measured in v2 (unchanged).** `libopus 1.3.1` loads through `ctypes.util.find_library("opus")`; the NULL-payload
PLC call returns a full frame. In-process 48 kHz decode + `scipy.signal.resample_poly(·, 1, 3)` agrees with
ffmpeg at 26.0-26.9 dB on the WB profiles (16 kHz decode 20.3-20.4) and 17.0-22.8 dB on the SILK-layer profiles;
in every profile the two codec paths are 7.8-14.5 dB closer to each other than ffmpeg is to the input (PV2 §2.3
table). Cost 0.109 vs 0.258 CPU-s per codec call.

**Mechanism** (unchanged). The codec stage fires on the parent's draw, as in B2f. For a clip the lever picks
(`rng4.random() < p`), `draw_codec_params(rng)` makes the parent's draws in exactly `codecs_augm`'s order
(`utils/rtc_augment.py` l.186-193). A `ctypes` libopus 1.3.1 encoder (16 kHz mono) uses the *same* profile
(bitrate; max bandwidth from the cutoff 4000 / 6000 / 8000 → NB / MB / WB; frame; application; VBR on, complexity
10). A Gilbert-Elliott mask drops packets: L ~ U(`--aug_opus_loss`), B ~ U(`--aug_opus_burst`), p_bg = 1/B,
p_gb = L·p_bg/(1 − L), stationary start state. A **48 kHz** decoder decodes received packets and conceals lost ones
with the NULL-payload PLC call; the output is trimmed by 3 × the 16 kHz look-ahead, resampled 48 → 16 kHz with
`resample_poly`, and padded / cut to the input length. Encoder and decoder are created per clip and destroyed in
`finally`. Unpicked clips call the original `codecs_augm`. One component: real packet loss with codec-native
concealment; FEC left out.

**Loss range** (O1.4). `0.01 0.20` = ADD-C's non-zero PLR span; the overall share of training packets lost is
0.3 × 0.105 ≈ 3 %. Burst U(1, 4) packets is a design choice; it overlaps the reserved `sim_ops.packet_loss` GE
family (p_bg U(0.3, 0.7), `utils/sim_ops.py` l.122), hence the G6 label "loss-pattern family seen, codec +
concealment unseen".

**Flags.** `--aug_opus_plc_p` (float, default **0.0**); `--aug_opus_loss` (2 floats, default `0.01 0.20`;
0 ≤ lo ≤ hi < 0.5; `0 0` = in-process codec without loss, the attribution control); `--aug_opus_burst` (2 floats,
default `1.0 4.0`; 1 ≤ lo ≤ hi).

**Default equivalence.** The B3 object, so every codec call is the original `codecs_augm` with the same parent draws;
libopus never loaded; loss and burst flags unread; the model untouched.

**Implementation sketch** (unchanged). `SLS_setup/b4wave/opus_plc.py` (~160 LOC): `_lib()` (lazy `ctypes` binding,
`find_library("opus")` then `{sys.prefix}/lib/libopus.so.0`); `draw_codec_params(rng, apps)`; `ffmpeg_opus(...)`
fallback on pre-drawn values; `opus_loss_roundtrip(audio, rng4, kbps, cutoff, frame, application, loss_range,
burst_range, sr, dec_sr=48000) -> (out, realised_loss)` with `try / finally` destroy; `selftest()` (library loads,
ctl set/get round trip for bitrate and max bandwidth, a concealed gap is not all-zero, finite output of the right
length). `augment.py` codec branch: counts `codec:{app}`, `opus_plc`, `plc_l0_5 / plc_l5_10 / plc_l10p`,
`fail_opus_plc`; init raises if `opus_plc_p > 0` with `--aug_codec_profiles random` or the codec off;
`assert_training_ops(codecs=["libopus"], filters=["opus_plc"])`. Header: `B4 Opus PLC: on p=0.3 loss=0.01..0.20
burst=1..4 decode=48k+resample_poly libopus=libopus 1.3.1`. `checks.py::check_opus_plc`: selftest,
`codec_draw_order`, fallback bit-identity, zero-loss attribution.

**Hypothesis.** Concealment artefacts are read as synthesis. Exposing both classes to libopus PLC under bursty loss
on 30 % of codec calls makes concealment class-neutral, and heldout's repeat-PLC slice (after non-Opus codecs) and
heldout F1 improve. The local evidence for the premise is weak (small, non-robust, non-dose-dependent deficit), so
this arm is a pre-registered **outcome** test of ADD-C's result in the RTC setting.

**Predicted signature** (panel-calibrated, §1.1).
- **Mechanism, capped at "directional"**: online PLC DiD* < 0, i.e. **DiD < −0.57**, and the PLC DiD below the
  `apm_ns` and `dtx_gate` arms' PLC DiD. The contrast-level "shown" bar (DiD ≤ −1.39) would need the PLC bona to end
  0.44 logit *more* bona-ward than no-PLC bona (216 % of the typical gap, z −0.93 for full closure), so "shown" is
  not claimed for this lever. Report-only: the offline PLC contrast and bona logit / recall by realised-loss bin
  (none / < 0.05 / 0.05-0.10 / ≥ 0.10, online and offline).
- **Outcome:** heldout F1 up ≥ 0.5; expected WF1 Δ **0 to +0.4**.
- **Must not move:** matched F1 down ≤ 0.5; matched qq and wechat bona recall each down by ≤ 6 points (B2f merge vs
  final already differ by 5.05 / 5.77, AUD §3.4); PLC spoof misses ≤ 26 / 25; clean F1 within 0.5; ep1 TrainAcc
  within 2.

**Falsifier.**
- (1) **Mechanism refuted** — online PLC DiD* ≥ 0, i.e. **DiD ≥ −0.57** — **and** heldout F1 Δ < +0.5: libopus PLC
  does not carry over. Drop.
- (2) WF1 Δ < −0.5 with the shared under-fit signature: a B3d-type codec-side under-fit. Drop.
- (3) **Attribution control (pre-registered):** if the arm is positive on WF1, `--aug_opus_plc_p 0.3
  --aug_opus_loss 0 0` runs before promotion; if it alone gives ≥ half of the arm's WF1 gain, the gain is credited
  to the decoder path, not to PLC. This control also serves as an extra placebo arm for the NS and VAD contrasts.

**Equivalence test.** Items 1, 2 and 4 of §2.1 (no `ctypes` handle); item 3 with `opus_plc_p = 1.0` and the codec on
(parent `stage:*` and `codec:<app>` counts equal the base); `codec_draw_order` (`draw_codec_params` leaves
`rng.bit_generator.state` where `codecs_augm` leaves it); `ffmpeg_opus` on pre-drawn values bit-identical to
`codecs_augm`; zero-loss attribution on 3 training clips × 7 profiles: aligned SNR(ffmpeg, in-process) ≥
max(15 dB, SNR(ffmpeg, input) + 5 dB) on every profile (v2 minimum margin: wechat 17.0 vs 15).

**Cost.** ~280 LOC (binding 50, round trip + GE + 48 kHz decode / resample 110, draw / fallback 40, checks 80).
VRAM 0. Projected 0.85 under a saturated loader; 1.0 declared.

**Smoke** (4.5 min): header `^B4 Opus PLC: on p=`; `opus_plc` / Σ `codec:*` within [0.25, 0.35] and
`fail_opus_plc` = 0; every `plc_l*` bin > 0; 7 apps covered; selftest passed; `checks timing` in-process CPU-s per
codec call below the ffmpeg path's; finite losses; rate ≤ 1.3 × S0.

**Risks.**
- **Weak local premise** (new): the online PLC deficit is within one null SD and shows no dose response; a null
  result on heldout is the most likely mechanism outcome, and the arm stands or falls on WF1 / heldout F1.
- Restart wipe of system ffmpeg / libopus: the selftest raises at start-up (optional: conda-forge `libopus`).
- Variadic `ctl` through `ctypes`: guarded by the set/get round trip.
- Transfer to repeat-PLC after G.7xx / GSM is unproven (ADD-C is in-distribution).
- SILK-layer decoder residual vs ffmpeg (17-22 dB): bounded by the zero-loss check and the control arm.
- 40 / 60 ms frames conceal more audio per lost packet.

## 3. Shared changes

| what | files |
|---|---|
| **Config and augmenter classes** (unchanged): `RTCAugConfigB4(RTCAugConfigB3)` with the seven fields; `RTCAugmenterB4(RTCAugmenterB3)` whose `__call__` mirrors the parent (`len < 16` return → `rng = _get_rng()` → `_stage` → `_post_dsp` (`wrtc_ns` → `wrtc_dtx`) → codec branch); `rng4 = default_rng([seed, pid, 0xB4])` re-created on a pid change; B4 `meta_keys` / `describe()` suffixes only for active levers; start-up selftests; `fail_*` counters; `assert_training_ops` for the new op names; `build_rtc_augmenter_b4` returns `build_rtc_augmenter_b3(args)` when every B4 flag is at default. | `SLS_setup/b4wave/augment.py` |
| `build_augmenter_b4` / `build_dataset_b4`, copy-adapted from b3wave; dev uses `SpoofAudioDataset`, train reuses `SpoofAudioDatasetB3`. | `SLS_setup/b4wave/data.py` |
| **Trainer** (scaffolded copy of `b3wave/train.py`): a "B4 wave" argparse group with value checks (p in [0, 1]; levels ⊂ {6, 12, 18, 21}; loss 0 ≤ lo ≤ hi < 0.5; burst 1 ≤ lo ≤ hi); the `b4wave: <version>` line; lever headers only when on; `config.yaml` `b4wave_version` + `b4_levers`; B4 keys in the existing `Epoch N B3 aug` line; per-label meta sums → `Epoch N B4 aug-by-class` and `metrics.jsonl["b4"]`. | `SLS_setup/b4wave/train.py` |
| **Checks**: `cpu` = `augmenter` (items 1-4 per lever), `codec_draw_order`, `reserve_names`, `apm` (**named asserts `shape_finite` / `no_gain` / `monotone` / `no_hpf`, values printed**, O2.2), `apm_balance`, `dtx`, `dtx_balance`, `opus_plc` (incl. zero-loss attribution); `timing` (CPU-s per clip via `process_time` + `RUSAGE_CHILDREN`, one process, OMP = 1, 40 random training clips; projected saturated-loader factor per lever; fails above 1.3, reports above 1.15 for `apm_ns`); **`dsp_slices`**: `scripts/b4_dsp_slices.py` on RUN must reproduce (i) the §1.1 B2f contrast rows (n, recall, G, D), (ii) the **null-panel μ / σ / min / max and bars** of the three primary contrasts to 2 decimals, (iii) the **pinned tertile table** (518 / 500 / 482; 63 / 73 / 109 and 66 / 76 / 107; half B 43 / 47 / 46 and 44 / 47 / 42) and the criterion-(ii) panel row (μ −10.2, max 0), (iv) the VAD-by-fill n (75 / 61 vs 246); `model --ckpt` / `optim` as in b3wave. In `all`, a lever check whose package is missing prints `SKIP`. | `SLS_setup/b4wave/checks.py` |
| Flag table, reproduction statement, telemetry keys, install lines, **corrected G6 labels** (O2.3), the verdict rules of §1.1. | `SLS_setup/b4wave/README.md` |
| **Mechanism readout, calibrated (O2.1, O2.3, O2.4).** `scripts/b4_dsp_slices.py --run <arm local_eval/readout> --ref <B2f same readout> [--ref_pair <B2f merge> <B2f final>] [--null_panel <readout dirs…>] [--null_panel_merge <readout dirs…>] [--placebo <other arms' same-readout dirs…>] [--halves AB\|B] [--json_out f]`. The default `--null_panel` is pinned in the script: `SLS_setup/exp/{B2b_qwen17_rtc_musan_lvl_cos8_lr1e5_s1234_epoch8_bs16_20261008040736, B2c_qwen17_rtc_musan_lvl_cos8_lr5e6_s1234_epoch8_bs16_20261008085023, B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925, B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846, B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234_epoch8_bs8_20261010145615, B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234_epoch8_bs8_20261010234200, B3d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_codecrand_s1234_epoch8_bs8_20261011081946, B3e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_snr0to20_s1234_epoch8_bs8_20261011170031}/local_eval/epoch_8` plus `B2d_qwen17_rtc_musan_lvl_cos10_lr5e6_s1234_epoch10_bs16_20261008132750/local_eval/epoch_10`; the merge panel is the same runs' `merge_top5`; `B2a_qwen17_rtc_musan_lvl_cos8_s1234_epoch8_bs16_20261007231952` is printed and flagged "failed run, excluded". For every contrast (NS, NS `afftdn` vs `anlmdn`, VAD, **VAD zero / CNG fill**, PLC online, PLC offline) it prints: n_T / n_C; recall T / C (arm, ref); G and D; the clipped-logit DiD with a 2,000× bootstrap CI; **the panel μ / σ / min / max, DiD*, z and the bar**; the clip DiD and clip DiD*; T fixed / broken with an exact sign-test p; the half-B DiD and DiD* per readout; **each placebo arm's DiD and the arm's rank**; the contrast verdict (shown / directional / inconclusive / refuted, "provisional" while placebos are missing) and the lever-level verdict of §1.1. It also prints exclusive-stage counts, the **pinned tertile table** and drop_high − drop_low with its panel μ / σ / max and z, the bona logit / recall by realised-loss bin, the §2.0 spoof-miss guards and the probe rows. It reads only `scores/*.{txt,logits}`, `metrics.json`, `probe_report.json`, the sim `meta.csv` and the dev source wavs. No GPU. ~260 LOC (v2 ~200). | `scripts/b4_dsp_slices.py` |

No file outside `SLS_setup/b4wave/` and `scripts/b4_*` changes. `b3wave`, `utils`, `model`, `main_*.py` and the old
scripts stay byte-identical.

## 4. Eval impact

- No state_dict change, no new eval profile. Checkpoints are plain `ModelSLS` state_dicts, scored with
  `sls_local17` and `--loudness_norm --loudness_norm_lufs -23`, as for B2f.
- The mechanism readout is `scripts/b4_dsp_slices.py`. The analyst runs it per arm on both readouts with
  `--ref` = B2f's same readout, `--ref_pair` = B2f merge / final, the default null panel, and `--placebo` = the
  other B4 arms' readouts that exist. The wave SUMMARY re-runs it once all arms are read and re-issues the
  verdicts (provisional → final). Primary: full set, final readout; half B is the re-centred sign check; the WF1
  decision stays on half B with the wave's CI rule.
- G6 labels as in §2.0 (VAD label corrected). `sim_unseen_v1` stays unread (music leak, AUD G6).
- Wave rules as in B3 (`exploop/B3/wave.json` `rules`): band ±0.5; promote = CI low > 0 and clean drop ≤ 0.5.
  Reference B2f 91.98 / 92.14 (AUD §1).
- Run length: 8 × ~31 min × ≤ 1.06 ≈ 4.4 h per arm, under the 10 h cap.

**After the wave: a recommendation, not a pre-registered arm** (O1.8, verdict names updated). The three stages sit
at disjoint points of the chain and draw from `rng4` only when on, so any combination is well defined. If
`apm_ns` and `dtx_gate` both reach lever-level "mechanism shown" (or one of them does and `opus_plc` is
directional with a positive outcome), with WF1 null or better, a combined "platform chain" run (the TFCL Mix
analogue) is the natural next step, because promotion needs about 14 fewer FA (§1.1). It is scheduled as a coupled
group or a separate wave by the owner or exp-designer. No dose sweeps, except the `apm_ns` lower-p retry (only if
its mechanism was **shown**) and the `opus_plc` zero-loss control (only if positive).

**Ideas considered and not proposed** (unchanged from v2): AGC2 inside `apm_ns` (second component; level treatment
is "not a lever", AUD §2 row 9); WebRTC HPF (second component); pause insertion / attenuation fill in `dtx_gate`
(rhythm component; `agate` reserved); libopus DTX as the gate (fired on 7.1 % of bona vs 1.2 % of spoof frames,
measured in v1); in-band FEC (second decoder component); `opuslib` dependency (system libopus via `ctypes`); I4
AugMix (raises effective SNR by ≈ 8.7 dB, vision-only evidence); I5 echo suite (is the `sim_echo_v1` recipe, not in
WF1); I6 DeepAugment-DeepFilterNet (vision-only evidence, not installed, CPU risk at ~90 % of quota).

**Dependencies** (owner action at the gate; nothing installed by this step). Import names: `apm_ns` →
`pywebrtc_audio`; `dtx_gate` → `webrtcvad`; `opus_plc` → none.
```
/root/.conda/envs/rtc-sdd/bin/python -m pip install pywebrtc-audio webrtcvad-wheels
/root/.conda/envs/rtc-sdd/bin/python -c "import pywebrtc_audio, webrtcvad; print('ok')"
# optional hardening against container restarts (system ffmpeg + libopus get wiped):
# conda install -n rtc-sdd -c conda-forge libopus
```
All are local DSP or software codecs: "local WebRTC AEC / NS / AGC / VAD simulation explicitly allowed; software
codecs allowed" (roadmap §2, AUD §5).

## 5. Responses to the critic

| id | action | response |
|---|---|---|
| **O2.1** (major) | **accept** | Reproduced exactly (computed): NS panel DiD μ +0.32 / σ 0.27 / min −0.03, VAD −0.57 / 0.24 / −0.81, PLC online −0.57 / 0.41 / −1.11; B2f merge − final +0.22 / −0.05 / +0.08. All four required parts are in:<br>(1) `--null_panel` (B2b-B2e, B3a-B3e at their final readouts, B2d `epoch_10`; merge side on `merge_top5`; B2a printed and flagged "failed run, excluded"); the script prints each contrast's μ / σ / min / max (§1.1 table, §3).<br>(2) Re-centred: DiD* = DiD − μ, z = DiD* / σ. "Shown" = DiD ≤ min(μ − 2σ, panel min, μ − 0.4·G) **and** CI upper bound < μ **and** clip DiD − panel clip μ ≥ 0.4·D **and** half-B DiD* < 0 on both readouts (each on its own readout's panel). "Refuted" = DiD* ≥ 0.<br>(3) Placebo arms: a lever's own-contrast DiD must be below every other B4 arm's DiD on that contrast; verdicts are provisional until all arms are read and the wave SUMMARY re-issues them.<br>(4) Thresholds and falsifier (1) restated on DiD*: `apm_ns` shown at DiD ≤ −0.22, refuted at DiD ≥ +0.32; `opus_plc` refuted at DiD ≥ −0.57; `dtx_gate` (i) re-centred (DiD < −0.57 for directional, ≤ −1.05 for shown), (ii) unchanged.<br>*Added*: a power statement per contrast. The bar needs 46 % of the typical NS gap (reachable), 100 % of the typical VAD gap (z −2.00 for full closure) and 216 % of the typical PLC gap (z −0.93: unreachable). Hence the lever-level rules: `dtx_gate` is anchored on (ii) plus a directional VAD contrast; `opus_plc` is capped at "directional" and decided on outcome + guards + its zero-loss control. The §4 recommendation and the `apm_ns` lower-p retry now key on these calibrated verdicts. Outcomes are not re-centred because B2f is not an outlier on heldout F1 (88.27 vs panel mean 88.50, SD 1.35; computed from `metrics.json`). |
| **O2.2** (minor) | **modify** | Accepted: the gain check is one-sided (`[no_gain]`: output RMS ≤ input RMS + 0.5 dB at every level) on a speech-like non-stationary input (150 Hz harmonic complex under a 4 Hz on / off envelope, −30 dBFS); the no-HPF check runs on 6 s of white noise (5.5 s after priming), Welch-averaged over all frames, DC excluded; every assert is named in the failure message, and `checks apm` prints all measured values. **Modified: the band.** CRIT2's 62.5-187.5 Hz band (256-point bins 1-3) would let the HPF it is meant to catch through: a 2nd-order 80 Hz high-pass reads only 1.56 dB there (5.66 / 0.67 / 0.14 dB per bin; 1.38 dB measured on 5 s of white noise; `scipy.signal`, computed), under the 3 dB tolerance. With 512-point frames, bins 1-3 = 31.25-93.75 Hz, the same HPF reads 5.00 dB (4.90 measured) and a 100 Hz HPF 7.15 dB, while ~340 averaged frames keep the estimate within about ±0.2 dB. The monotone check gets a 1 dB tolerance plus removed(21) > removed(6). The selftest runs in the implement step's `checks apm` before any smoke, so a legitimate suppressor tilt surfaces to the owner, not as an unannounced arm stop. |
| **O2.3** (minor) | **accept** | G6 label corrected (§2.0): "gating family and zero fill seen; detector (energy VAD) and white-noise CNG colour unseen", noting that the CNG level overlaps (heldout 0.5 × floor = −6 dB; ours −6..0 dB). `b4_dsp_slices.py` prints the VAD contrast split by fill (zero 75 / CNG 61 vs 246 offline bona controls; computed, matching CRIT2's 75 / 61). New fact from the panel: the typical VAD deficit sits in the CNG half (typical gap 0.89 vs 0.15 for zero fill). Pre-registered reading: a gain confined to `vad:zero` (zero-fill DiD* ≤ −0.42, CNG-fill DiD* ≥ 0) is near-in-distribution and is not read as transfer; transfer needs CNG-fill DiD* < 0. |
| **O2.4** (minor) | **accept** | Pinned in frame counts (§1.1): first 64,600 samples of the clean source (tiled if short), 201 non-overlapping 320-sample frames, e = 10·log10(mean(x²) + 1e-12) in float64, threshold `numpy.percentile(e, 95)` − 30 dB; low k ≤ 32, mid 33-50, high k ≥ 51. This reproduces the v2 table exactly (518 / 500 / 482; saturated 63 / 73 / 109 merge and 66 / 76 / 107 final; half B 43 / 47 / 46 and 44 / 47 / 42; computed). CRIT2's 484 / 534 / 482 came from the float cut 0.159 < 32 / 201 = 0.15920, which moves the 17 sources (34 clips) at k = 32 to mid. The plan now states that n counts each of the 750 bona sources twice (259 / 250 / 241 sources). `checks dsp_slices` must reproduce the table. The criterion-(ii) null panel was recomputed with the pinned cuts: full-set μ −10.2 / σ 5.8 / max 0, half B max −5 (final) and −2 (merge); no null run passes. |

**Alternatives.**

| critic alternative | action | detail |
|---|---|---|
| Null-panel and placebo-arm calibration of the mechanism readout | **adopt** | Fully specified in §1.1 and §3 (O2.1): pinned 9-run panel, re-centred DiD*, z, bar = min(μ − 2σ, panel min, μ − 0.4·G), placebo rank among the B4 arms (plus the `opus_plc` zero-loss control when it runs), provisional verdicts re-issued at the wave SUMMARY. About 60 extra lines in `b4_dsp_slices.py`, no GPU. |
| Dose-response signature for `opus_plc` (bona with realised loss ≥ 0.1 vs < 0.05 inside the PLC slice) | **decline** (kept as a report-only table) | There is no dose-dependent deficit for a lever to close, and over all sources its sign is the opposite of the premise (computed). All sources, n 69 / 65: B2f gap −0.48 (recall 72.5 vs 64.6, the high-loss bona is *more* bona-ward); 10-run mean −0.52, SD 0.42; only 2 of 10 runs are above 0. Online only, n 36 / 30: B2f +0.17, panel DiD μ −0.59, SD 0.67. By bin (online, 10-run mean recall): none 73.0, < 0.05 67.7, 0.05-0.10 70.0, ≥ 0.10 71.4. The PLC deficit is presence-related and small, not dose-related, so a dose contrast would be a null-on-null readout with SD ≥ 0.42 at n ≈ 65 per side. The by-bin table is printed as a report, and the finding is recorded as a risk against `opus_plc`'s premise (§2.3). |

## 6. Changes since v2

1. **Mechanism readout calibrated on a null panel and placebo arms** (O2.1): pinned 9-run panel (B2a flagged),
   DiD* / z, bar = min(μ − 2σ, panel min, μ − 0.4·G), re-centred CI / clip / half-B conditions, placebo rank,
   provisional → final verdicts, a power statement per contrast, lever-level rules (`dtx_gate` anchored on (ii);
   `opus_plc` capped at "directional"). The B2e cross-run column is replaced by the panel.
2. **Per-lever thresholds and falsifiers restated on DiD***: `apm_ns` shown ≤ −0.22 / refuted ≥ +0.32;
   `dtx_gate` (i′) < −0.57 directional, ≤ −1.05 strong, refuted = VAD DiD ≥ −0.57 and drop_high − drop_low ≤ −10.2;
   `opus_plc` refuted ≥ −0.57. The §4 recommendation and the `apm_ns` lower-p retry key on the calibrated verdicts.
3. **`apm_ns` selftest rebuilt** (O2.2): named asserts `shape_finite` / `no_gain` (one-sided, speech-like AM input)
   / `monotone` (1 dB tolerance) / `no_hpf` (6 s white noise, 512-point Welch, 31.25-93.75 Hz vs 1-4 kHz, DC
   excluded); values printed. LOC 190 → 200.
4. **G6 VAD label corrected and VAD split by fill** (O2.3), with the near-in-distribution reading for `vad:zero`.
5. **Tertile measure pinned in frame counts** (O2.4); reproduces the v2 table; n counts each source twice; panel of
   criterion (ii) recomputed with the pinned cuts.
6. **`opus_plc` motivation restated honestly**: the online PLC deficit is small, within one null SD and not
   dose-dependent; the arm is an outcome test of ADD-C, with a new "weak local premise" risk. Mechanism, flags,
   code and cost unchanged.
7. **Wording**: TFCL's VAD is "the last front-end stage", not "the most damaging" (CRIT2 §4). B2f's position on the
   panel is stated exactly (NS gap third smallest; VAD gap largest; PLC gap second largest of 10).
8. `scripts/b4_dsp_slices.py` grows to ~260 LOC (panel, placebo, fill split, loss bins); `checks dsp_slices` must
   also reproduce the panel stats and the pinned tertile table.
