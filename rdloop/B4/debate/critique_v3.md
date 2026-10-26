# rdloop B4 - critique of proposal v3

> research-critic, CRITIQUE round 3 of 3, 2026-10-17.
>
> **Read:**
> - cycle files: `rdloop/B4/{brief.md,cycle.json,user_notes.md}` (no owner notes);
> - debate files: `debate/proposal_v3.{md,json}`, `debate/lint_v3.json` (no errors; `needs_user`: apm_ns →
>   `pywebrtc_audio`, dtx_gate → `webrtcvad`), `debate/ledger.json`, `debate/critique_v2.md`, `scout.json` (idea ids I1-I6).
>
> **Code checked:**
> - `SLS_setup/utils/rtc_augment.py`: l.244 (pid-seeded `make_rng`), l.349-352 (`_get_rng`);
> - `SLS_setup/b3wave/augment.py`: l.104 (`RTCAugmenterB3`), l.223 (`build_rtc_augmenter_b3`);
> - `SLS_setup/b3wave/train.py`: l.55 (`persistent_workers`);
> - `SLS_setup/b3wave/heldout_reserve.py`: l.33-34 (reserved `afftdn`, `vad_gate`, `packet_loss`);
> - `scripts/build_dev_noisy_sim.py`: l.185-201 (heldout NS / AGC / VAD / BW are independent draws on offline sources only);
> - `docs/experiments_roadmap.md`: l.108-112 (local WebRTC simulation and software codecs allowed).
>
> **"(computed)"** means a read-only join, run in this session, of
> `SLS_setup/exp/<run>/local_eval/<readout>/scores/sim_{heldout,matched}_v1.{txt,logits}` and `metrics.json`
> with `data/dev_noisy_sim/{heldout,matched}_v1/meta.csv`, plus the dev source wavs for the tertiles.
> - Scripts: `/tmp/b4crit3/panel3.py`, `/tmp/b4crit3/tert3.py`, plus inline `metrics.json` / `scipy.signal` one-liners.
> - Nothing in the repo was written except this file.
> - No progress-split file was read.

## 1. Verdict

v3 answers all four open objections. I re-derived its numbers independently and **every one of them reproduces
exactly**:

| what | reproduced values |
|---|---|
| NS panel (μ / σ / min / max) | +0.32 / 0.27 / −0.03 / +0.68 |
| VAD panel (μ / σ / min / max) | −0.57 / 0.24 / −0.81 / −0.15 |
| PLC-online panel (μ / σ / min / max) | −0.57 / 0.41 / −1.11 / +0.03 |
| half-B panel means | +0.57 / +0.23; −0.23 / −0.34; −0.67 / −0.59 |
| panel clip-DiD means | −8.2 / +5.4 / +6.2 |
| VAD-by-fill rows | zero 75 vs CNG 61; typical gaps 0.15 / 0.89 |
| pinned tertile table | 259 / 250 / 241 sources; 518 / 500 / 482 clips; saturated 63 / 73 / 109 and 66 / 76 / 107; half B 43 / 47 / 46 and 44 / 47 / 42; 17 ties at k = 32 |
| criterion-(ii) panel | μ −10.2 / −13.4 / −6.8, max 0 / −5 / −2 |
| heldout F1 (half B, `metrics.json`) | B2f 88.27 / 87.92; panel 88.50 (SD 1.35) / 88.25 |
| B2a | half-B WF1 85.18, heldout F1 79.85 |
| no-HPF band attenuation (80 Hz 2nd-order HPF) | 5.00 dB (512-point bins 1-3) vs 1.56 dB (256-point) |

The calibrated rule also holds up out of sample. I applied v3's conditions (a)-(d) **leave-one-out** to each of
the nine panel runs, recomputing μ, σ, the minimum and the clip / half-B means without the run itself:
- **No panel run reaches (a) on any of the three primary contrasts**, so the false "shown" rate that O2.1 found
  (3 / 10 on PLC) is gone.
- A null run is "refuted" on NS in 6 / 9 cases and "directional-eligible" in 3 / 9, which is the intended
  behaviour.

On O2.2, the architect correctly rejected my suggested 62.5-187.5 Hz band: an 80 Hz HPF reads only 1.56 dB
there. Their 31.25-93.75 Hz band (512-point) catches it at 5.00 dB.

I fetched the pywebrtc-audio bindings source. Its `NoiseSuppressor` wraps `webrtc::NoiseSuppressor` +
`AudioBuffer` directly, not `AudioProcessing`, so no HPF can be forced by APM's "HPF required with NS" logic. The
O1.1 concern is closed at the source level.

The levers, hooks, RNG isolation, default-off path, cost (≈ 4.4 h per arm, 0 extra VRAM) and smoke budget
(20 ≤ 25 min) are unchanged from v2 and still sound.

**No blocker or major objection remains. Verdict: agree.**

Three new **minor** points concern the edges of the new readout. They can be folded into `b4_dsp_slices.py`
and the README at implement time:
- a failed run can fake an NS "shown" (B2a does);
- the outcome thresholds that let an arm escape falsifier (1) sit inside cross-run noise;
- the placebo rule (e) has no rule for an arm that never runs.

## 2. Objections

| id | lever | severity | category | claim | evidence | required change |
|---|---|---|---|---|---|---|
| O3.1 | (all) | minor | eval | **"Shown" is not conditioned on run health, and a failed run passes it.** Conditions (a)-(d) compare DiDs only, and the placebo rule (e) cannot catch a single broken arm, because the other arms are healthy. The failed B2a, evaluated leave-one-out against the 9-run panel, has NS DiD −1.42 (z −6.38) and passes (a), (c) and (d). It would read NS "mechanism shown" while its half-B WF1 is 85.18 and its heldout F1 79.85. The same run is wildly off on the non-target VAD contrast (DiD +0.67, z +5.11). This matters because `apm_ns` falsifier (2) grants a lower-p retry exactly when WF1 drops **and** the mechanism is "shown", and the §4 combined-run recommendation keys on "shown". | Computed (`/tmp/b4crit3/panel3.py`, LOO): B2a NS z −6.38, a = c = d = True; VAD z +5.11. `SLS_setup/exp/B2a_…/local_eval/epoch_8/metrics.json`: half_B wf1 0.8518, sim_heldout_v1 f1 0.7985. `proposal_v3.md:350-351` (retry only if shown), `:609-615` (§4 keys on shown), `:118-121` (placebo rule). | In `b4_dsp_slices.py`, print the arm's z on **every** contrast. Flag the run "anomalous" if \|z\| > 3 on a non-target primary contrast, or if the shared under-fit falsifier or the clean-F1 guard fires. While flagged, report the lever verdict as "confounded (run anomaly)" instead of "shown". The `apm_ns` lower-p retry then needs "shown" without that flag. |
| O3.2 | (all) | minor | eval | **The outcome thresholds that let an arm escape falsifier (1) are inside cross-run noise.** v3 states the half-B heldout-F1 panel SD (1.35) but keeps B2f-referenced thresholds of +0.5 / +0.8 (z ≈ 0.4 / 0.6). Against B2f: 5 / 9 panel runs reach +0.5 on the final readout, 6 / 9 on the merge readout and 6 / 9 on at least one readout; 4 / 9 reach +0.8 (final). The panel mixes config effects with noise, so it bounds the null rather than measuring it. Even so, the one-flag B2e reads +0.12 / +0.58. Consequences:<br>• falsifier (1) of `apm_ns` and `opus_plc` fires for a null arm only about 0.5 × 3/9 ≈ 1 time in 6;<br>• the §4 "opus_plc directional with a positive outcome" path is weak evidence. | Computed from the 9 runs' `local_eval/{epoch_8\|epoch_10,merge_top5}/metrics.json` (`half_B.sets.sim_heldout_v1.f1`). Final Δ vs 88.27: B2b −2.14, B2c +1.19, B2d −0.07, B2e +0.12, B3a +2.05, B3b +0.89, B3c +0.59, B3d −1.59, B3e +1.07. Merge Δ vs 87.92: −2.11, +1.06, −0.01, +0.58, +2.28, +1.48, +0.63, −1.56, +0.60. `proposal_v3.md:140-142`, `:343`, `:348-349`, `:546`, `:552-553`, `:611-613`. | Next to every heldout-F1 threshold, print the panel SD and the number of panel runs that clear it. In falsifier (1), an arm escapes on outcome only with heldout F1 Δ ≥ +0.5 on **both** readouts (`opus_plc`: state the readouts). For the §4 recommendation, define "positive outcome" as the wave's WF1 CI rule, not heldout F1 alone. |
| O3.3 | (all) | minor | eval | **The placebo rule (e) has no rule for a missing arm.** Verdicts stay "provisional / pending" until all B4 arms have readouts, but two arms depend on owner-installed packages (lint `needs_user`). An arm that never runs (install failure, a stop rule such as `apm_ns`'s 1.15 timing cap, or a smoke failure) leaves (e) undefined. | `proposal_v3.md:125-126`, `:200-201` (apm stop rule), `debate/lint_v3.json` (`needs_user` apm_ns, dtx_gate). | At SUMMARY, evaluate (e) on the arms that exist, plus the `opus_plc` zero-loss control if it ran. With **no** placebo arm, cap the contrast at "directional". Print the placebo count next to every verdict. |

## 3. Resolved since the last round

| id | how (verified) |
|---|---|
| O2.1 | All four required changes are in. (1) The panel (B2b-B2e, B3a-B3e; B2d `epoch_10`; B2a flagged and excluded) is pinned in §3, and the directory names match `SLS_setup/exp/`. (2) DiD* / z re-centring; bar = min(μ − 2σ, panel min, μ − 0.4·G); re-centred CI / clip / half-B conditions. (3) Placebo arms. (4) Thresholds and falsifiers restated. Every panel statistic reproduces (computed). A leave-one-out application of (a)-(d) gives no "shown" for any of the nine panel runs on NS, VAD or PLC. The per-contrast power statement is honest: the PLC bar is unreachable, so `opus_plc` is capped at "directional". |
| O2.2 | Named asserts with printed values. `[no_gain]` is one-sided on a speech-like AM input. `[monotone]` has a 1 dB tolerance. `[no_hpf]` runs on 5.5 s of white noise, 512-point Welch, bins 1-3 vs 1-4 kHz. The band change against my suggestion is justified (computed with `scipy.signal`): an 80 Hz 2nd-order HPF reads 16.43 / 5.66 / 1.85 dB per bin, 5.00 dB band power, against 1.56 dB in my 62.5-187.5 Hz band; a 100 Hz HPF reads 7.15 vs 2.30 dB. The selftest runs in the implement step's `checks apm` before any smoke. The bindings source confirms that no HPF exists in the `NoiseSuppressor` path, so the assert is a guard, not an expected failure. |
| O2.3 | G6 label corrected: "gating family and zero fill seen; detector and white-noise CNG unseen". The VAD contrast is split by fill (75 / 61 vs 246, reproduced). The pre-registered near-in-distribution reading for `vad:zero` is in. |
| O2.4 | Measure pinned in frame counts: 201 × 320 samples, float64 energy, `percentile(e, 95)` − 30 dB, k ≤ 32 / 33-50 / ≥ 51. I reproduce 259 / 250 / 241 sources and the whole table, with 17 ties at k = 32 and 15 at k = 50. My v2 484 / 534 / 482 came from the float cut, as v3 says. The doubled n (each source counted twice) is stated. |

## 4. Citation checks

| url | ok | note |
|---|---|---|
| https://raw.githubusercontent.com/strands-labs/pywebrtc-audio/main/bindings/webrtc_audio_bindings.cpp | yes | `NoiseSuppressor` holds `std::unique_ptr<webrtc::NoiseSuppressor> ns_` and an `AudioBuffer`, with `config.target_level = static_cast<webrtc::NsConfig::SuppressionLevel>(level)`. The error text reads "level must be 0-3 (6/12/18/21 dB)". `process_buffer` runs Split → `Analyze` → `Process` → Merge. `HighPassFilter` is created only in `AudioProcessor` (`if (hp_enabled_ \|\| aec_enabled_)`) and `EchoCanceller`. So the NS-only path has no HPF. |
| https://github.com/strands-labs/pywebrtc-audio | yes | `NoiseSuppressor.process(audio)` processes "audio of any length" and "Internally splits into 10ms frames". HPF appears only as an `AudioProcessor` option ("also enabled automatically with AEC"). |
| https://github.com/wiseman/py-webrtcvad | yes | Quotes match: "A frame must be either 10, 20, or 30 ms in duration" and "3 is the most aggressive". Sample rates are 8 / 16 / 32 / 48 kHz. |
| https://arxiv.org/html/2607.17761 | yes | §4.4 quote matches: "The Mix strategy partially alleviates the degradation, yet still suffers noticeable performance drops under more severe distortions (e.g., AGC and VAD)". The order "AEC → NS → AGC → VAD" is stated for mainstream systems. Table 2 XLSR-AASIST: clean-only 0.23 / 22.20, Mix 0.52 / 17.39. v3's corrected "last front-end stage" wording is accurate. |
| https://arxiv.org/abs/2107.12018 | partly | UR21 ("UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021") augments with transmission / compression codecs and IRs. The abstract does not mention packet loss. v3 cites it without a specific claim, so this is fine as general background, not as packet-loss evidence. |
| https://arxiv.org/html/2504.12423v2 | yes | Verified in v1 / v2: PLR 0 / 1 / 5 / 10 / 20 % and the "EER remains unchanged … 0.003%" quote. Not re-fetched. |

## 5. Alternatives

- **A B2f replicate arm** (b4wave entry, every B4 flag off, same argv; ≈ 4.4 h) as a fourth, owner-optional
  arm. It is the only true same-config null for all three contrasts **and** for heldout F1 / WF1, which addresses
  O3.2 at the root. It is also the strongest placebo for (e) and a full-scale end-to-end check of the default-off
  path. The parent augmenter RNG is pid-seeded (`utils/rtc_augment.py:244`, `:349-352`), so it is a genuine
  retrain, not a bit-replay. The cost is one extra arm, so it is the owner's call at the gate. It is not required.

## 6. Per-lever stance

| lever | stance | note |
|---|---|---|
| `apm_ns` | **keep** | NS-only backend confirmed at the source level. Best-powered contrast (bar −0.22, 46 % of the typical gap). Selftest robust and correctly banded. Add the run-anomaly flag (O3.1), because its lower-p retry keys on "shown". |
| `dtx_gate` | **keep** | Criterion (ii) is the best-calibrated test in the plan: no null run passes it, and it is reproduced with the pinned cuts. The VAD contrast is honestly demoted to "directional", with the fill split pre-registered. Note that (ii) needs about a 15-clip differential shift (z +2.6 from the panel mean), so a modest real effect may read "inconclusive", not "shown". |
| `opus_plc` | **keep** | Weak local premise stated honestly; the mechanism is capped at "directional", and the arm stands as an outcome test of ADD-C with an attribution control. Its outcome escape from falsifier (1) is noisy (O3.2). The cost reduction (0.85 projected) and the codec-draw-order equivalence test are good. |
