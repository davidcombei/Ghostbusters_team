# rdloop B4 - critique of proposal v2

> research-critic, CRITIQUE round 2 of 3, 2026-10-17. Read: `rdloop/B4/{brief.md,cycle.json,user_notes.md}` (no
> owner notes), `debate/proposal_v2.{md,json}`, `debate/lint_v2.json` (no errors; `needs_user` apm_ns →
> `pywebrtc_audio`, dtx_gate → `webrtcvad`), `debate/ledger.json`, `debate/critique_v1.md`, `scout.json`.
> Code: `SLS_setup/utils/rtc_augment.py` (l.180-200 codec, l.355-368 `__call__`, `finalize`),
> `SLS_setup/b3wave/{augment.py,heldout_reserve.py}`, `SLS_setup/utils/sim_ops.py` (l.86-135),
> `scripts/build_dev_noisy_sim.py` (l.185-201), `SLS_setup/utils/data_utils.py` (`pad_audio`), RUN `logs/train.log`.
>
> "(computed)" means a read-only join, run in this session, of `SLS_setup/exp/<run>/local_eval/<readout>/scores/sim_{heldout,matched}_v1.{txt,logits}`
> with `data/dev_noisy_sim/{heldout,matched}_v1/meta.csv`, plus the dev source wavs for the tertiles. Scripts live
> in `/tmp/b4crit/`. Nothing in the repo was written except this file, and no progress-split file was read.
> The **null panel** is the 10 earlier runs with local_eval logits and no NS / VAD / PLC exposure:
> - B2a, B2b, B2c, B2e, B3a, B3b, B3c, B3d, B3e: `epoch_8`;
> - B2d: `epoch_10`.

## 1. Verdict

v2 answers every v1 objection, and answers it well. I verified the main changes:
- **NS backend.** The backend is NS-only. The pywebrtc-audio README (fetched) gives the standalone
  `NoiseSuppressor(sample_rate, num_channels, level)`, where `process()` takes int16 / float32 of any length and
  levels 0-3 are 6 / 12 / 18 / 21 dB. The HPF is listed only under `AudioProcessor`.
- **Reproduction.** I reproduced the same-source contrasts, the gaps and the within-run SEs. My B2f merge-vs-final
  NS DiD is +0.22 with a bootstrap half-width of about 0.39, and B2e-vs-B2f VAD / PLC are −0.74 / −0.73 (the
  proposal reports the opposite sign convention, B2f − B2e).
- **Train-log anchors.** RUN `train.log` l.1 has `TrainAcc=87.36%` and l.15 has `TrainLoss=0.050114`.
- **Citations.** ADD-C's PLR grid is correct.

The levers, hooks, RNG isolation and default-off path were sound in v1 and are unchanged.

One new **major** remains, and it concerns the readout v2 built at my request. The pre-registered verdict
("shown" = logit DiD ≤ −0.4·G with a clip-bootstrap CI excluding 0, plus the clip DiD and the half-B signs) is
calibrated only against *within-run* noise. I applied the exact rule to the 10 earlier runs that never saw NS,
VAD or PLC:
- `opus_plc`'s primary contrast reads **"mechanism shown" for 3 of 10**, including the one-factor B2e;
- `dtx_gate`'s criterion (i) reads "shown" for 3 of 10;
- `apm_ns`'s contrast reads **"refuted" for 7 of 10**.

The reason is that B2f is an outlier on these slices. Excluding the failed B2a, the null-panel mean DiD is
−0.57 on VAD and −0.57 on PLC, already past the −0.42 / −0.38 "shown" thresholds, and +0.32 on NS. Any retrain
regresses toward the panel. The 40 %-of-gap rule I proposed in v1 measures the distance from B2f, not the lever
effect. The fix is cheap: calibrate against the null panel and the other B4 arms (O2.1).

`dtx_gate`'s tertile criterion (ii) holds up against the null, which is reassuring. Four minors follow.
Verdict: **revise** (one revision left; O2.1 is a script and wording change).

## 2. Objections

| id | lever | severity | category | claim | evidence | required change |
|---|---|---|---|---|---|---|
| O2.1 | (all) | **major** | eval | **The mechanism verdict is not calibrated against run-to-run variation, and B2f is an outlier on the contrasts.** The threshold (−0.4·G = −0.34 / −0.42 / −0.38) and the bootstrap CI (resampling clips) cover only within-run noise. Unrelated retrains move the same-source contrasts by 0.3-1.1 logit, in a direction set by B2f's position, not by any lever. Applying the proposal's full "shown" rule (§1.1 l.138-143) to the 10-run null panel gives:<br>• NS: 1 shown (the failed B2a) and **7 refuted**;<br>• VAD (i): **3 shown** (B2e, B3b, B3e);<br>• PLC online: **3 shown** (B2c, B2d, B2e) and 1 refuted.<br>Consequences:<br>• a null `opus_plc` arm has about a 1-in-3 chance of "mechanism shown";<br>• a real `apm_ns` effect at the −0.34 threshold would on average read about 0.0 (inconclusive or refuted);<br>• the verdict feeds the §5 combined-run recommendation and the `apm_ns` lower-p retry, so it misleads the wave's next step.<br>The B2e row the proposal prints (0.73-0.81) already hinted at this (§1.1 point 2) but did not change the rule. | Computed, null panel vs B2f `epoch_8`, full set, clipped-logit DiD (B2a, B2b, B2c, B2d, B2e, B3a, B3b, B3c, B3d, B3e):<br>• **NS** (276 / 106): −1.42, −0.02, −0.03, +0.39, +0.36, +0.49, +0.41, +0.62, +0.68, +0.01. Excluding B2a: mean **+0.32**, SD 0.27.<br>• **VAD** (136 / 246): +0.67, −0.15, −0.17, −0.68, −0.74, −0.51, −0.70, −0.67, −0.65, −0.81. Excluding B2a: mean **−0.57**, SD 0.24.<br>• **PLC online** (114 / 254): −0.79, −0.42, −1.08, −1.11, −0.73, −0.54, −0.09, −0.33, −0.88, +0.03. Excluding B2a: mean **−0.57**, SD 0.41.<br>Full rule (CI, clip DiD ≥ 0.4·D, half-B sign on both readouts): PLC B2c −1.08 [−1.74, −0.47], clip +9.3, half B −1.29 / −1.19 → shown; B2d −1.11 [−1.91, −0.36] → shown; B2e −0.73 [−1.47, −0.05] → shown.<br>B2f merge-vs-final: NS +0.22, VAD −0.05, PLC +0.08 (CI half-width about 0.38), i.e. within-run noise is much smaller than cross-run.<br>`proposal_v2.md:138-143` (rule), `:112-116` (only B2e as cross-run reference), `:725-731` (§5 uses "mechanism shown"). | In `scripts/b4_dsp_slices.py` and the signatures:<br>(1) Add `--null_runs` with the panel above at their final readouts, B2a flagged or excluded, and print each contrast's null mean / SD / min.<br>(2) Re-centre: DiD* = DiD − null mean. Restate "shown" as DiD* ≤ −0.4·G **and** DiD below the panel minimum (or ≤ null mean − 2·SD), plus the existing clip and half-B conditions, re-centred the same way. Restate "refuted" as DiD* ≥ 0, not DiD ≥ 0.<br>(3) Add a **placebo-arm** check: each B4 arm is one flag off B2f with the same code and seed, so it is the tightest null for the *other* levers' contrasts. Pre-register that a lever's own-contrast DiD must be more negative than the other B4 arms' DiD on that contrast.<br>(4) Restate the per-lever thresholds (−0.34 / −0.42 / −0.38) and falsifier (1) of `apm_ns` and `opus_plc` on DiD*. `dtx_gate` (ii) can stay as is (see §5 of this report). |
| O2.2 | apm_ns | minor | feasibility | Two start-up selftest asserts can fail for legitimate NS behaviour and stop the arm at start-up.<br>(a) **Tone check.** "A −30 dBFS 1 kHz tone at 6 dB within ±3 dB" uses a stationary tone. A quantile noise estimator learns a stationary tone as noise, and the clip is *primed* on its own first 0.5 s, so attenuation up to the level's 6 dB floor is expected. The check's purpose ("no gain stage") is one-sided.<br>(b) **No-HPF check.** The "< 100 Hz within 3 dB of 1-4 kHz" check rests on two FFT bins (DC and 62.5 Hz at 16 kHz / 256-point), so it is noisy on a short excerpt. | `proposal_v2.md:338-343` (asserts), `:349` (selftest at start-up when on), `:333` (priming). README (fetched): `process()` "Internally splits into 10ms frames". | Make the gain check one-sided on a speech-like or amplitude-modulated input: output RMS ≤ input RMS + 0.5 dB at every level. Run the no-HPF check on ≥ 5 s of white noise after priming, averaged over frames, with the 62.5-187.5 Hz band rather than < 100 Hz (or exclude DC). A selftest failure must name the failing assert. |
| O2.3 | dtx_gate | minor | leakage | The G6 label misstates the heldout VAD stage. The reserved `sim_ops.vad_gate` fills gated frames 50 / 50 with **exact zeros** or white noise at 0.5 × floor. The label says "implementation unseen (energy VAD, white-noise fill)". `dtx_gate`'s `zero` fill, and the zero half of its `mix`, is therefore the *same* fill as about half of the heldout VAD slice: on the full set, offline bona VAD fills are zero 75 / cng 61. Only the detector (energy, p10 + margin, hangover) and the CNG colour are unseen. This is not leakage (the reserve is by op name, `b3wave/heldout_reserve.py:34`, and webrtcvad is allowed), but the VAD contrast mixes a near-seen and a less-seen half. | `SLS_setup/utils/sim_ops.py:103` (`fill = "cng" if rng.random() < 0.5 else "zero"`), `:110` (`out[gated] = 0.0`). Computed: heldout VAD fill counts zero 336 / cng 293 (all clips); offline bona full set zero 75 / cng 61. `proposal_v2.md:276`. | Correct the label: "gating family and zero fill seen; detector (energy VAD) and white-noise CNG unseen". Have `b4_dsp_slices.py` print the VAD contrast split by heldout fill (zero / cng), and read a gain confined to `vad:zero` as near-in-distribution. |
| O2.4 | dtx_gate | minor | eval | The tertile cut points as written do not reproduce the §1.1 table. The non-speech fraction is quantised to k / 201 frames (64,600 / 320), so many clips tie at the cuts. With the stated rule (20 ms frames, < p95 − 30 dB, first 64,600 samples, tiled if short) and cuts 0.159 / 0.249, I get n = 484 / 534 / 482 (matched + heldout), where the proposal has 518 / 500 / 482. The tertile quantiles still come out at exactly 0.159 / 0.249, so the difference is the tie rule at 32 / 201 = 0.1592. The low / mid saturated counts move accordingly: mine are 58 / 78 and 61 / 81 against 63 / 73 and 66 / 76; the high tertile, 109 / 107, matches. Matched and heldout also share the same 750 bona sources, so the effective n is 750, not 1,500. | Computed (rtc-sdd env, `soundfile`): q33 / q67 = 0.159 / 0.249; tertile n 484 / 534 / 482 (tiled) and 484 / 536 / 480 (untiled). `data/dev_noisy_sim/{heldout,matched}_v1/meta.csv`: 750 bona `source_utt` each, overlap 750. `proposal_v2.md:146-157`. | Pin the measure in the script: cuts as frame counts (e.g. low ≤ 32 / 201, high > 50 / 201) or as exact floats with the tie rule, the energy definition (mean square, dB, 1e-12 floor) and the short-clip tiling. `checks dsp_slices` must reproduce the table with them. State that n counts each source twice. |

## 3. Resolved since the last round

| id | how |
|---|---|
| O1.1 | The backend is the `pywebrtc_audio` standalone `NoiseSuppressor`, which has no HPF / AGC / AEC (README fetched: HPF is an `AudioProcessor` option only). The levels flag is in dB and the no-HPF assert was added. Its robustness is refined in O2.2 (minor). |
| O1.2 | All five parts were done: same-source contrasts (reproduced), paired fixed / broken counts with a sign test, a within-run reference, 40 % thresholds, exclusive-stage counts, and the tertile readout. The cross-run calibration gap is new evidence, raised as O2.1. It corrects my own v1 suggestion ("within the merge-vs-final discordance"), so it does not move the goalposts. |
| O1.3 | `gap_ids` of `opus_plc` = `["G1"]` (json), and matched qq / wechat are now a guard. |
| O1.4 | The range is ADD-C's 1-20 % (fetched: "Five PLRs, 0%, 1%, 5%, 10%, and 20%"). The burst is stated as a design choice, and the G6 label marks the loss pattern as seen. |
| O1.5 | `checks apm_balance` was added with WARN thresholds, plus NS spoof-miss guards in both directions. |
| O1.6 | Rephrased (§0, §2.1). |
| O1.7 | CPU-s per clip is measured (0.350), the quota is accounted for, and `checks timing` gets a saturated-loader factor and a 1.15 stop rule for NS. |
| O1.8 | §5 is a recommendation, not an arm. |
| O1.9 | 48 kHz decode + `resample_poly` (measured closer to ffmpeg on 4 / 7 profiles), destroy in `finally`, and a tighter zero-loss check. |

## 4. Citation checks

| url | ok | note |
|---|---|---|
| https://github.com/strands-labs/pywebrtc-audio | yes | `pip install pywebrtc-audio`; `from pywebrtc_audio import …`; `NoiseSuppressor(sample_rate=16000, num_channels=1, level=1)`; `process(audio)` takes int16 / float32 of any length and splits it into 10 ms frames internally; levels 0-3 = 6 / 12 / 18 / 21 dB; HPF only as an `AudioProcessor` option; Python 3.10-3.14 wheels, Linux x86_64. The import name in the lint (`pywebrtc_audio`) is correct. |
| https://arxiv.org/html/2504.12423v2 | yes | "Five PLRs, 0%, 1%, 5%, 10%, and 20%"; "The EER remains unchanged, with a slight improvement of 0.003% from C0 to C1" (Augmented-trained models). No PLC method and no burst model is named, consistent with the plan calling GE a design choice. |
| https://arxiv.org/abs/2603.14033 | yes | WCR, "What Counts as Real? …": the abstract contains "detectors can often identify that speech has been processed". |
| https://arxiv.org/html/2607.17761 | partly | WebRTC implementations are used, in the order AEC → NS → AGC → VAD. Table 2 XLSR-AASIST: clean-only 0.23 / 22.20, Mix 0.52 / 17.39. The proposal calls VAD "the most damaging front-end stage" (`proposal_v2.md:453`), but Table 1's columns are cumulative, so the VAD column is the full chain. The page does not isolate VAD's own damage, and Mix's training data is not explicitly defined. These caveats do not change the levers. |

## 5. Alternatives

- **Null-panel and placebo-arm calibration (fix for O2.1).** The evidence is already on disk: 10 runs with heldout
  logits. The B4 wave adds 3 one-flag arms that are mutual placebos. A lever's mechanism is "shown" only if:
  - its own-contrast DiD beats the panel (re-centred threshold plus below the panel minimum), **and**
  - it beats the other two B4 arms on the same contrast.

  This costs one argument and about 30 lines in `b4_dsp_slices.py`, and no GPU.
- **Keep `dtx_gate`'s tertile criterion (ii) as the anchor of its verdict.** Against the null panel, drop_high −
  drop_low is:

  | run | B2a | B2b | B2c | B2d | B2e | B3a | B3b | B3c | B3d | B3e |
  |---|---|---|---|---|---|---|---|---|---|---|
  | full set | +14 | −13 | −10 | −21 | −4 | −13 | −12 | −9 | −11 | −1 |
  | half B | −9 | −17 | −12 | −15 | −13 | −14 | −13 | −13 | −15 | −6 |

  Only the failed B2a passes "≥ 5", and no run passes "half B ≥ 0" (computed with my tertile definition, O2.4).
  It is the best-calibrated mechanism test in the plan. A similar *non-B2f-centred* signature would strengthen the
  other two levers. For example, for `opus_plc`, compare heldout bona with `plc` loss ≥ 0.1 against loss < 0.05
  within the PLC slice (a dose-response reading needs no cross-run reference).

## 6. Per-lever stance

- **apm_ns: modify (readout).** The lever is sound now: NS only, a balance check, guards in both directions. Its
  falsifier (1) "NS DiD ≥ 0 ⇒ refuted" fires for 7 / 10 unrelated runs, so restate it on the re-centred DiD (O2.1).
  Make the selftest asserts robust (O2.2).
- **dtx_gate: keep.** Criterion (ii) holds up against the null panel. Fix the G6 label and split the VAD readout by
  fill (O2.3), and pin the tertile definition (O2.4). Criterion (i) should be re-centred like the others (O2.1).
- **opus_plc: modify (readout).** Implementation, cost (0.85 projected) and attribution control are good. Its primary
  contrast is the most exposed to O2.1: 3 / 10 null runs read "shown", including the one-factor B2e. Without
  re-centring or a placebo arm, "mechanism shown" is not evidence for this lever.
