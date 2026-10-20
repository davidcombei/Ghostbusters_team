# rdloop B4 - critique of proposal v1

> research-critic, CRITIQUE round 1 of 3, 2026-10-17. Read: `rdloop/B4/{brief.md,cycle.json,user_notes.md}`
> (no owner notes), `debate/proposal_v1.{md,json}`, `debate/lint_v1.json` (no errors; `needs_user`
> apm_ns → `webrtc_noise_gain`, dtx_gate → `webrtcvad`), `01_audit.md`, `02_sota.md`. Code: `SLS_setup/utils/rtc_augment.py`,
> `SLS_setup/b3wave/{augment.py,data.py,train.py,heldout_reserve.py}`, `SLS_setup/utils/{heldout_reserve.py,sim_ops.py}`,
> `scripts/build_dev_noisy_sim.py`, `scripts/rdloop/rdlib.py` (lint, `build_seed`), `scripts/exploop/exploop_lib.py`
> (log regexes), RUN `logs/train.log`, `exploop/B3/wave.json` `rules`.
> "(computed)" = read-only `python3 -I` joins of RUN `local_eval/<readout>/scores/sim_heldout_v1.txt` with
> `data/dev_noisy_sim/heldout_v1/meta.csv` (half B, bona), run in this session. Nothing was written except this file. No
> progress-split file was read.

## 1. Verdict

This is a sound and unusually well-prepared plan. The three CPU-side levers sit at disjoint points of the chain. Each
draws from a separate RNG stream. Defaults-off returns the B3 augmenter object itself. No lever touches the state_dict
or VRAM. The op names avoid the reserve (`b3wave/heldout_reserve.py:33-36`), and all three implementations are on the
allowed list (`docs/next_steps_plan.md:66-67`; roadmap l.108-109).

I re-derived the §1 DSP-slice table exactly from the score files. I also checked the codec draw order
(`utils/rtc_augment.py:186-193`), the promotion arithmetic (≈ 14 FA) and the B2f train-log anchors (`train.log` l.1 and
l.15). The cited sources say what the plan claims.

There are no blockers. Two majors stand in the way of agreement:
- **O1.1.** The chosen NS binding, `webrtc-noise-gain`, hard-wires the WebRTC high-pass filter on. So `apm_ns` is
  NS + HPF, not the "NS alone" the plan claims. The fallback binding has a standalone NS with no HPF.
- **O1.2.** The slice readout that carries every lever's mechanism claim compares against a control that is confounded
  with source:
  - the "no-DSP" bona slice is 124 of 128 online-source clips;
  - every NS / AGC / VAD slice is 100 % offline.

  Some predicted gains are also as large as the whole stage-attributable deficit, or larger. Some falsifier bands are
  inside the noise between the merge and final readouts of the same run.

Both fixes are cheap. The rest are minors. Verdict: **revise**.

## 2. Objections

| id | lever | severity | category | claim | evidence | required change |
|---|---|---|---|---|---|---|
| O1.1 | apm_ns | **major** | confound | `webrtc-noise-gain` always enables the APM high-pass filter. Only NS and AGC are optional. The lever is therefore NS + HPF on 30 % of clips: two components, contrary to "NS alone is one component". A low-band cut can carry class information (room rumble / hum in bona recordings vs clean TTS). With this backend no control can separate it. The plan already suspects this (it "prints the < 100 Hz energy change, in case…") but does not decide what to do. | Source `src/webrtc_noise_gain_cpp.cpp` (fetched): "self->audio_config.high_pass_filter.enabled = true;" is set unconditionally; NS / AGC only `if > 0`. Proposal `proposal_v1.md:189` ("NS alone is one component"), `:216`. PYWRTC README (fetched): standalone `NoiseSuppressor(sample_rate=16000, num_channels=1, level=1)`, level 0-3 = 6 / 12 / 18 / 21 dB; HPF is a separate `AudioProcessor` flag, default off; wheels for Python 3.10-3.14. | Make `pywebrtc-audio`'s standalone `NoiseSuppressor` the primary backend (NS only, no HPF). Its levels 0-3 map one-to-one to WNG 1-4 (kLow…kVeryHigh). Define `--aug_apm_ns_levels` backend-neutrally (e.g. 6 / 12 / 18 / 21 dB, or 0-3) and update the dependency import name at the gate. Make `check_apm` *assert* (not print) that the < 100 Hz band energy of a white-noise input at a high SNR changes by no more than the NS gain. If WNG stays the backend, rename the component "WebRTC capture path (HPF + NS)" and pre-register an HPF-only attribution control, as for `opus_plc`. |
| O1.2 | (all) | **major** | eval | The mechanism readout cannot attribute a gain to the lever, for three reasons. **(a) Source confound.** Heldout NS / AGC / VAD / band-limit run only on offline sources, so the "no-DSP" bona control (n = 128) is 124 online + 4 offline (computed). "Gain on `ns:afftdn` > gain on no-DSP" and "VAD gain" therefore compare offline against online populations. A generic offline-bona shift satisfies them. **(b) Overlap.** Slices are mostly multi-stage. `ns:afftdn` has only 8 exclusive bona clips; 42 / 74 co-occur with `bw`, 26 / 74 with `plc` and 23 / 74 with `dynaudnorm` (computed). So `opus_plc` alone moves a third of the `afftdn` slice. **(c) Sizes vs deficit and noise.** Against same-source controls, the stage-attributable bona deficit (computed, merge / final) is: `plc` online 65.2 vs 71.8 and offline 67.2 vs 70.1, about 6 clips in total (final about 6); VAD 63.0 vs 72.7 (offline), about 7 / 4 clips. The predicted `plc` gain of ≥ 7 of 130 and the VAD gain of ≥ 6 of 73 equal or exceed the whole stage-specific deficit, so meeting them needs a non-specific gain. Meanwhile merge vs final of the *same* run already differ by 4 discordant clips on `afftdn` (3 fixed / 1 broken) and on VAD (3 / 1). The falsifier bands "≤ 2 clips" (`afftdn`) and "< 3 clips" (VAD) sit inside that within-run noise. | Computed in this session: `offline with NS n=146 recall 66.4 / 67.1, without NS n=55 76.4 / 74.5`; `offline with VAD n=73 63.0 / 65.8, without n=128 72.7 / 71.1`; `online plc n=66 65.2 / 68.2, online no-plc n=124 71.8 / 71.8`; `offline plc n=64 67.2 / 65.6, no-plc n=137 70.1 / 70.8`; bona half-B src × DSP: offline+dsp 197, offline none 4, online+dsp 66, online none 124. `scripts/build_dev_noisy_sim.py:189` (`if row["src"] == "offline":` gates NS/AGC/VAD/BW), `:199` (PLC on all). Proposal `:238`, `:385`, `:584`, `:249`, `:397`. | In `scripts/b4_dsp_slices.py` and the signatures: **(1)** Use same-source controls. For `apm_ns`: offline bona with NS vs offline bona without NS (n = 146 / 55), plus `afftdn` vs `anlmdn`. For `dtx_gate`: offline with VAD vs offline without VAD (73 / 128). For `opus_plc`: within-src PLC vs no-PLC, with the online half as the main readout. Report each signature as a difference-in-differences in clips. **(2)** Print paired discordant counts per slice (fixed / broken vs `--ref`) with an exact sign / McNemar p, and print the B2f merge-vs-final discordance as the noise reference. **(3)** Restate the signature thresholds as a fraction of the stage-attributable deficit (e.g. ≥ 40 %), and the falsifiers as "DiD ≤ 0 or within the merge-vs-final discordance". **(4)** Also report exclusive-stage counts so overlap is visible. **(5)** `dtx_gate`: the saturated-core count (136 / 133) is not mechanism-specific. Add a stratification of saturated FA by the eval clip's non-speech fraction (webrtcvad on the local sim audio, analysis only) and predict that the drop concentrates in the high non-speech tertile. |
| O1.3 | opus_plc | minor | scope | G2 is listed as a target, but `sim_matched_v1` has no loss and the lever's matched prediction is "unchanged" (±3 points on qq / wechat). Nothing in the plan predicts a G2 gain. | `proposal_v1.md:44`, `:55`; json `gap_ids: ["G1","G2"]`; the matched prediction at `:586-587`. | Drop G2 from `gap_ids`, or state a matched-side prediction with a falsifier. |
| O1.4 | opus_plc | minor | leakage | The loss range is justified by the heldout realised loss ("mean 0.085, which equals the heldout mean 0.085"). That is an eval-set statistic used as a training design argument. The loss *pattern* is also the reserved op's family: `sim_ops.packet_loss` is Gilbert-Elliott with p_bg U(0.3, 0.7) (mean burst 1.4-3.3) and p_gb U(0.01, 0.08), and the lever is GE with burst U(1, 4) and loss U(2, 15) %. The range itself came from the SOTA card (ADD-C 0-20 %), so this is not leakage in fact, but the stated rationale invites it. | `proposal_v1.md:522-524`; `SLS_setup/utils/sim_ops.py:115-135` (l.122 the GE draws). | Justify the range from ADD-C (PLRs 0 / 1 / 5 / 10 / 20 %, fetched) or RTC measurements, not from heldout `meta.csv`. In the G6 labelling, call heldout `plc` "loss-pattern family seen (GE), codec + concealment unseen", so a slice gain is not read as transfer. |
| O1.5 | apm_ns | minor | confound | The realised effect of NS is likely class-asymmetric. Bona recordings carry a natural noise floor that NS removes, while many TTS spoofs have little to remove, so "processed" may shift toward bona-only. The plan measures class asymmetry for DTX (`dtx_balance`) but not for NS. The per-class telemetry only counts *application*, not the realised change. | `proposal_v1.md:150-153` (telemetry counts only), `:377-378` (dtx_balance exists for DTX only). | Add `checks apm_balance`, mirroring `dtx_balance`: on 200 + 200 training clips (with and without a noise stage) print the realised dB removed per class (mean and quartiles). Name a guard if the class means differ by more than a set margin. |
| O1.6 | '' | minor | evidence | "B2f never trained on any suppressor" is overstated. RTCFake's training split contains online clips transmitted through Zoom and QQ (P01-P02), which went through their NS / VAD / codec, though on clean input. The accurate statement is "never noisy speech through a suppressor, and no simulated DSP". | RTCFake html (fetched): training platforms P01-P02 = Zoom, QQ; online training clips 6,890 real / 30,235 fake. `proposal_v1.md:164`. | Rephrase. This also explains why NS on *clean* clips may be a near no-op on online-source training audio. |
| O1.7 | (all) | minor | budget | The cost argument quotes wall-clock times at load ~25 but not the CPU budget. The container's cgroup quota is 16 CPUs (`cpu.max 1600000 100000`) for 64 loader workers. At B2f's ≈ 41 clips/s that is ≤ ≈ 0.39 CPU-s per clip if the loader is saturated. Whether B2f is loader-bound is unknown, so the 1.02-1.05 factors are unverified. The smoke rate on a 758-clip subset is dominated by start-up. | Command output: `/sys/fs/cgroup/cpu.max` = `1600000 100000`; RUN `train.log` epochs ≈ 30.5 min for 75,785 clips; `proposal_v1.md:272-279`. | `checks timing` should report **CPU-seconds** (`time.process_time`, single worker) per clip for the full B2f path (load + loudness + stage + codec + featurize) and for each new stage. Print the projected epoch factor assuming the loader is saturated. |
| O1.8 | '' | minor | scope | §5 pre-registers a combined arm with all signature-positive levers on together. That is a multi-factor change against B2f and conflicts with the wave's one-factor-per-iteration rule. | `proposal_v1.md:676-678`. | State it as a recommendation to the owner / exp-designer (coupled group or a separate wave), not a pre-approved arm. |
| O1.9 | opus_plc | minor | confound | The in-process path adds a second component: a libopus decoder at 16 kHz, where B2f decodes with ffmpeg's native decoder at 48 kHz and resamples. The aligned SNR between the paths is only 17-18 dB, and the control arm runs only if the arm is positive. Libopus states created per clip in persistent workers must be destroyed. | `proposal_v1.md:487-499`, `:595-598`; `utils/rtc_augment.py:196` (ffmpeg decode, 16 kHz output). | Optional: decode in-process at 48 kHz and `resample_poly` to 16 kHz to mimic ffmpeg's path and shrink the component. Keep the zero-loss attribution check. Destroy the encoder / decoder in a `finally` (code-review item). |

## 3. Resolved since the last round

None (round 1).

## 4. Citation checks

| url | ok | note |
|---|---|---|
| https://github.com/rhasspy/webrtc-noise-gain | yes | API as claimed: `AudioProcessor(auto_gain_dbfs [0,31], noise_suppression_level [0,4])`, `Process10ms` → `.audio` / `.is_speech`, 16 kHz mono int16, "0 = disable". CHANGELOG 1.3.0: "Transition to ABI3 compatible wheel", so cp312 is covered. |
| https://raw.githubusercontent.com/OHF-Voice/webrtc-noise-gain/master/src/webrtc_noise_gain_cpp.cpp | **no (for the plan's claim)** | `high_pass_filter.enabled = true` is unconditional; NS levels 1-3 → kLow / kModerate / kHigh, else kVeryHigh. The lever is NS + HPF → O1.1. |
| https://github.com/strands-labs/pywebrtc-audio | yes | Standalone `NoiseSuppressor(sample_rate, num_channels, level 0-3 = 6/12/18/21 dB)`; HPF only as an `AudioProcessor` flag (default off); Python 3.10-3.14 wheels. |
| https://arxiv.org/html/2607.17761 | yes | "we adopt the standard implementations provided by WebRTC". Table 1 XLSR-AASIST 0.23 → … → 22.20. Table 2: clean-only 0.23 / 22.20, AFE-only 7.06 / 11.84, Mix 0.52 / 17.39. "The Mix strategy partially alleviates the degradation". The page does not state that Mix trains on the same chain as test (training noise MUSAN vs eval DNS), so "same chain" is the plan's inference. |
| https://arxiv.org/html/2309.11827 | yes | Abstract quote and Sil-mask definition match. WebRTC VAD with 10 ms frames. Table VI DF: SENet 22.95 → 22.14, LCNN 22.46 → 20.41, AASIST 16.92 → 16.06. On LA it helps SENet and hurts LCNN / AASIST ("mixed" is accurate). |
| https://arxiv.org/abs/2106.12914 | yes | "EER increases from 3.6% (with silence) to 15.5% (trimmed silence)." |
| https://arxiv.org/html/2504.12423v2 | yes | PLRs 0 / 1 / 5 / 10 / 20 %; "average degradation of 5.30% in EER", "2.33% (EER)", "The EER remains unchanged, with a slight improvement of 0.003% from C0 to C1". The PLC method is not named, and the gain is in-distribution (the plan says so). |
| https://arxiv.org/html/2604.23742v1 | yes | Fig. 1 caption "key distortion sources, including noise suppression, echo cancellation, codec encoding and decoding" (continues with packet loss); §5.4 quote matches. The training split includes Zoom / QQ online clips → O1.6. |
| https://github.com/daanzu/py-webrtcvad-wheels | yes | `import webrtcvad`; 2.0.12 added Python 3.12 / 3.13 builds; frames 10 / 20 / 30 ms; mode 0-3, 3 most aggressive. |

## 5. Alternatives

- **NS backend (I1).** Use `pywebrtc-audio`'s standalone `NoiseSuppressor` instead of `webrtc-noise-gain`: NS without
  the hard-wired HPF, the same four suppression levels, and py3.12 wheels. This keeps `apm_ns` a single component at no
  cost (O1.1).
- **DTX mechanism readout (I2).** Stratify the saturated noisy-bona FA by the eval clip's non-speech fraction, measured
  with webrtcvad on the local sim audio as read-only analysis. A "gated / non-speech ⇒ spoof" mechanism predicts the
  drop concentrates in the high non-speech tertile. A generic gain does not (O1.2 (5)).
- **Decoder-path parity (I3).** Decode in-process at 48 kHz and resample to 16 kHz, as the ffmpeg path does. This
  shrinks the "in-process vs ffmpeg" second component before any control arm is needed (O1.9).

## 6. Per-lever stance

- **apm_ns: modify.** The idea and the hook are right, the evidence is the strongest of the three (TFCL Mix), and the
  `afftdn` / offline-NS deficit is large (≈ 15 clips against offline bona without NS, computed). Switch to an NS-only
  backend (O1.1) and add `apm_balance` (O1.5).
- **dtx_gate: modify.** The design is careful: class telemetry, `dtx_balance`, op name off the reserve, and spoof-miss
  guards. Only the readout needs work. Use same-source controls, paired counts and a non-speech stratification for the
  saturated core, and set thresholds above the within-run noise (O1.2).
- **opus_plc: modify.** It is dependency-free, its equivalence is well tested (draw order, bit-identical fallback) and it
  has a pre-registered attribution control. Fix the readout (O1.2), the G2 claim (O1.3) and the rationale for the loss
  range (O1.4). Decoder parity (O1.9) is optional.
