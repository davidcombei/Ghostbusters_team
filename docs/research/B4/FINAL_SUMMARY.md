# rdloop B4 - final summary (b4wave)

> Written 2026-10-18 by the rdloop B4 SUMMARY step (research-auditor, attempt a0). Every number is copied from a
> file read in this session, and the file is named next to it. **(computed)** marks simple arithmetic on copied
> numbers, or counts I made with read-only `python3 -I` / `grep` over the named logs. Hypotheses are labelled
> **hypothesis**. No progress-split score was read.
>
> **Files read.** Cycle: `rdloop/B4/{cycle.json, brief.md, user_notes.md, approval.json, STATUS.md, state.json,
> events.jsonl, logs/supervisor.out, plan_agreed.md, 01_audit.md, 02_sota.md, scout.json}`;
> `debate/{ledger.json, critique_v3.md}`; `review/{code_review_r1.md, code_review_r2.md, code_review_r2.json}`;
> `impl/{progress.md, implement_core.json, implement_apm_ns.json, implement_dtx_gate.json, implement_opus_plc.json,
> fix_f1.json}`; `tests/{facts.json, gpu.json, compat.json, compat/*.log, regression/old_scores_compare.json,
> regression/R3_module_gpu_checks.log, smoke/S0_levers_off.log, smoke/S_apm_ns.log, smoke/S_dtx_gate.log,
> smoke/S_opus_plc.log, static/checks_cpu.log (tail), static/base_checks_all.log, static/exploop_test_*.log}`;
> `isolation/manifest.json`; `handoff/{wave_seed.json, dryrun_wave/wave.json, dryrun_wave/STATUS.md}`.
> Also `SLS_setup/b4wave/README.md` (section list, dose rule), line counts of `SLS_setup/b4wave/*.py` and
> `scripts/b4_dsp_slices.py`, B2f `local_eval/{merge_top5,epoch_8}/metrics.json` (`half_B` only), and
> `exploop/B3/{loop_state.json, events.jsonl}` (phase timestamps only).
>
> Owner note of 2026-10-17 13:48 (O3.1-O3.3, readout only) applies; it was implemented in
> `scripts/b4_dsp_slices.py` and the README verdict rules (`impl/progress.md` "Shared readout";
> `review/code_review_r1.md` §2 "shared").

## 1. Headline

The cycle built `SLS_setup/b4wave/`: three default-off, CPU-side, one-component augmentation levers that put
simulated **platform DSP** into training for the first time: WebRTC noise suppression (`apm_ns`), webrtcvad DTX
gating (`dtx_gate`) and in-process libopus packet loss with decoder PLC (`opus_plc`). It also built a calibrated
mechanism readout, `scripts/b4_dsp_slices.py`, with a null panel of 9 earlier runs, placebo arms and the owner's
O3.1-O3.3 rules. Every static, regression, smoke and compat check passes on code sha `d163f934…`
(`tests/facts.json`).

**Readiness: ready**, with one owner decision. As built, `dtx_gate` under-doses its target:
- smoke: 2 of 233 processed clips are ≥ 20 % gated, against 32.1 % for the held-out `vad_gate` (computed, §2);
- the critic's expectation: a null `dtx_gate` arm "will almost certainly read 'inconclusive (under-dosed)'"
  (`review/code_review_r2.md` §6).

## 2. Levers implemented

The base for every comparison is B2f: half-B WF1 **91.98 / 92.14** (merge / final), F1 clean 97.60 / 97.69, F1 noisy
89.57 / 89.76, heldout F1 87.92 / 88.27 (B2f `local_eval/{merge_top5,epoch_8}/metrics.json` `half_B`; also
`01_audit.md` §1).

| lever | flags (default → wave value) | gap | sources (verified) | smoke: time × vs levers-off · peak VRAM · projected 8-epoch h | smoke telemetry (computed from the smoke log) | notes |
|---|---|---|---|---|---|---|
| `apm_ns` | `--aug_apm_p` 0.0 → **0.3**; `--aug_apm_ns_levels` 6 12 18 21 (unchanged) | G1 | TFCL https://arxiv.org/html/2607.17761; RTCFake https://arxiv.org/html/2604.23742v1; WCR https://arxiv.org/abs/2603.14033; PYWRTC https://github.com/strands-labs/pywebrtc-audio | **1.126** · 9.94 GiB (S0 9.94) · **4.66 h** (`tests/facts.json`) | `wrtc_ns*` 218 of 758 codec calls = 0.288; bona 44/144 = 0.306, spoof 174/614 = 0.283; no `fail_*` key; 0 `failed (` lines (`tests/smoke/S_apm_ns.log`) | **Backend:** standalone NS only (no HPF / AGC / AEC). The critic confirmed this against the bindings source (`debate/critique_v3.md` §1, §4).<br>**Selftest PASS:** `no_hpf` diff −0.09…−0.92 dB; `aligned` lag 0 at every level (S_apm_ns.log).<br>**Deviation:** 96-sample latency compensation plus a 5th assert (`impl/progress.md`).<br>**Balance:** `apm_balance` max \|bona − spoof\| 0.30 dB, no WARN.<br>**CPU timing:** projection 1.012-1.049 (`implement_apm_ns.json`). |
| `dtx_gate` | `--aug_dtx_p` 0.0 → **0.3**; `--aug_dtx_fill` mix (unchanged) | G3, G1 | SIL https://arxiv.org/html/2309.11827, https://arxiv.org/abs/2309.11827; SSG https://arxiv.org/abs/2106.12914; TFCL https://arxiv.org/html/2607.17761; WVAD https://github.com/wiseman/py-webrtcvad | **0.928** · 9.94 GiB · **3.84 h** (`tests/facts.json`) | `wrtc_dtx_*` 233 / 758 = 0.307; bona 0.306, spoof 0.308. Gated-fraction bins `dtx_g0` 105, `g1_20` 126, `g20_40` 2, `g40p` 0, so the ≥ 20 % share is 2 / 233 = **0.0086** and the dose ratio vs held-out 0.321 is **≈ 0.027**, against a 0.25 rule. No `fail_*` key; 0 `failed (` lines (`tests/smoke/S_dtx_gate.log`) | **Under-dosed** (review F1.1, major). Cause: webrtcvad's own 100-330 ms overhang plus the drawn 40-200 ms hangover (`README.md` "Dose"; `review/code_review_r1.md` F1.1).<br>**Fixed in the readout only:** an under-dosed null reads "inconclusive (under-dosed)", never "refuted" (`fix_f1.json`).<br>**Selftest PASS:** 7 named asserts.<br>**Balance:** `dtx_balance` no WARN. |
| `opus_plc` | `--aug_opus_plc_p` 0.0 → **0.3**; `--aug_opus_loss` 0.01 0.20, `--aug_opus_burst` 1.0 4.0 (unchanged) | G1 (outcome test) | ADD-C https://arxiv.org/html/2504.12423v2; RTCFake https://arxiv.org/html/2604.23742v1; OPUSLIB https://github.com/orion-labs/opuslib (+ `opuslib/classes.py`); UR21 https://arxiv.org/abs/2107.12018 (background only: "abstract does not mention packet loss", `critique_v3.md` §4) | **0.913** · 9.94 GiB · **3.78 h** (`tests/facts.json`) | `opus_plc` 235 / 758 = 0.310; bona 41/144 = 0.285, spoof 194/614 = 0.316. Loss bins `plc_l0_5` 47, `plc_l5_10` 61, `plc_l10p` 127. No `fail_*` key; 0 `failed (` lines (`tests/smoke/S_opus_plc.log`) | **Encoder:** packets byte-identical to ffmpeg's Ogg packets, 21/21 runs, so only the receiver differs (`implement_opus_plc.json`).<br>**CPU timing:** projected loader factor 0.847-0.859 (`impl/progress.md`).<br>**Weak local premise:** the online PLC deficit is within one null SD and not dose-dependent, so the mechanism verdict is **capped at "directional"** (`plan_agreed.md` §1.1, §2.3).<br>**LOC:** about 383 code lines excluding checks and readout, near the 400 cap (`implement_opus_plc.json`). |

Shared:
- **Hook and RNG:** the hook order is stage → `wrtc_ns` → `wrtc_dtx` → codec. B4 draws come from a separate
  `rng4 = default_rng([seed, pid, 0xB4])`.
- **Levers off:** with every lever off, `build_rtc_augmenter_b4` returns the B3 object itself (`plan_agreed.md` §2.0;
  `review/code_review_r1.md` §3).
- **Model:** no state_dict change (`state.json` `state_dict_change: false`); 0 extra VRAM in smoke (peak 9.94 GiB =
  S0, `tests/gpu.json`).
- **Module size:** `checks.py` 2,393 lines and `scripts/b4_dsp_slices.py` 993 lines (`wc -l`).
- **Dependencies:** `pywebrtc_audio` and `webrtcvad` were already importable in the rtc-sdd env when core was
  built (`impl/progress.md` "Decisions (core)"); `needs_user` is empty (`state.json`).

## 3. Debate

Three rounds, outcome **agreed**, critic verdict "agree" in v3 (`tests/facts.json` `debate`). The owner approved
`apm_ns`, `dtx_gate` and `opus_plc` on 2026-10-17 13:49 (`approval.json`).

| round | levers | blocker / major / minor open | resolved | stances |
|---|---|---|---|---|
| v1 | apm_ns, dtx_gate, opus_plc | 0 / 2 / 7 | 0 | modify ×3 |
| v2 | same | 0 / 1 / 3 | 9 | apm_ns modify, dtx_gate keep, opus_plc modify |
| v3 | same | 0 / 0 / 3 | 13 | keep ×3 |

(`tests/facts.json`; `debate/ledger.json`)

**Objections that changed the plan** (`debate/ledger.json`):

| id | severity | claim (short) | resolution |
|---|---|---|---|
| O1.1 | major | The `webrtc-noise-gain` backend always enables the APM high-pass filter, so the lever would be NS + HPF (two components). | Switched to `pywebrtc-audio`'s standalone `NoiseSuppressor` (NS only). The selftest asserts no HPF in 31.25-93.75 Hz. |
| O1.2 | major | The slice readout could not attribute gains: offline vs online confound, overlapping slices, falsifier bands inside within-run noise. | Same-source contrasts (DiD), paired fixed / broken counts with a sign test, exclusive-stage counts, and the non-speech tertile stratification for `dtx_gate`. |
| O2.1 | major | The mechanism verdict was not calibrated on run-to-run variation. A null `opus_plc` would read "shown" in 3 of 10 earlier runs. | A null panel of 9 runs (B2b-B2e, B3a-B3e; failed B2a excluded), DiD* / z re-centring, bar = min(μ − 2σ, panel min, μ − 0.4·G), placebo arms (e), and a power statement per lever: `opus_plc` capped at "directional". Leave-one-out: no panel run reaches (a) (`critique_v3.md` §1). |
| O1.3-O1.9, O2.2-O2.4 | minor | G2 dropped from `opus_plc`; loss range justified from ADD-C, not heldout; `apm_balance` check; CPU-seconds timing; combined arm only as a recommendation; 48 kHz decode like ffmpeg; robust named apm selftest; VAD G6 label and fill split; tertiles pinned in frame counts. | All resolved by v3. |
| O3.1-O3.3 | minor (left open by the critic) | A failed run can fake "shown"; outcome escape thresholds sit inside cross-run noise (heldout-F1 panel SD 1.35); no rule for a missing placebo arm. | **Adopted by the owner at the gate** (`user_notes.md`) and implemented in the readout: run-anomaly flag → "confounded (run anomaly)"; escape needs heldout F1 Δ ≥ +0.5 on **both** readouts; "positive outcome" = the wave WF1 CI rule; with 0 placebo arms the verdict is capped at "directional", and the placebo count is printed. |

**Contested points:**
- **O2.2 band.** The architect replaced the critic's 62.5-187.5 Hz no-HPF band with 31.25-93.75 Hz, because an
  80 Hz HPF reads only 1.56 dB in the critic's band against 5.00 dB in the new one. The critic accepted
  (`critique_v3.md` §3).
- **`opus_plc` dose-response signature.** Declined and kept as a report-only table (`plan_agreed.md` §5).
- **Critic's optional B2f replicate arm.** Not taken up at the gate: only 3 levers were approved (`critique_v3.md`
  §5; `approval.json`).

**Code review:**
- **Round 1 (verdict fix):** F1.1 major, the `dtx_gate` dose; F1.2 minor, the apm selftest woke OpenBLAS (546.91 →
  0.53 CPU-s after the fix); F1.3 minor, silent worker-side lever failures.
- **Round 2: approve**, with one open minor, F2.1 (`review/code_review_r2.md`). F2.1: for `dtx_gate`, falsifier (1)'s
  outcome escape prints heldout F1 in one block and WF1 in another. The README uses WF1, and the owner is to confirm.

## 4. Dropped ideas

No approved lever was dropped at the gate, implement, review or smoke stage (`tests/facts.json` `dropped_levers:
null`).

| idea / lever | stage | reason |
|---|---|---|
| I4 AugMix-style single-view chain mixing | debate (not proposed) | Raises effective SNR by ≈ 8.7 dB; vision-only evidence (`plan_agreed.md` §4) |
| I5 echo suite (multi-tap self-echo, far-end + AEC3 residual) | debate (not proposed) | It is the `sim_echo_v1` recipe, which is not in WF1 (`plan_agreed.md` §4) |
| I6 DeepAugment with weight-perturbed DeepFilterNet | debate (not proposed) | Vision-only evidence; not installed; CPU risk at ~90 % of quota (`plan_agreed.md` §4) |
| AGC2 inside `apm_ns`; WebRTC HPF | debate | Second component; level treatment is "not a lever" (AUD §2); O1.1 |
| Pause insertion / attenuation fill in `dtx_gate`; libopus DTX as the gate | debate | Rhythm component (`agate` is reserved); libopus DTX fired on 7.1 % of bona vs 1.2 % of spoof frames (`plan_agreed.md` §4) |
| In-band FEC in `opus_plc`; `opuslib` dependency; 16 kHz in-process decode | debate | Second decoder component; system libopus via `ctypes` instead; 48 kHz decode + resample to match ffmpeg (O1.9) |
| G2 as an `opus_plc` target | debate | Matched has no packet loss, so nothing predicts a G2 gain (O1.3) |
| `opus_plc` dose-response signature | debate | No dose-dependent deficit to close; report-only (`plan_agreed.md` §5) |
| B2f replicate arm (critic alternative) | gate | Optional; not in the approved lever list (`critique_v3.md` §5; `approval.json`) |
| Two-view consistency variants (TFCL, PCL, JS) | scout | Two views double the epoch, over the 1.3× cap; a loss, not augmentation (`02_sota.md` §4) |
| Codec zoo (SILK, AMR, EVS, more Opus randomisation) | scout | Adjacent to the exhausted B3d (−2.20); G.7xx / GSM / speex reserved (`02_sota.md` §4) |
| Low-pass / band-limit / 8 kHz resampling | scout | `bandlimit` / `resample8k` are reserved held-out filters (`02_sota.md` §4) |
| SE as an inference pre-processor; neural-codec round trip; babble from own speech | scout | Rule status "ask" / "likely allowed"; label-noise risk (`02_sota.md` §4) |
| Noise-pool expansion (DEMAND / FSD50K / DNS) | scout | Data task, not a code lever; no ADD-specific evidence (`02_sota.md` §4) |
| SNR shape, chaining, RawBoost, SpecAugment | scout | Flags already exist in b3wave, so exploop can screen them; range widening is exhausted (`02_sota.md` §4) |
| Rhythm-perturbation copy-synthesis; Proteus chain mining; speech-mask training | scout | New spoofs forbidden; needs an offline policy loop; keeps the shortcut (`02_sota.md` §4) |

## 5. Regression and isolation

| check | result | file |
|---|---|---|
| Old-run scores reproduced | B2f `epoch_8` re-scored on 5 sets (`dev_online_clean` n 7465, `sim_matched_v1` 4000, `sim_heldout_v1` 4000, `probe_orig` 1000, `probe_noise_only` 1000): **max \|Δlogit\| 0.0, flips at 0.5 = 0, metric diffs 0.0** on every set | `tests/regression/old_scores_compare.json` |
| Base checks | `b3wave.checks model --ckpt <B2f ep8>` rc 0; `b3wave.checks optim` rc 0; `b3wave.checks all` 22/22 | `tests/gpu.json`; `tests/static/base_checks_all.log` |
| Module GPU checks | 5/5: `ckpt_keys_exact`; `logits_identical_levers_off` (4 dev clips, max \|dlogit\| 0.00e+00); `lever_forward_backward` for each of the 3 levers | `tests/regression/R3_module_gpu_checks.log` |
| Levers-off vs base trainer (micro-run, augmenter off in both) | b3wave and b4wave both: TrainLoss 0.7691, DevLoss 0.7197, LocalWF1 29.47; **rel diff 0.0000 / 0.0000** | `tests/facts.json` (`R4_*`) |
| Levers-off with the augmenter on | Covered at CPU level: `levers_off_returns_b3_object`, `forced_b4_bit_identical`, `train_items_bit_identical`, `levers_off_imports_nothing`; `checks cpu` **126/126** in 274 s | `review/code_review_r1.md` §3; `tests/static/checks_cpu.log` |
| Manifest | **138 protected files and 2107 run files unchanged**; 0 violations | `tests/facts.json`; `isolation/manifest.json` (created 2026-10-18 11:19:26) |
| Exploop | `test_lib` 13 tests OK, `test_view` 14 OK. Seed: 3 levers; `validate_levers` pass (one-factor validator); `smoke_run_contract` pass (5 smoke runs); `exploop init` rc 0; `exploop plan` rc 0 ("iteration 1 not started: next is DESIGN") | `tests/static/exploop_test_*.log`; `tests/compat.json`; `tests/compat/*.log` |
| Eval path | Merge + eval of the `apm_ns` smoke checkpoint works (1000 clips scored) | `tests/facts.json` (`E_eval_merge_path`) |

**History of the GPU phase.** The first GPU pass (2026-10-17 17:22-17:37) was BLOCKED:
- `R1_old_scores`: "local_eval exited 1". The cause is not recorded in the files I read.
- `R4_equivalence` FAIL with the augmenter on in both runs: train_loss rel diff 0.0451, dev_loss 0.3848.
- The first pass's training jobs reported a 21.6 GiB peak, the b3wave base micro-run included.

After the owner unblocked on 2026-10-18 11:19 (manifest re-baselined), the supervisor ran 12 checks, including a new
augmenter-off `R4_levers_off_micro`, and all passed (`events.jsonl`; `logs/supervisor.out` l.125-172).

**Hypothesis** for the augmenter-on mismatch: the parent augmenter RNG is pid-seeded (`critique_v3.md` §5 cites
`utils/rtc_augment.py:244, :349-352`), so two augmenter-on runs are different retrains, not replays. The same
config also gave different smoke numbers in the two passes: `S_dtx_gate` DevLoss 1.4203 on day 1 vs 0.9224 on
day 2 (`events.jsonl`). **The 1 % smoke losses and WF1 are therefore not evidence of any lever effect.**

**Cost**:
- **Claude spend:** **$84.67** (`tests/facts.json` `cost_usd` 84.6684). By agent: auditor $4.20 (audit; the summary
  step is still running), scout $3.94, architect $28.42, critic $17.09, engineer $31.02 (`STATUS.md`).
- **Wasted attempts:** three invalid `propose v1` attempts cost $12.74 (computed: $5.60 + $3.76 + $3.38 from
  `events.jsonl`) before the owner unblocked.
- **GPU time for tests:** 13 min in the passing pass (2026-10-18 11:21:34-11:34:39, `tests/gpu.json`) plus 15 min in
  the blocked pass (`events.jsonl`).
- **Wall clock:** 2026-10-17 10:57 → 2026-10-18 11:35, including two owner blocks.

## 6. Recommended wave

Run all three arms **one flag off B2f** with the base argv, `--batch_size 16` and the seed's `b4wave.train` entry
(`handoff/wave_seed.json`). Rules: band ±0.5; promote = CI low > 0 and clean drop ≤ 0.5
(`handoff/dryrun_wave/wave.json` `rules`). References: B2f 91.98 / 92.14 (`handoff/dryrun_wave/STATUS.md`).

| order | arm | why this position | projected h (train) | falsifier (`handoff/wave_seed.json`) |
|---|---|---|---|---|
| 1 | `apm_ns` `--aug_apm_p 0.3` | Largest gap (G1: heldout carries 4.23 / 4.11 of the missing WF1, `01_audit.md` §4). Best-powered contrast: the bar asks for 46 % of the typical NS gap. Strongest RTC-specific evidence: TFCL's Mix arm, 22.20 → 17.39 EER (`plan_agreed.md` §1.1, §2.1) | 4.66 | (1) NS DiD ≥ +0.32 and heldout F1 Δ < +0.5 on both readouts → Drop. (2) WF1 Δ < −0.5 with matched F1 down > 1.0 → Drop (one lower-p retry only if "shown" without the anomaly flag). (3) Full-set offline-NS spoof misses > 29 → Drop. (4) Shared under-fit: ep1 TrainAcc < 85.36 or ep8 TrainLoss > 0.0752, with heldout bona recall down |
| 2 | `opus_plc` `--aug_opus_plc_p 0.3` | Direct external evidence (ADD-C) and the cheapest arm. Its mechanism is capped at "directional", so it is decided on the WF1 CI rule and the guards | 3.78 | (1) Online PLC DiD ≥ −0.57 and heldout F1 Δ < +0.5 (both readouts) → Drop. (2) WF1 Δ < −0.5 with the under-fit signature → Drop. (3) If positive on WF1: run `--aug_opus_plc_p 0.3 --aug_opus_loss 0 0` before promotion; if that control gives ≥ half the gain, credit the decoder path, not PLC |
| 3 | `dtx_gate` `--aug_dtx_p 0.3` | Last, so the owner can decide the dose while arms 1-2 run (§7 Q1). As built it can show the mechanism but not refute it; it is still a placebo arm for the NS and PLC contrasts (rule (e)) | 3.84 | (1) VAD DiD ≥ −0.57 and drop_high − drop_low ≤ −10.2 and WF1 Δ < +0.5 (both readouts) → Drop **only at an adequate dose**; otherwise "owner decides a dose variant". (2) Clean F1 down > 0.5, or spoof-miss guards broken (noisy > 36 / 35, VAD > 25 / 22) → Drop. (3) Shared under-fit |

**Expected hours: ≈ 14 h.**
- Training: 12.28 h from the smoke projections (computed: 4.66 + 3.78 + 3.84).
- Eval and analysis: about 0.6 h per arm. In B3a, EVAL → `eval_done` took 22 min and ANALYZE 13 min
  (`exploop/B3/events.jsonl`, 2026-10-10 14:08:55 → 14:44:02).
- Optional: the `opus_plc` zero-loss control (~3.8 h, only if `opus_plc` is positive on WF1) and a B2f replicate arm
  (~4.1 h, `base_run_hours` 4.14, §7 Q2).

The smoke time factors come from 48-step runs and are noisy:
- `dtx_gate` reads 0.928 and `opus_plc` 0.913, while the CPU projections were ~1.00 and 0.85;
- `apm_ns` reads 1.126, while its CPU projection was 1.012-1.049.

All three are under the 1.3 cap, and every arm is under the 10 h cap.

**The wave spec must fix:**
- **Readout per arm:** the analyst runs `scripts/b4_dsp_slices.py` on each arm's merge and final readouts with
  `--ref` = B2f's same readout and `--placebo` = the other arms that exist, plus `--lever <id>` for the
  lever-specific reading.
- **Final verdicts:** the wave SUMMARY re-runs it with `--summary` and re-issues every verdict (provisional → final).
  The placebo count is printed. With 0 placebo arms the verdict is capped at "directional" (O3.3).
- **G6 labels:** after B4, heldout `ns:*`, `vad:*` and `plc` are "family seen, implementation unseen". A gain on
  `vad:zero` alone is near-in-distribution (`plan_agreed.md` §2.0).
- **One flag per arm against B2f.** The dry-run search space also allows the B3 consistency group coupled with
  `--batch_size`; it must not leak into these arms. Keep `--batch_size 16` (base argv).
- **Smoke.** The seed sets `smoke: off`, which is valid only for code sha `d163f934…`. Any code change, such as a
  dose variant, needs a new smoke.
- **Early check:** it expects `^b3wave: ` and `^b4wave: ` (both printed in every smoke log) and forbids
  `failed \(`, which now also catches the first lever failure per worker (F1.3).
- **Combined run.** A combined "platform chain" run comes only after the wave, as a recommendation, if the calibrated
  verdicts and the WF1 CI rule allow it (`plan_agreed.md` §4).

## 7. Risks and questions for the owner

**Risks**
1. **`dtx_gate` is structurally under-dosed.**
   - Smoke: 0.86 % of processed clips are ≥ 20 % gated, against 32.1 % on held-out. Dose ratio ≈ 0.027 against the
     0.25 rule (computed from `tests/smoke/S_dtx_gate.log`; held-out values from `README.md`, `fix_f1.json`).
   - Even forced mode 3 gates a mean 0.082 / 0.069 (bona / spoof) of a clip, against a held-out mean of 0.233
     (`impl/progress.md` dtx_balance).
   - **Analysis of the README definition:** the dose ratio is a share of *processed* clips, so raising
     `--aug_dtx_p` does not raise it. Mode and hangover are not flags, so a real dose variant is a code change.
2. **Promotion needs spill-over.** About 14 fewer noisy bona FA are needed to clear the ±0.5 band, while the
   half-B stage deficits are 10.8 (NS), 3.9 (VAD) and 2.4 + 3.3 (PLC) clips (`plan_agreed.md` §1.1). Heldout F1
   panel SD is 1.35, so single-seed outcome deltas below ~1 point are within cross-run spread (`critique_v3.md`
   O3.2).
3. **B3 lesson 4.** Every input-side change that made training harder in B3 cut the late fit (`01_audit.md` §6). The
   shared under-fit falsifier guards this, but the B3 runs had JS consistency on, so the risk on B2f without
   consistency is untested.
4. **Placebo rule (e) needs all three arms.** If an arm fails or is skipped, verdicts lose their tightest null; with 0
   placebo arms they are capped at "directional". There is no same-config B2f replicate (critic alternative).
5. **`opus_plc` codec-path differences.**
   - The zero-loss attribution check passes per profile (pooled), not per clip. Train_038736 reads qq 14.78 dB and
     wechat 12.96 dB, below the 15 dB floor; the pooled wechat margin is +1.22 dB.
   - ffmpeg 4.2's pre-skip leaves a ~69-sample offset between picked and unpicked clips.
   - The zero-loss control arm covers both (`implement_opus_plc.json`; `review/code_review_r1.md` §2).
6. **Container restarts.** A restart wipes system ffmpeg and libopus. With `opus_plc` on, start-up then raises
   instead of passing audio through silently (`implement_opus_plc.json`). Optional hardening: `conda install -n
   rtc-sdd -c conda-forge libopus` (`README.md` l.53).
7. **Isolation baseline.** Isolation was verified against a manifest the owner re-baselined at both unblocks
   (2026-10-17 12:17:41 and 2026-10-18 11:19:26, `events.jsonl`). The first-pass `R1_old_scores` failure cause is
   not on record. It passed with max \|Δlogit\| 0 on re-run.
8. **Base choice.** B2f is not the numerically best run: B3a scores 93.06 / 92.94 but costs 7.87 GPU-h
   (`01_audit.md` §1). A B4 gain on B2f does not transfer to B3a's recipe automatically, and stacking consistency
   breaks the 1.3× epoch cap.

**Questions**
1. `dtx_gate`: run it as built (expected "inconclusive (under-dosed)"; still a placebo arm), or commission a
   higher-dose **code** variant before it runs, e.g. hangover U(0, 200) ms or a lower-overhang / forced-aggressive
   mode? A higher `--aug_dtx_p` does not change the dose ratio.
2. Add the critic's optional B2f replicate arm (b4wave, all levers off, ~4.1 h) as the same-config null and an extra
   placebo?
3. F2.1: confirm the `dtx_gate` falsifier (1) escape metric. The plan text and the code use WF1; O3.2's wording uses
   heldout F1.
4. Accept the per-profile (pooled) reading of the `opus_plc` zero-loss attribution check?
5. Wave order: apm_ns → opus_plc → dtx_gate (this report) or the seed order apm_ns → dtx_gate → opus_plc?
6. Install conda-forge `libopus` before the wave to harden `opus_plc` against restarts?

## 8. Files

| what | path |
|---|---|
| module | `SLS_setup/b4wave/` (`augment.py`, `dsp_apm.py`, `dsp_dtx.py`, `opus_plc.py`, `data.py`, `train.py`, `checks.py`, `README.md`) |
| readout script | `scripts/b4_dsp_slices.py` |
| agreed plan | `rdloop/B4/plan_agreed.md` (sha `92c44cd85bba`), `plan_agreed.json` |
| audit | `rdloop/B4/01_audit.md`, `audit.json` |
| SOTA | `rdloop/B4/02_sota.md`, `scout.json` |
| debate | `rdloop/B4/debate/` (`proposal_v1-3`, `critique_v1-3`, `lint_v1-3`, `ledger.json`) |
| review | `rdloop/B4/review/code_review_r1.md`, `code_review_r2.md`, `diff_r1.patch`, `diff_r2.patch` |
| implementation notes | `rdloop/B4/impl/progress.md`, `impl/*.json` |
| tests | `rdloop/B4/tests/{facts.json, gpu.json, static.json, compat.json}`, `tests/regression/`, `tests/smoke/`, `tests/static/` |
| wave seed / dry run | `rdloop/B4/handoff/wave_seed.json`, `rdloop/B4/handoff/dryrun_wave/` |
| cost / events | `rdloop/B4/tests/facts.json` (`cost_usd`), `rdloop/B4/STATUS.md`, `rdloop/B4/events.jsonl` |
