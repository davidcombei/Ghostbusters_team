# rdloop B4: code review r1

> research-critic, CODE REVIEW round 1, 2026-10-17. Scope: `SLS_setup/b4wave/` (whole module vs `b3wave`) and
> `scripts/b4_dsp_slices.py`, through `rdloop/B4/review/diff_r1.patch`. Plan: `rdloop/B4/plan_agreed.md` / `.json`
> (approved: `apm_ns`, `dtx_gate`, `opus_plc`), plus the owner notes of 2026-10-17 13:48 (O3.1-O3.3, readout only).
>
> **Read:**
> - the full module: `augment.py`, `train.py` (diff vs b3wave), `data.py`, `dsp_apm.py`, `dsp_dtx.py`, `opus_plc.py`,
>   `__init__.py`, `README.md`;
> - `checks.py`: the generic groups, `_equiv_*`, `check_apm`, `check_codec_draw_order`, timing, gpu, overfit, main;
> - all of `scripts/b4_dsp_slices.py`;
> - the base it hooks: `utils/rtc_augment.py` l.150-390, `b3wave/augment.py`, `b3wave/data.py`;
> - contract code: `scripts/exploop/{supervisor,exploop_lib,view}.py`, `scripts/rdloop/rdlib.py` (`build_seed`);
> - logs: `rdloop/B4/impl/progress.md`, `rdloop/B4/tests/static.json`, `rdloop/B4/tests/static/*.log`.
>
> **Ran (CPU only, nothing written):**
> - a timing probe of `dsp_apm.measure()` (`python -B`, `CUDA_VISIBLE_DEVICES=`);
> - two read-only `python3 -I` joins of `data/dev_noisy_sim/heldout_v1/meta.csv`.
>
> No progress-split file was read.

## 1. Verdict

The implementation is careful, and it conforms to the plan on all three levers and the shared readout:
- **Default-off path:** with every flag off it is the B3 object itself, and it adds no RNG draws.
- **RNG isolation:** every lever draws from `rng4` only. The codec-slot lever consumes the parent stream exactly as
  `codecs_augm` does: 1,000 start states and 14 real ffmpeg calls are bit-identical (`checks_cpu.log`, codec_draw_order).
- **Tests:** the equivalence tests compare against `build_rtc_augmenter_b3` outputs, so they are not tautological.
  `checks cpu` passes 117/117 (`rdloop/B4/tests/static/checks_cpu.log:278`), and `b3wave.checks all` passes 22/22.
- **Readout:** the O3.1-O3.3 amendments are implemented as the owner adopted them.

**One major finding.** The `dtx_gate` arm, as built, gates far less audio than the reserved held-out `vad_gate` it
is meant to neutralise:
- training: 0.5-2.5 % of processed clips have ≥ 20 % of their samples gated;
- held-out `vad_gate`: 32.1 % of clips.

So the pre-registered falsifier (1) "refuted → Drop" would read a null arm as "gating does not drive the core" when
the arm simply under-dosed it. The required fix is readout-only: print the realised dose and pre-register an
"under-dosed" reading. Any dose change to training stays the owner's call.

Two minor findings concern a start-up CPU spike in the `apm_ns` selftest and silent lever failures.

**Verdict: fix** (F1.1).

## 2. Plan conformance

| lever | ok | note |
|---|---|---|
| `apm_ns` | yes | **Flags:** `--aug_apm_p` 0.0 and `--aug_apm_ns_levels` [6, 12, 18, 21] (argparse choices plus `validate_b4_args`); exact plan header; op `wrtc_ns`; lazy `pywebrtc_audio`; fresh `NoiseSuppressor` per clip, primed on min(0.5 s, len).<br>**Deviation, justified:** 96-sample latency compensation plus a fifth assert `[aligned]` (`dsp_apm.py:19-24, 87-88`). Without it every processed clip would start with 6 ms of the priming tail.<br>**Selftest:** the 4 named asserts plus `aligned`, with values printed.<br>**`apm_balance`:** max \|bona − spoof\| 0.30 dB, no WARN (`checks_cpu.log:112-131`).<br>**Timing:** factor 0.996 (`:238`). |
| `dtx_gate` | yes (code) | **Matches the plan:** webrtcvad mode / frame / hangover U(40, 200) drawn from `rng4`; zero / LPC-CNG / mix fill; 5 ms ramps; exact header; op `wrtc_dtx`.<br>**Extra asserts:** `speech_kept` and `cng_colour`.<br>**Balance:** no WARN.<br>**Timing:** 0.985.<br>**Problem:** the dose is far below the eval op's (F1.1). |
| `opus_plc` | yes | **Encoder:** packets byte-identical to ffmpeg's Ogg packets (21/21).<br>**Path:** GE mask with stationary start; 48 kHz decode, 3 × look-ahead trim, `resample_poly`; encoder and decoder destroyed in `finally`; fallback is ffmpeg on the same pre-drawn profile; init raises with random profiles or with the codec off.<br>**Timing:** 0.841 (`:242`); 0.1086 vs 0.2495 CPU-s per call (`:245`).<br>**Accepted deviation:** the zero-loss attribution bar is evaluated per profile on pooled energies, not per clip. Per clip, Train_038736 reads qq 14.78 and wechat 12.96 dB, below 15. The pooled wechat margin is +1.22 dB (`:223-225`). This is a defensible reading of "on every profile", and the README states it (l.174-179). The in-process path still sits 9-12 dB closer to ffmpeg than ffmpeg sits to the input. Falsifier (3), the zero-loss control arm, is the safeguard.<br>**LOC:** above the 280 estimate (311 code lines in `opus_plc.py`), mostly selftest and checks. |
| shared | yes | **Registry and hooks:** `RTCAugmenterB4.__call__` mirrors the parent line for line, with `_post_dsp` and the `_codec` dispatch; `rng4 = default_rng([seed, pid, 0xB4])` is re-created on a pid change.<br>**Telemetry:** per-class line and `metrics.jsonl["b4"]` appear only when a lever is on.<br>**`config.yaml`:** `b4wave_version`, `module_sha256`, `b4_levers`.<br>**Readout:** pinned panel, B2a excluded; DiD*, z, bar; (a)-(e); provisional vs `--summary`; O3.1 anomaly flag → "confounded (run anomaly)"; O3.2 panel SD, clearing counts and both-readout escape; O3.3 placebo count and the 0-placebo cap.<br>**Kept deviation:** the `b3wave:` header line stays and `b4wave:` is added (the plan said "replaces"). This is harmless: base header lines stay a subset, and the exploop early check needs `^b4wave: ` (`rdlib.py:833`). |

## 3. Checklist

- **Default-off equivalence.**
  - `build_rtc_augmenter_b4` returns `build_rtc_augmenter_b3(args)` when no lever is on (`augment.py:382-383`).
  - The trainer adds no RNG draws: `module_sha256`, `validate_b4_args` and `active_levers` are deterministic. `track_by_class=False` keeps the b3wave loop, and `metrics.jsonl` stays key-for-key (`train.py` diff).
  - The model is untouched: `checks gpu` asserts exact checkpoint keys and identical logits.
  - Tests:
    - `levers_off_returns_b3_object`;
    - `forced_b4_bit_identical` (fixed and random profiles, with and without the codec);
    - `levers_off_imports_nothing`: a fresh process with no lever modules and no libopus mapped;
    - `train_items_bit_identical`: real clips;
    - `parser_reproduces_b3wave_on_base_argv`.
  - Item 3 is tested per lever with identity / ffmpeg stubs **and** the real backends. Parent `stage:*` and `codec:*` counts equal B3's, and the `rng4` draws are replayed draw by draw.
- **Leakage.**
  - `assert_training_ops` runs for `wrtc_ns`, `wrtc_dtx`, `opus_plc` and `libopus`. The reserved `vad_gate`, `packet_loss`, `afftdn`, `anlmdn` and `g722` still raise (`checks reserve_names`).
  - No `utils/sim_ops`, eval pool, dev, half B or progress path is reachable from training code (grep of `augment.py`, `dsp_*.py`, `opus_plc.py`, `data.py`, `train.py`).
  - The CNG white fallback only fires on digital-silence content, at −70 dBFS, so the "white CNG colour unseen" G6 label still holds.
- **Isolation.**
  - Imports go one way (b4wave → b3wave / utils).
  - The manifest check passes (138 protected files).
  - `scripts/__pycache__/b4_dsp_slices.cpython-312.pyc` (15:35) comes from the supervisor's own `py_compile` static job, not from the module.
- **Correctness / numerics.**
  - **GE mask:** p_gb = L·p_bg/(1−L) gives stationary loss L; measured 0.0993 at L 0.1 and 0.2009 at L 0.2.
  - **LPC:** solved by `solve_toeplitz` on a masked autocorrelation (positive semi-definite, so stable). The `np.roots` check is a guard.
  - **Hangover:** a forward dilation at sample level.
  - **Opus framing:** the frame count covers len + look-ahead, so the trimmed output is ≥ len.
  - **Length:** `finalize` keeps it.
  - **Class split:** `add_class_meta` uses label 1 = bonafide (`utils/data_utils.py:18-21`).
  - **Unchanged:** dtypes, EMA, merge and `--resume` paths are untouched; `b4_levers` is recomputed on resume.
- **Exploop contract.**
  - `Epoch N/M TrainLoss=` and `Epoch N LocalWF1=` are unchanged. The new `Epoch N B4 aug-by-class` line matches neither regex (`checks trainer`), and exploop parses no B3 / B4 telemetry.
  - `Experiment saved to:`, the checkpoint names, the run-dir naming and `config.yaml` `command` are unchanged.
  - Every flag is a literal `b4.add_argument`, so it is grep-able by `trainer.flag_sources`.
  - The headers match the plan's `expect_regex`.
- **Performance.**
  - Per-clip Python loops are per 10-30 ms frame or per packet; the measured factors are all ≤ 1.0 (`checks_cpu.log:236-245`).
  - The `apm_ns` start-up selftest wakes OpenBLAS (F1.2).
- **Tests.**
  - `cpu`, `gpu --ckpt` (forward and backward per lever on) and `overfit --lever` (SKIP: no params added) exist.
  - The forked-worker path (64 workers, ctypes / pywebrtc after fork) is only covered by the smoke run. That is acceptable.

## 4. Findings

| id | lever | severity | category | file:line | issue | fix |
|---|---|---|---|---|---|---|
| F1.1 | dtx_gate | **major** | plan | `SLS_setup/b4wave/dsp_dtx.py:84` (hangover on top of webrtcvad's own overhang); `scripts/b4_dsp_slices.py:626` (lever "refuted") | See "F1.1 detail" below. | See "F1.1 detail" below. |
| F1.2 | apm_ns | minor | perf | `SLS_setup/b4wave/dsp_apm.py:142` | **The start-up selftest wakes numpy's multi-threaded OpenBLAS.** `_lag` runs 401 `np.dot` calls on ~90 k-sample vectors at each level, and the spinning BLAS threads are billed to the process.<br>• Static run: apm selftest CPU **546.91 s** vs dtx 1.12 s and opus 0.36 s (`checks_cpu.log:100, 157, 205`).<br>• My run (8 CPUs): `measure()` 1.3 s wall / 4.8 CPU-s; with `OPENBLAS_NUM_THREADS=1`, 1.3 / 1.3. Four `wrtc_ns` calls right after cost 0.39 vs 0.04 CPU-s (threads still spinning).<br>This happens at start-up in the main process only, so it is minor. `dsp_dtx.lpc` already avoids exactly this (`dsp_dtx.py:128-129`). | Compute the lag without BLAS: elementwise products and sums, or `np.correlate` on a 1 s excerpt; or wrap `measure()` in `threadpoolctl.threadpool_limits(1)` if available. Keep `[aligned]`. |
| F1.3 | (apm_ns, dtx_gate) | minor | exploop_compat | `SLS_setup/b4wave/augment.py:284, 308` | **Worker-side lever failures are silent.**<br>• Exceptions pass the clip through and are only counted.<br>• The wave's early check forbids `failed \(` (`scripts/exploop/supervisor.py:606`), but nothing is printed.<br>• The parent prints `codec … failed (…)` for codec failures, and `opus_plc`'s ffmpeg fallback keeps that message.<br>So a systematic worker-side failure (the start-up selftests run in the main process only) first shows as `fail_*` counts in the epoch-1 B3 line, about 30 min in, and quietly turns the arm into a B2f replicate. | Print the first failure per process: `[b4wave] wrtc_ns failed (<exc>); passing through`, and the same for `wrtc_dtx`. That puts lever failures under the same early-check policy as codec failures. Counting stays as is. |

### F1.1 detail: `dtx_gate` under-doses the mechanism it targets

**Issue.** The realised gating dose is far below the reserved eval op's. A null arm would therefore be pre-registered
as "refuted → Drop" when it never exercised the mechanism.

Training dose, measured on real training clips (`checks_cpu.log:174, 178`, `dtx_balance` rows, drawn parameters):

| condition | class | mean gated fraction | clips with ≥ 20 % gated (`dtx_g20_40` + `dtx_g40p`) |
|---|---|---|---|
| none | bona | 0.058 | 5 / 200 |
| none | spoof | 0.046 | 4 / 200 |
| after the B3 stage | bona | 0.045 | 3 / 200 |
| after the B3 stage | spoof | 0.032 | 1 / 200 |

Eval-op dose. Held-out `vad_gate` (computed from `data/dev_noisy_sim/heldout_v1/meta.csv`, n 629):

| statistic | value |
|---|---|
| median gated fraction | 0.125 |
| mean gated fraction | 0.233 |
| clips < 1 % gated | 1.3 % |
| clips 1-20 % gated | 66.6 % |
| clips 20-40 % gated | 15.7 % |
| clips ≥ 40 % gated | 16.4 % |

That is 32.1 % of clips at ≥ 20 % gated, against 0.5-2.5 % in training. The plan expected modes 2-3 to fill
`dtx_g40p` (`plan_agreed.md:479`); it is 0-1 of 200 clips. The cause is in the README itself (l.102-103):
webrtcvad's internal overhang of 100-330 ms, plus U(40, 200) ms, gates only pauses longer than ~0.2-0.5 s.

The engineer logged this as an open issue with no decision (`impl/progress.md`, dtx_gate "Measured"). Meanwhile
falsifier (1) (`plan_agreed.md:460-461`) and the script's lever verdict (`b4_dsp_slices.py:626`) would read a null
as "gating is not what drives the core. Drop."

**Fix (readout and README only, no training change).**
1. With `--lever dtx_gate`, print the arm's realised gated-fraction bin shares, per class and summed over epochs,
   from `metrics.jsonl["b4"]["aug_by_class"]` (`dtx_g*` counts). Print the held-out `vad_gate` shares from
   `heldout_v1/meta.csv` next to them (1.3 / 66.6 / 15.7 / 16.4 %).
2. Pre-register a dose rule in the script and in the README verdict rules. Let dose_ratio be the arm's
   (`g20_40` + `g40p`) share divided by the held-out share (0.321). If dose_ratio < 0.25, the lever verdict cannot
   be "refuted": it reads "inconclusive (under-dosed)", and falsifier (1) reports "under-dosed: owner decides a dose
   variant" instead of "Drop". A positive result still counts.
3. List the dose under "open issues for the owner" before smoke. A higher-dose variant within the same component
   (for example hangover U(0, 200) ms, or `p` up) is a training change and needs the owner.

## 5. Resolved since the last round

None (first review round).

## 6. Notes, no action needed

- **apm_ns selftest values:** `no_gain` reads −6.0 / −12.4 / −18.4 / −21.5 dB on the AM harmonic complex. Once
  primed on it, the suppressor treats the stationary periodic signal as noise. The one-sided assert allows this,
  as the plan intended. On real clips NS removes 0.6-1.1 dB with no stage and 5-12 dB in low-energy frames with
  noise (`checks_cpu.log:114-129`), so the backend behaves as a suppressor.
- **opus_plc path alignment:** the in-process path is sample-aligned with the input. The ffmpeg 4.2 path lags it
  by ~70 samples (`checks_cpu.log:224-228`, "lag +70"). This makes no difference after the random 4.04 s crop.
  The attribution control covers the decoder difference.
