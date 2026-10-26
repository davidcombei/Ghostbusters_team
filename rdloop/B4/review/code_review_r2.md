# rdloop B4: code review r2

> research-critic, CODE REVIEW round 2, 2026-10-17. Scope: the changes since round 1 (`rdloop/B4/review/diff_r2.patch`):
> `SLS_setup/b4wave/{augment.py, dsp_apm.py, checks.py, README.md}` and `scripts/b4_dsp_slices.py`. Plan:
> `rdloop/B4/plan_agreed.md` / `.json` (approved `apm_ns`, `dtx_gate`, `opus_plc`), plus the owner notes of 2026-10-17
> 13:48 (O3.1-O3.3).
>
> **Read:**
> - the whole diff;
> - the engineer's report `impl/fix_f1.json` and `impl/progress.md` ("fix_f1");
> - the static results `tests/static.json` and `tests/static/*.log`;
> - in the current tree:
>   - `scripts/b4_dsp_slices.py`: `analyse`, `lever_verdict`, the printing;
>   - `SLS_setup/b4wave/train.py`: `b4_class_row`, `read_metrics`, the metrics.jsonl writer;
>   - `SLS_setup/utils/sim_ops.py`: `vad_gate`;
>   - `scripts/exploop/supervisor.py`: the early check;
> - plan §2.1-2.3 falsifiers and §4.
>
> **Ran (CPU only, nothing written outside a tempdir):**
> - `python -B -m b4wave.checks lever_failures`: 3/3 PASS;
> - `python -B -m b4wave.checks dsp_slices`: 20/20 PASS;
> - `diff -r` of `review/snap_r2` against the working tree: identical (module and script).
>
> No progress-split file was read.

## 1. Verdict

All three round-1 findings are fixed as required, and the levers-off path is unchanged.
- **F1.1 (major):** fixed in the readout and the README only, with no training change:
  - the dose is printed next to the held-out `vad_gate` dose;
  - a pre-registered dose rule turns an under-dosed null into "inconclusive (under-dosed)", and falsifier (1) then
    reads "owner decides a dose variant" instead of Drop;
  - the issue is listed for the owner (`README.md` "Dose", `impl/fix_f1.json` `open_issues`).
- **F1.2 (minor):** `_lag` no longer calls BLAS. The selftest's CPU time drops from 546.91 s to 0.53 s
  (`tests/static/checks_cpu.log:100`).
- **F1.3 (minor):** the first lever failure per process and op now prints a line that the exploop early check
  (`failed \(`) catches. The fix also extends this to `opus_plc`'s ffmpeg fallback, which is justified.

The static suite passes on the round-2 code:
- `checks cpu`: 126/126 (`tests/static/checks_cpu.log:299`);
- `b3wave.checks all`: 22/22;
- exploop `test_lib` / `test_view`: 13 / 14 OK;
- manifest: 138 protected files unchanged.

One new minor finding (F2.1): a `dtx_gate` report prints two different definitions of the falsifier (1) outcome
escape. It is readout wording only.

**Verdict: approve.** No blocker or major finding is open.

## 2. Plan conformance

| lever | ok | note |
|---|---|---|
| `apm_ns` | yes | **Changed:** only `dsp_apm._lag` (`np.sum(xc * y[...])`, `dsp_apm.py:145`) and the failure print (`augment.py:295`). The `[aligned]` assert is kept; lag is 0 at every level.<br>**Unchanged:** flags, header and selftest asserts; timing factor 1.012 (`checks_cpu.log:246`). |
| `dtx_gate` | yes | **Training code:** only the failure print changed (`augment.py:319`). The dose stays as approved, and the dose change is left to the owner as round 1 required.<br>**Readout:**<br>• the dose block (`dtx_dose`);<br>• the dose rule (`DTX_DOSE_MIN` 0.25; dose unknown is treated like under-dosed);<br>• `lever_verdict` (`b4_dsp_slices.py:768-772`);<br>• `dtx_gate_reading`, with plan §2.2 falsifiers (1)-(3) (`:649-684`).<br>**Falsifiers:** (2) uses the plan's noisy and VAD spoof-miss guards (`DTX_GUARDS`; bounds 36 / 35 and 25 / 22 are the plan's "must not move" values) and clean F1 down by more than 0.5. (3) is the shared under-fit rule (fit_bad AND heldout bona recall down, `:507-513`). |
| `opus_plc` | yes | **Changed:** only the failure print before the ffmpeg fallback (`augment.py:355`), with the action "ffmpeg fallback with the same profile". It is an extension of F1.3, and the engineer justifies it: a systematic libopus failure would otherwise make the arm a silent B2f replicate. |
| shared | yes | **`_lever_failed`** (`augment.py:256-264`): it counts every failure and prints once per (op, pid). `fail_printed` lives on `RTCAugmenterB4` only.<br>**Levers off:** `build_rtc_augmenter_b4` still returns the B3 object, so nothing new is reachable with the levers off.<br>**`train.py`:** not touched in round 2. |

## 3. Checklist (round-2 changes)

- **Default-off equivalence.** The new code sits in four places, none reachable with the levers off:
  - `RTCAugmenterB4.__init__` (a dict, no RNG draw);
  - the exception branches of the lever methods;
  - the apm selftest (apm on only);
  - the CPU readout and checks.

  The groups `augmenter`, `imports`, `trainer`, `data` and `flags` still pass inside 126/126. `module_sha256` changes,
  as expected, because `augment.py`, `dsp_apm.py` and `checks.py` are module files.
- **Leakage.** `heldout_vad_dose()` reads `data/dev_noisy_sim/heldout_v1/meta.csv` inside the CPU readout script only.
  Nothing under `SLS_setup/b4wave/` imports the script; the readout already reads that metadata for its contrasts.
  No eval statistic reaches training.
- **Isolation.**
  - The diff touches only the module, `scripts/b4_dsp_slices.py` and the cycle directory (scratch probes,
    progress, logs).
  - The manifest check passes.
  - The fabricated arm in `checks dsp_slices` is built in a `tempfile` directory with symlinks to B2f's readouts.
- **Correctness.**
  - **Dose metric:** the held-out `gated` is the fraction of frames gated (`sim_ops.py` `vad_gate`, `round(gated.mean(), 3)`).
    The lever's `gated_frac` is the fraction of samples gated (`dsp_dtx.py:158`). Both are the gated share of the
    clip, compared conditional on the op being applied, so they are comparable.
  - **Bins:** the edges equal `dsp_dtx.gated_bin`, tested on a grid that includes the edges.
  - **metrics.jsonl:** `arm_dtx_dose` reads the format `b4_class_row` writes (`train.py:76-88`), and the later row
    for an epoch wins, as in `read_metrics` (`:248`). Every row the trainer writes has `epoch` (`:959`).
  - **Escape:** it needs both readouts (`len(wf1) == 2`), so an arm read before its merge exists cannot escape. This
    is consistent with O3.2.
  - **Printing:** `fmt` handles `None` (`dose unknown`). `crit2` values are ints (`:+d`).
- **Exploop contract.** No change to the `Epoch` lines, `Experiment saved to:`, `config.yaml`, checkpoints or the
  run-dir name. The new `[b4wave] <op> failed (` line matches the forbidden pattern
  (`scripts/exploop/supervisor.py:604-607`). It is enforced only in the smoke log and in the early window of the real
  run (`:740-760`, 10 min / 200 iterations), as for the parent's codec failures. So a sporadic failure late in a
  run does not kill it, while a systematic one is caught at the first failing clip.
- **Tests.**
  - `lever_failures` uses raising stubs per lever. It checks the count, a single line and a reprint after a pid
    change.
  - The six new `dsp_slices` checks cover:
    - the held-out dose (n 629, shares 1.3 / 66.6 / 15.7 / 16.4 %, ≥ 20 % 0.321; `checks_cpu.log:268-269`);
    - the summed arm dose from a fabricated `metrics.jsonl`, with a levers-off row and a repeated epoch;
    - the rule edges;
    - the lever verdict per dose;
    - the falsifier decisions, including the one-readout escape.

  The tests are not tautological: the dose is recomputed from data, and the verdict cases go through the script's
  own functions.

## 4. Findings

| id | lever | severity | category | file:line | issue | fix |
|---|---|---|---|---|---|---|
| F2.1 | dtx_gate | minor | other | `scripts/b4_dsp_slices.py:899` vs `:661` | **A `dtx_gate` report prints two different falsifier (1) outcome escapes.**<br>• The generic outcome block prints `falsifier (1) outcome escape (heldout F1 delta >= +0.5 on merge_top5 AND final)`. This is the O3.2 wording, written for `apm_ns` / `opus_plc`, whose falsifier (1) is on heldout F1.<br>• The new dtx decision lines use **WF1** Δ ≥ +0.5 on both readouts (`DTX_WF1_ESCAPE`, from the plan §2.2 text "with WF1 Δ < +0.5").<br>• An arm with heldout F1 +0.6 / +0.6 but WF1 +0.3 / +0.3 would therefore read "escape True" in one block and "falsifier (1) fires" in the other.<br>Both readings have a source: the plan text for WF1, and the literal owner note O3.2 for heldout F1. The engineer documented the choice (`README.md:323`, `impl/progress.md` fix_f1). It matters only at an adequate dose; under-dosed, falsifier (1) never drops. | **Wording only.**<br>• For `--lever dtx_gate`, label the generic escape line "(apm_ns / opus_plc; dtx_gate: see its decision lines)".<br>• Or print both metrics' escapes in the dtx decision line, keeping WF1 as the decision metric.<br>The owner confirms the metric at the post-smoke review. |

## 5. Resolved since round 1

| id | how |
|---|---|
| **F1.1** (major) | **The script prints the dose.** `print_dtx_dose` shows the arm's per-class and pooled `dtx_g*` shares, summed over epochs, next to the held-out `vad_gate` shares, plus the dose ratio and rule.<br>**The rule is pre-registered.** It is in the README verdict rules (`README.md` "dtx_gate dose rule") and the code: ratio < 0.25 or unknown → the lever verdict cannot be "refuted" (`b4_dsp_slices.py:772`), and falsifier (1) → "owner decides a dose variant".<br>**A positive result still counts at any dose.** Falsifiers (2) and (3) still drop.<br>**The issue is listed for the owner.** It is in the README "Dose" bullet and `fix_f1.json` `open_issues`.<br>I confirm the 0.25 threshold and the "dose unknown ⇒ no refuted" reading that the engineer asked about. |
| **F1.2** (minor) | `_lag` sums elementwise products (`dsp_apm.py:145`). The static log shows apm selftest CPU 0.53 s (`checks_cpu.log:100`); round 1 had 546.91 s. |
| **F1.3** (minor) | `_lever_failed` (`augment.py:256-264`) is used by all three levers. The `lever_failures` check passes: 3/3, one line each, reprinted after a pid change (`checks_cpu.log:235-241`; reproduced). |

## 6. Notes for the owner (no action required by the engineer)

- **Expected reading of the `dtx_gate` arm as built.** I computed the arm's ≥ 20 %-gated share from the `dtx_balance`
  bins in `tests/static/checks_cpu.log:174, 178` (drawn parameters):

  | condition | ≥ 20 %-gated clips | dose ratio |
  |---|---|---|
  | none | (4+1+4+0) / 400 = 2.25 % | 0.070 |
  | after the B3 stage | (3+0+1+0) / 400 = 1.0 % | 0.031 |

  Both are far below the 0.25 rule. Run as approved, a null arm will almost certainly read "inconclusive
  (under-dosed)". Its ~4.4 h of GPU can show the mechanism, but it cannot refute it. Whether to run it as is, or to
  approve a higher-dose variant first, is the owner's call; the training change is not in scope here.
- **What a lever failure does to a run.** The new `[b4wave] <op> failed (` line follows the same policy as a codec
  failure: a single failure in the smoke run, or in the first ~10 min of a real run, stops the run and sends it to
  redesign. On the real training clips in the checks, every lever had 0 failures (`fail_*` counts in
  `checks_cpu.log:45, 53, 107, 159`). The only failure path seen is a non-16 kHz input (22.05 kHz), and the training
  loader cannot produce one: it loads at `sr=16000` (`SLS_setup/utils/data_utils.py:192`).
- **Still to come.** The GPU checks (`gpu --ckpt`) and the forked 64-worker path of the failure print are covered by
  the regression and smoke steps, not by this review.
