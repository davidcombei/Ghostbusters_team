> **Snapshot for the exploop supervisor** (copied 2026-10-10 from `/root/.claude/plans/we-have-implemented-some-ethereal-whistle.md`,
> approved 2026-10-09 in session 0863b758). Agents read this copy, never `~/.claude/plans/` (plan files get reused).
> Changes since approval, agreed 2026-10-10: the loop now runs under `scripts/exploop/supervisor.py` (not in a live
> session); launchers are rendered by the supervisor from the designer's JSON (no launcher agent); the per-iteration
> 2 % smoke run is OFF by default and replaced by an early check of the real run (header lines, no traceback / codec
> failures, ETA), see `wave.json` `smoke` / `early_check`; per-epoch results go to `STATUS.md` / `events.jsonl` instead
> of the chat; memory updates happen in `/exp-loop wrapup B3` after the wave. B3a was trained (2026-10-09 21:48 ->
> 2026-10-10 05:41) but never evaluated: the wave is adopted at B3a's EVAL step.

# B3 wave: 5-iteration training exploration from B2f (CLI flags only)

## Context

B2f (`SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`) is the current best: Codabench WF1 92.49 from its top-5 merge, local half_B WF1 **91.98** (merge_top5), 92.14 (epoch 8).

The b3wave package (`SLS_setup/b3wave/`, committed in 94b2c2a) adds training levers. Every lever is a CLI flag, and each defaults to the old behaviour:

| Flag | Lever |
|---|---|
| `--consistency` | paired consistency |
| `--sampler` | balanced / language-balanced sampler |
| `--aug_snr_db` / `--aug_snr_dist` | SNR range |
| `--aug_codec_profiles` | codec zoo |
| `--aug_max_stages` | augmentation chaining |
| `--specaug` | SpecAugment |
| `--llrd` / `--freeze_bottom_k` | layer-wise LR decay / bottom-k freezing |

The goal is to run 5 iterations, one lever change per iteration, built on the current best. Each run is evaluated with `/post-train-eval`, logged in `docs/experiment_log.csv`, and documented in `docs/experiments-exploration/b3-experiments/`. The best experiment moves to whichever run has the higher local half_B WF1.

B2f's documented weaknesses, which drive the lever choice:
- Noisy bona recall is ~74 against 99 for spoof.
- 16–20% of noisy bona sit at p ≥ 0.99, and no LR or schedule change moved this.
- English bona trails Chinese bona by 15–20 recall points.
- Bona recall on the g726/gsm codecs is ~65–70.
- Bona recall at SNR 0 dB is 61–66.
- `noise_only` and `sil_only` bona clips are scored 100% spoof.
- Dev loss rises from epoch 1, so the run lost the level-aug regulariser.

## Decisions agreed with the user

- **Best readout:** half_B WF1@0.5 of the **top-5 merge**, compared merge against merge. The bar starts at B2f's 91.98. Epoch 8 is always evaluated and logged as a second row.
- **A new best** is any run whose merge has a strictly higher half_B WF1. The next iteration then builds on that run's flags.
- **MUSAN pool:** keep B2f's `--musan_dir ../data/augmentation/musan`. Because music is in that pool, `sim_unseen_v1` is not read in this loop.
- **Compute:** ~8 h consistency runs are allowed.
- **Provenance:** b3wave is committed (94b2c2a, tree clean). I make no further commits unless asked.
- **Who launches runs:** the user handed the loop over, so I launch the runs in-session. Constraints: CLI flags only, no code changes, at most 10 epochs, `--keep_topk_by_wf1 7 --keep_epochs 8`, seed 1234.

## Base command (B2f flags on the B3 trainer)

`cd SLS_setup && python -u -m b3wave.train` with B2f's exact `config.yaml` command:
- data: RTC aug, rnnoise/esc50/rirs pools, `--musan --musan_dir ../data/augmentation/musan`, `--loudness_norm --loudness_norm_lufs -23`
- optimiser: `--optim adamw --lr 2.5e-6 --head_lr_mult 10 --weight_decay 0.01`
- schedule: `--lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 5e-7 --num_epochs 8`
- stability: `--grad_clip 1.0 --ema_decay 0.9995`
- eval and pruning: `--local_eval_sets ../configs/local_eval_sets.yaml --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 --keep_epochs 8`
- run: `--batch_size 16 --num_workers 64 --seed 1234`

With every B3 flag at its default, b3wave reproduces the original trainer. Every eval passes `--loudness_norm --loudness_norm_lufs -23`, which `/post-train-eval` reads from `config.yaml`.

## Iteration 1 (B3a): paired consistency

**Change:** add `--consistency js --consistency_lambda 0.3 --consistency_pairs both`.

**Why this lever first:**
- It is the highest-evidence lever in the roadmap (T3.1, P0, +2–5 noisy F1).
- It attacks the dominant residual error, noisy bona being scored as spoof, which is a separability problem: noisy EER ~10, and the saturated tail never moved.
- It does this by making the offline/online RTC twins (37,125 pairs, both views augmented) and independently augmented views of the same clip agree in output.

**Compute side-effect:** `--batch_size 8` is required, meaning 8 pairs = 16 clips per step, the same GPU memory as B2f. This doubles the steps per epoch, so the run takes ~8 h.

**Track and launcher:**
- Track: `B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234`
- Launcher: `scripts/run_B3a.sh`

**Pre-registered rule** (stated in the script header and the report):
- **Promote:** Δ merge half_B > +0.5 with the paired-bootstrap CI excluding 0, and clean F1 down by no more than 0.5.
- **Null:** Δ within ±0.5.
- **Negative:** Δ below −0.5.
- **Expected signature:** noisy EER down, `noise_only` bona→spoof down, heldout/matched bona recall up, clean flat.

## Candidate queue for iterations 2–5 (chosen adaptively, one factor each)

The next lever is picked from the previous run's analysis. Each iteration builds on the current best's flags.

**If consistency helps:**
- A dose step: `--consistency_lambda 1.0`, or `js_emb`.
- Augmentation levers stacked on top of it.

**Augmentation levers:**
- **`--aug_codec_profiles random`:** targets the heldout codec bona recall (g726/gsm are reserved, so the gain has to come from generalisation).
- **`--aug_snr_db 0 20`:** targets the SNR-0 slice. Uniform keeps the high-SNR mass, while the beta mode-8 draw dropped the ≥15 dB share from 33% to 13% in smoke.
- **`--aug_max_stages 2`:** compound degradations.
- **`--specaug`:** regularisation, since dev loss rises.
- **`--use_level_aug` stacked on the loudness norm:** restores the regulariser B2f lost.

**Sampler:**
- Flags: `--sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5`.
- CE weights are re-derived so the effective class prior of 0.1/0.9 at ~19% bona stays constant: ≈`--ce_weights 0.32 0.68`. The launcher agent computes the exact value from `train_label.txt`.
- This isolates exposure (English bona) from the operating point. CE 0.5/0.5 would cut bona's effective weight from ~2.1:1 to 1:1 and deepen the spoof bias.

**Optimiser:** `--llrd 0.9`. 0.85 at lr 2.5e-6 nearly freezes the bottom layers.

**Selection rule:** pick the lever whose target is the largest remaining error in the latest report, and prefer a configuration change to a seed repeat.

## Per-iteration loop

1. **Design (me).** Pick the lever, then write down the rationale, the track name and the falsification rule. The base is the current best's `config.yaml` command.
2. **Launcher agent (fresh general-purpose agent per iteration).**
   - Writes `scripts/run_B3<x>.sh`, modelled on `scripts/run_B2f.sh` but using `python -u -m b3wave.train`.
   - Keeps the same preflight: ffmpeg, the rtc-sdd python, pyloudnorm.
   - Adds a header with the one change, evidence, falsification rule and eval note.
   - Runs `bash -n`.
   - Runs a smoke of the exact flags: `--train_subset_frac 0.02 --num_epochs 1 --out_path ./exp_smoke --track B3<x>_smoke`.
   - Checks the header lines (`b3wave`, `Consistency`/`Sampler`/`SNR`, `loudness_norm`), checks for no Traceback/OOM and no `failed (` codec lines, and notes the step rate.
   - Returns the script path, the full command and the smoke result. It does not launch the full run.
3. **Launch (me).**
   - Check the GPU is idle and ffmpeg is present.
   - Run `bash scripts/run_B3<x>.sh`.
   - Within ~5 min, check the nohup log `SLS_setup/<track>_<ts>.log` for the header lines, codec failures and step rate.
4. **Per-epoch display (me).**
   - For each epoch, run a background Bash `until` watcher (`run_in_background`).
   - It exits when the Nth `LocalWF1` line lands in `exp/<run>/logs/train.log`, or when a Traceback/OOM appears, or when the trainer PID disappears.
   - On each notification, post the epoch's TrainLoss / DevLoss / LocalWF1 / F1clean / F1noisy / WF1oracle / EERnoisy, the Δ against B2f's same epoch (from its `metrics.jsonl`) and the best epoch so far. Then re-arm.
   - **If the PID died with no traceback:**
     - Compare the log mtime with `ps -o lstart= -p 1`.
     - Reinstall ffmpeg if a container restart wiped it.
     - Resume with `bash scripts/run_B3<x>.sh --resume exp/<run>`.
5. **`/post-train-eval <run> --k 5 --baseline <best>/local_eval/merge_top5`** (Skill tool).
   - The skill's "still training" check greps `main_train.py`, so I also check `pgrep -af b3wave.train`.
   - It produces: top-5 selection, the merge eval on the 17 sets, the probe report, the compare against the current best, progress scores, and a validated `submission.zip`.
   - Manual additions, with the same `$MODEL` flags including loudness:
     - (a) `scripts/local_eval.py --model_path ckpt/epoch_8_*.pth --out local_eval/epoch_8 --names <17 sets>`, then `probe_report.py`, then `compare_evals.py` against the best's `epoch_8`.
     - (b) `scripts/b3_compare_evals.py <best>/merge_top5 <run>/merge_top5 --labels … --ci 2000`, and the same for ep8. Copy each `compare_ci.json` to `compare_ci_vs_<label>.json` so later comparisons don't overwrite it.
     - (c) A `compare_all_readouts.txt` with columns B2f-top5, B2f-ep8, run-top5, run-ep8, plus the current best if that is not B2f.
6. **Analysis agent (fresh general-purpose agent).**
   - Writes `docs/experiments-exploration/b3-experiments/<YYYYMMDD_HHMMSS>_<track>.md`, structured like the B2f report:
     - provenance
     - §1 setup and flag table vs base
     - §2 results vs B2f and the current best for both readouts, with CI, per set, probes, slices and the lever-response table
     - §3 per-epoch behaviour
     - §4 what went OK
     - §5 what went wrong
     - §6 verdict on the pre-registered rule
     - §7 decision and ranked next-flag proposals
   - It never reads or analyses progress scores.
   - It returns the headline table, the good and bad takes, and the ranked proposals.
7. **Log (me), using the log-experiment skill's script.**
   - `scripts/log_experiment.py --run_dir … --tag merge_top5 --id B3<x>-<lever>-screen --ref <best row id> --change_vs_ref … --decision … --notes … --dry_run`, shown in-session. Then I write it without a per-row pause, since the loop was handed over.
   - The same again for `--tag epoch_8` with id `B3<x>-<lever>-ep8`.
   - Then `scripts/b3_log_ci.py --id … --ci_json …` fills `ci_low`/`ci_high`.
   - **Decision column:**
     - `promote` = new best, CI excluding 0 and clean F1 down no more than 0.5
     - `revisit` = new best on the point estimate only (it still becomes the best)
     - `drop` = not better
8. **Decide (me).**
   - Update the best if the merge half_B beats it.
   - Post a short iteration summary: headline Δ, CI, what moved, and the next lever with its reason.
   - Go to step 1.

## After iteration 5

- Write a wave summary, `docs/experiments-exploration/b3-experiments/<ts>_B3_wave_summary.md`, covering:
  - the iteration table
  - what each lever did
  - the final best and its submission zip path
  - the recommended full-run recipe
- Update the memory notes on the B3 outcome and recipe, and the loop handover of 2026-10-09.
- Make no commits. List the changed files: `scripts/run_B3*.sh`, `docs/experiment_log.csv`, `docs/experiments-exploration/b3-experiments/*`.

**Budget:** B3a takes ~8 h, and the other runs ~4–8 h depending on whether consistency is in the base. Each run's eval and smoke add ~15–20 min. The whole loop is ~30–45 h.

## Critical files

- Trainer and docs: `SLS_setup/b3wave/train.py`, `SLS_setup/b3wave/README.md`
- Launcher template: `scripts/run_B2f.sh`
- Evaluation: `scripts/local_eval.py`, `scripts/compare_evals.py`, `scripts/b3_compare_evals.py`, `scripts/probe_report.py`, `scripts/make_submission.py`
- Logging: `scripts/log_experiment.py`, `scripts/b3_log_ci.py`
- Skill: `.claude/skills/post-train-eval/SKILL.md`
- Registry: `docs/experiment_log.csv`
- Report template: `docs/experiments-exploration/20261009_175355_B2f_*.md`
- B2f reference evals: `SLS_setup/exp/B2f_*/local_eval/{merge_top5,epoch_8}`

## Verification (per iteration)

- **Smoke:** passes, and the header lines show exactly the intended flags.
- **Config:** the `config.yaml` `command` equals the launcher's command, and `git_sha` is clean on 94b2c2a or later.
- **No silent failures:** no `failed (` codec lines and no Traceback in the nohup log. All 8 epochs are present in `metrics.jsonl`.
- **`/post-train-eval` outputs:**
  - `local_eval/merge_top5/metrics.json`
  - `compare_vs_<best>.txt`
  - `model_merging_top5/submission.zip`, validated by `make_submission.py`
- **Epoch 8 outputs:** `local_eval/epoch_8/metrics.json` and the CI JSONs exist.
- **Registry:** `log_experiment.py` reports no WARNING about missing report sets, and the CSV has two new rows with CI columns filled.
- **Report:** the md exists in `docs/experiments-exploration/b3-experiments/`.
