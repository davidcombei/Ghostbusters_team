---
name: post-train-eval
description: Post-training pipeline for a finished run in SLS_setup/exp/<run> - pick the top-K epochs by local WF1, score their weight merge on the 17 local report sets (+ probe report), compare it with a baseline experiment, write the progress-split predictions and pack the Codabench submission.zip. Use when a training run has finished and the user says "evaluate the run", "do the post-training steps", "top-5 merge + progress predictions", "make the submission for <run>", or invokes /post-train-eval <run_dir> [--k N] [--baseline <eval dir>].
---

# Post-training evaluation of a run

Five steps for one finished run: **1** select the top-K epochs → **2** local eval of their merge →
**3** compare with a baseline → **4** progress predictions → **5** Codabench zip.

Use the `rtc-sdd` env: `PY=/root/.conda/envs/rtc-sdd/bin/python`. Commands run from the repo root
unless the step says `cd SLS_setup`. Every number you report comes from the scripts' output; never
type metric values yourself.

## Arguments

- **run dir (required)**: `SLS_setup/exp/<run>`, `exp/<run>`, an absolute path, or just `<run>`. Resolve it to
  `RUN=SLS_setup/exp/<run>` (from the repo root) and `RUN_REL=exp/<run>` (from `SLS_setup/`). If no argument is
  given, take the newest `SLS_setup/exp/*/` with a `metrics.jsonl` and confirm it with the user.
- `--k N` (optional, default **5**): how many epochs to merge. Outputs are tagged `top{K}`.
- `--baseline <path>` (optional): a local_eval output dir (`SLS_setup/exp/<run>/local_eval/<tag>`) to compare
  against. If it is missing, ask for it (see step 0).

## 0. Preflight (read-only; do all of it before any GPU job)

1. **Model args.** Read `$RUN/config.yaml`: `args.ssl_name`, `args.n_layers`, `args.num_epochs`,
   `args.keep_topk_by_wf1`, `args.earlystop_metric`, `args.earlystop_epoch`, `args.track`. Every eval command
   gets `MODEL="--ssl_name <ssl_name> --n_layers <n_layers>"` from here (the stored `...-hf` name is fine).
   If `args.loudness_norm` is `true`, append `--loudness_norm --loudness_norm_lufs <args.loudness_norm_lufs>`
   to `MODEL`: the run normalised every clip's loudness at read time and evaluation must do the same
   (`scripts/local_eval.py` and `main_eval.py` take the same two flags; `eval_checkpoints.py` reads them itself).
   No `config.yaml` → stop: the run predates the per-epoch metrics and `select_topk.py` cannot rank it.
2. **Has training finished?**
   - `pgrep -af main_train.py`: if a process mentions the run's track or run dir, training is still running.
     Stop and tell the user (a top-K picked now would change later).
   - Otherwise compare the last epoch in `$RUN/metrics.jsonl` with `num_epochs`. If it is short, find the best
     epoch by `wf1.wf1`. If `earlystop_metric` is `wf1` and `last - best >= earlystop_epoch`, it **early-stopped**
     (main_train prints that only to stdout, not to `logs/train.log`). Report it with the best epoch. If it is
     short for any other reason it was probably killed (container restart): tell the user it can continue with
     `--resume $RUN_REL` and ask whether to evaluate the partial run anyway.
3. **K is available.** If `keep_topk_by_wf1` is non-zero and K > it, the extra checkpoints were pruned:
   ask for a K ≤ `keep_topk_by_wf1`.
4. **GPU.** `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`. If the GPU is busy
   (another run), say so and ask whether to continue or to use another GPU (`CUDA_VISIBLE_DEVICES=<id>`).
5. **Baseline (needed in step 3; ask now so the GPU jobs never wait on the user).** If `--baseline` was not
   given, list the candidates:
   ```bash
   ls -d SLS_setup/exp/*/local_eval/*/ | grep -v "/ckpts/" | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done
   ```
   Leave out the run itself. Ask with AskUserQuestion: up to 4 of the most relevant as options. Put
   `B0_*/local_eval/merge_top5` first, marked (Recommended), when it exists: it is the 83.45 progress
   reference. Prefer `merge_top*` and `best_by_wf1` tags, and the user can type another path under "Other".
   Accept a run dir as the answer too; if it holds several local_eval tags, ask which one.

Then post a short plan (run, K, model args, baseline, outputs) and go on. Don't ask for a further confirmation.

## 1. Select the top-K epochs

```bash
cd SLS_setup
$PY ../scripts/select_topk.py --run $RUN_REL --k $K
```

- Writes `$RUN/top{K}_by_wf1/merge_list.txt`. It ranks by the per-epoch WF1 in `metrics.jsonl`, which is
  computed on the half-A selection sets, so half B stays clean for step 2.
- Show the printed table. Point out when the chosen epochs are all early (e.g. the best epoch is 1–3 of a
  30-epoch run), which suggests overfitting or a schedule that never annealed. Don't block on it.
- If it exits with "checkpoints missing", the epochs were pruned: report it and ask for a smaller K.

## 2. Local evaluation of the top-K merge (GPU)

**Cache check first.** `local_eval.py` reuses `<out>/scores/<set>.txt` whenever the set's utterance list
matches, **without checking which checkpoints produced them**. If `$RUN/local_eval/merge_top{K}/` already
exists, compare its `merge_list.txt` (absolute paths) with the new list resolved against `SLS_setup/`:

```bash
OUT=$RUN/local_eval/merge_top$K
if [ -d $OUT/scores ]; then
  diff <(sed 's#^#SLS_setup/#' $RUN/top${K}_by_wf1/merge_list.txt | xargs -n1 realpath | sort) \
       <(xargs -n1 realpath < $OUT/merge_list.txt 2>/dev/null | sort) >/dev/null \
    && echo "same checkpoints: cached scores are valid" || echo "DIFFERENT checkpoints: add --force"
fi
```

If the lists differ, or there is a `scores/` dir but no `merge_list.txt`, add `--force` to the command below.

Run it with Bash `run_in_background: true` (it takes a while). You are notified when it exits; don't poll:

```bash
NAMES="dev_online_clean sim_matched_v1 sim_heldout_v1 sim_echo_v1 probe_orig probe_sil_only probe_noise_only probe_trimmed probe_padded probe_gain_up probe_gain_down probe_gain_m20 probe_gain_m6 probe_gain_p6 probe_gain_p10_limit probe_agc_dynaudnorm probe_loudnorm"
mkdir -p $OUT
$PY scripts/local_eval.py --merge_list $RUN/top${K}_by_wf1/merge_list.txt $MODEL \
    --out $OUT --names $NAMES [--force] 2>&1 | tee $OUT.log
```

- **`$NAMES` must stay unquoted**: `--names` takes separate arguments, and `--names "$NAMES"` passes one
  bogus set name (KeyError). Always pass all 17 sets; the default list breaks comparability with B0.
- On failure, show the traceback tail and stop. Don't start step 4 with a broken step 2 unless the user asks.

Then run the probe report (CPU, seconds) and quote the `[half_B]` line from `$OUT.log`:

```bash
$PY scripts/probe_report.py $OUT/scores        # writes $OUT/probe_report.json
```

## 3. Compare with the baseline

If the baseline has probe scores but no `probe_report.json`, create it first (CPU, reads cached scores only):
`[ -f <baseline>/scores/probe_orig.txt ] && [ ! -f <baseline>/probe_report.json ] && $PY scripts/probe_report.py <baseline>/scores`.

```bash
$PY scripts/compare_evals.py <baseline> $OUT --labels <BASE_LABEL> <RUN_LABEL> 2>&1 | tee $OUT/compare_vs_<BASE_LABEL>.txt
```

Labels are short, e.g. `B0-top5` and `B1-top5`. Add `$RUN/local_eval/best_by_wf1` as a third column if it
exists. Read the table and give the user a short interpretation:

- **Δ WF1 on half B** is the headline. With one seed, |Δ| ≲ 1.5 is within epoch-to-epoch noise; say so instead of
  calling a winner. (The decision rule in docs/next_steps_plan.md needs two seeds and a paired-bootstrap CI.)
- **Threshold shift vs separability.** Compare Δ`wf1` with Δ`wf1_oracle` and Δ`eer`. A large F1 drop with a small
  oracle/EER change, where spoof recall rises while bonafide recall falls (or the reverse), is a calibration shift.
  A calibration fold-in can recover it. EER moving means real separability changed.
- **Clean**: flag a clean F1 drop of more than 0.5.
- **Probes**: `sil_only bonafide` and `noise_only bonafide` with a high share predicted spoof means silence or
  noise is read as spoof. A higher `level_flip_bona` means more of the level shortcut.

## 4. Progress predictions with the same merge (GPU)

`main_eval.py` opens the merge-list paths exactly as written (relative to `SLS_setup/`), so it **must run from
`SLS_setup/`**. If `$RUN/model_merging_top$K/scores.txt` already exists and that dir's `merge_list.txt` equals
`$RUN/top${K}_by_wf1/merge_list.txt`, skip to step 5 unless the user asked to redo it.

Run in the background as in step 2:

```bash
cd SLS_setup
mkdir -p $RUN_REL/model_merging_top$K
$PY main_eval.py --model_merging --merge_list $RUN_REL/top${K}_by_wf1/merge_list.txt $MODEL --arch sls \
    --eval_data_path /mnt/paml-research/RTCFake/data/wav/progress \
    --protocol_path /mnt/paml-research/RTCFake/data/progress.txt \
    --score_path $RUN_REL/model_merging_top$K/scores.txt --batch_size 32 --num_workers 8 \
    2>&1 | tee $RUN_REL/model_merging_top$K/main_eval.log
```

## 5. Codabench archive

```bash
$PY scripts/make_submission.py $RUN/model_merging_top$K/scores.txt
```

It checks that all 30,279 progress ids are scored exactly once, with scores in [0, 1], and writes
`$RUN/model_merging_top$K/submission.zip` with `scores.txt` at the root. If it reports errors, show them and do
not hand over a zip. **Do not upload it**; the user submits on Codabench. **Never** compute or report statistics
of the progress scores (spoof rates, per-subset distributions). The rules forbid using progress outputs for
analysis or model selection.

## Finish

Report:
- a short table: half_B WF1 / F1 clean / F1 noisy / EER noisy for the run vs the baseline, with deltas;
- the merged epochs;
- the paths: `$OUT/metrics.json`, `$OUT/compare_vs_<BASE_LABEL>.txt`, `$RUN/model_merging_top$K/submission.zip`.

Remind the user:
- choose what to submit on **half B, not progress**;
- the progress leaderboard is a periodic sanity check, about once per milestone;
- after the score comes back, `/log-experiment $RUN` (tag `merge_top$K`) records the row with `progress_score`.

Don't commit anything.
