---
name: exp-report
description: Write the evidence-only analysis report of a finished training run against its references (setup, results on half_B for both readouts with CI, per-epoch behaviour, what went OK / wrong, verdict on the pre-registered rule, ranked one-flag proposals), or the wave summary. Used by the exploop ANALYZE / SUMMARY steps (exp-analyst) and interactively when the user says "analyze the run", "what went wrong with <run>", "compare <run> with <ref> and propose the next experiment", "write the iteration report", or invokes /exp-report <run> [--ref <eval dir>].
---

# Run analysis report

**The rule:** every number in the report is copied from a file you read in this session, and the
report names the file. Never type a metric from memory, never recompute the supervisor's verdict,
never read progress-split scores. Tables over prose.

## Where the evidence is (run dir `SLS_setup/exp/<run>/`)

| What | File |
|---|---|
| Flags | `config.yaml` (`command`, `args`, `git_sha`/`git_dirty`; `git.diff` if dirty) |
| Per epoch (selection sets, half A) | `metrics.jsonl` and `logs/train.log` (`Epoch N/M TrainLoss=… DevLoss=… LR=…`, `Epoch N LocalWF1=… F1clean=… F1noisy=… WF1oracle=… EERnoisy=…`; b3wave adds `CE= JS= Emb=` and an aug-count line) |
| Startup header (what really ran) | stdout log `SLS_setup/<track>_<ts>.log`. Grep the header lines and ignore the tqdm progress bars. |
| Merged epochs | `top5_by_wf1/merge_list.txt` |
| Readouts on half B (17 report sets) | `local_eval/merge_top5/` (the top-5 weight merge) and `local_eval/epoch_<N>/` (final EMA checkpoint), each holding `metrics.json` (`half_B.wf1.{wf1,f1_clean,f1_noisy,wf1_oracle,eer_noisy}`, `half_B.sets.<set>.{f1,eer,recall_spoof,recall_bona,thr_oracle,ece}`, `breakdowns`), `probe_report.json`, `scores/<set>.txt` (+ `.logits`) |
| Breakdown tables (family / snr / codec / src / lang / aec) | `local_eval/<tag>.log` |
| Comparisons | `compare_vs_<label>.txt`, `compare_all_readouts.txt`, `ci_vs_<label>.txt` + `compare_ci_vs_<label>.json` (paired bootstrap on half B) |
| Exploop verdict | `exploop/<wave>/iterNN/verdict.json` (Δ, CI, clean drop, decision, pre-registered verdict) and `design.result.json` (hypothesis, predicted signature, falsifier) |
| Registry | `docs/experiment_log.csv` |

Use `python3 -I` for JSON/CSV parsing. A one-threshold sweep over the half-B rows of
`scores/<set>.txt` is allowed; those are local sets. Progress scores (`model_merging_*/scores.txt`)
are not.

## Report structure

Model it on `docs/experiments-exploration/20261009_175355_B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234.md`.

- **Title:** `# Experiment exploration NN: <label> (`<track>`) - <one-line role>`.
- **Provenance blockquote:**
  - who wrote it and when;
  - the source files ("nothing quoted from memory");
  - previous reports;
  - readout definitions (top-5 merge vs final-epoch EMA, half_B).
- **`## 1. Setup`:**
  - the ONE changed flag, with a flag-diff table against the reference (from both config.yaml files);
  - run time from train.log timestamps;
  - checkpoints kept or pruned;
  - merged epochs;
  - header lines proving the change took effect.
- **`## 2. Results vs references`:**
  - 2.1 all iterations of the wave plus the fixed comparators, both readouts: WF1@0.5, WF1@oracle,
    F1 clean/noisy, EER noisy, with the CI;
  - 2.2 per set: F1, EER, recall_spoof, recall_bona, thr_oracle;
  - 2.3 probes: level_flip_bona, the noise_only and sil_only bona → spoof share;
  - 2.4 the slices that moved most;
  - 2.5 a lever-response table (iteration, lever, Δ merge, Δ final, verdict).
- **`## 3. Per-epoch behaviour`:**
  - a table against the reference at the same epoch: mini WF1, noisy EER, train loss / acc, dev
    loss, LR;
  - where the best epoch is, and whether the curve is still rising;
  - whether the final epoch beats the merge, and which merged epochs drag it down.
- **`## 4. What went OK`** (evidence).
- **`## 5. What went wrong`** (evidence):
  - the oracle-gap decomposition: a global threshold sweep vs set-specific thresholds;
  - the share of noisy bona at p_spoof ≥ 0.99;
  - the worst slices for bona recall;
  - overfitting signs, such as dev loss rising while train acc goes above 99 %.
- **`## 6. Verdict on the pre-registered rule`:** quote verdict.json and the falsifier from
  design.result.json, and say whether the predicted signature appeared, set by set.
- **`## 7. Decision and next-experiment proposals`:**
  - the decision on the reference: the new best or not, per the supervisor's verdict;
  - ≥ 3 ranked proposals. Each is ONE flag change (or one coupled group) against the run that will
    be the reference, and gives the hypothesis, predicted signature, falsifier and the error it
    targets;
  - configuration changes first; a seed repeat goes last;
  - what the recommended full run would be if the wave stopped now.

## Wave summary (SUMMARY step)

Model it on `docs/experiments-exploration/20261008_232331_B2_wave_summary_and_full_run_recommendation.md`.

- **§1** Starting point and loop rules.
- **§2** Iteration table, from the supervisor's `wave_table.md`, plus the per-epoch mini-WF1 table.
- **§3** Lessons: what each lever did.
- **§4** Recommended full run:
  - exact flags;
  - 2–3 seeds;
  - the readout or checkpoint to submit;
  - pre- and post-launch checks: `bash -n`, `which ffmpeg`, an empty `pgrep`, an idle GPU, the
    epoch-1 sanity values.
- **§5** Ranked follow-ups.
- **§6** Artefacts (paths).

## Interactive use

`/exp-report <run> [--ref <eval dir>]`:
1. Resolve the run and its readouts. If a readout is missing, suggest `/post-train-eval` first.
2. Pick the references: the given one, otherwise the registry's current best.
3. Write `docs/experiments-exploration/<YYYYMMDD_HHMMSS>_<track>.md`, with the timestamp from
   `date +%Y%m%d_%H%M%S`.
4. Reply with the path, the headline table and the top proposal.

Don't commit anything.
