---
name: log-experiment
description: Add a training/eval run's results as a row in docs/experiment_log.csv (the team's experiment registry), with owner, training git SHA, seed, GPU-hours, half_B WF1 and the decision. Use when the user says "log this run", "add the run to the experiment log", "register experiment", or invokes /log-experiment [run_dir].
---

# Log an experiment run

The row is built by `scripts/log_experiment.py`, which reads the run's local_eval `metrics.json`, `config.yaml` and `logs/train.log`. **Never type metric values into the CSV by hand**; every number comes from the script. Use the `rtc-sdd` env: `PY=/root/.conda/envs/rtc-sdd/bin/python`. Run commands from the repo root.

## 1. Resolve the run

- Use the path given as the argument (accept either `SLS_setup/exp/<run>` or `exp/<run>`). Otherwise take the newest `SLS_setup/exp/*/` that has a `config.yaml`, and confirm it with the user.
- List `<run>/local_eval/*/metrics.json` (ignore `local_eval/ckpts/`). If there are several tags (e.g. `best_by_wf1`, `merge_top5`), ask which one to log. Each tag is its own row.

## 2. If there is no metrics.json, evaluate first

Do not log without one. Show the user this command and run it **only after they confirm**, because it is a long GPU job. `--ssl_name` and `--n_layers` must match the run's `config.yaml` args.

```bash
cd SLS_setup
NAMES="dev_online_clean sim_matched_v1 sim_heldout_v1 sim_echo_v1 probe_orig probe_sil_only \
probe_noise_only probe_trimmed probe_padded probe_gain_up probe_gain_down probe_gain_m20 \
probe_gain_m6 probe_gain_p6 probe_gain_p10_limit probe_agc_dynaudnorm probe_loudnorm"
/root/.conda/envs/rtc-sdd/bin/python ../scripts/local_eval.py \
    --model_path exp/<run>/ckpt/best_by_wf1.pth --ssl_name <ssl> --n_layers <n> \
    --out exp/<run>/local_eval/best_by_wf1 --names $NAMES
```

Always pass the 17 `--names`. The default set list breaks comparability with B0.

## 3. Collect what the code can't know

Ask only for the fields the user has not already given. Ask them together in one AskUserQuestion call where possible.

- **id**: suggest one in the house style, derived from the track name, e.g. `B0-level-screen`, `L1-level-screen`. It must be unique in the CSV.
- **change_vs_ref**: the one factor changed relative to the reference, e.g. `B0 + level aug (A2.7a)`. Also ask for the reference row's id, passed as `--ref`, so the script prints the deltas.
- **decision**: `promote`, `drop`, `revisit`, or blank while pending. This is the user's call. Show them the Δ half_B WF1 and Δ clean F1 from the dry run, together with the decision rule the script prints.
- **notes**: free text, optional.
- **progress_score**: only if the model was submitted to the progress leaderboard.
- **ci_low / ci_high**: leave these blank unless the user hands over the numbers from a paired-bootstrap run. Never estimate them.

`owner` (git email handle), `git_sha` (training commit, `+dirty` if trained with local changes), `seed`, `gpu_h` and `half_b_wf1` are filled in automatically.

## 4. Dry run, confirm, write

```bash
$PY scripts/log_experiment.py --run_dir SLS_setup/exp/<run> --tag <tag> \
    --id <id> --change_vs_ref "<change>" --ref <ref_id> --decision <decision> --notes "<notes>" --dry_run
```

- Show the user the printed row and the deltas against the reference.
- Pass on any `WARNING` lines. Missing report sets mean the row is not comparable with other rows; say so.
- Once the user confirms, rerun the same command without `--dry_run`.
- If the script refuses because the id already exists, ask whether to overwrite the row (`--replace`) or choose a new id.

## 5. Finish

Report the row as written. Tell the user that `docs/experiment_log.csv` changed, and that they should commit it so the rest of the team can see the row. Do not commit unless asked.

## Notes

- `wf1@0.5` / `wf1@oracle` are the **full** WF1, kept for continuity with older rows. `half_b_wf1` is the primary readout (docs/dev_splits.md §5). Compare runs on `half_b_wf1`.
- `scripts/local_eval.py --registry docs/experiment_log.csv --id <id>` writes the same kind of row at eval time. Use this skill to log a run that was evaluated without `--registry`, or to fill in the decision afterwards (with `--replace`).
- Run `$PY scripts/log_experiment.py --backfill` after the CSV has been edited by hand or after new columns are added. It only fills empty `seed`, `half_b_wf1` and `gpu_h` cells.
