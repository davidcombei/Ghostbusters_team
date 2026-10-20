---
name: exp-loop
description: Create, start, monitor, steer and wrap up an unattended training-exploration wave run by the exploop supervisor (scripts/exploop/), which survives closed terminals, dropped SSH and Claude exits. Subcommands - new <wave> "<brief>", init, plan, start, status, tail, note, pause, unpause, stop, resume, wrapup. Invoke as /exp-loop <subcommand> <wave> [...].
disable-model-invocation: true
---

# exploop: unattended exploration waves

Run everything from the repo root, `/root/repos/rtc-sdd/Ghostbusters_team`. The control script is
`scripts/exploop/loopctl.sh <cmd> <wave>`. A wave lives in `exploop/<wave>/`:

| File | What it is |
|---|---|
| `wave.json` | the spec |
| `plan_snapshot.md` | the approved plan |
| `user_notes.md` | steering notes |
| `loop_state.json` | supervisor state; never edit it while a supervisor runs |
| `STATUS.md` | live status page |
| `events.jsonl` | the event log |
| `iterNN/` | prompts, step outputs, verdicts |

The supervisor (`scripts/exploop/supervisor.py`) waits, trains, evaluates, logs and resumes without
an LLM. It calls `claude -p --agent exp-designer` and `claude -p --agent exp-analyst` for the
judgement steps. The operator guide is `scripts/exploop/README.md`.

## new `<wave>` "`<brief>`" (+ @run references)

Draft a wave from a terse brief. Ask only what can't be inferred, in one AskUserQuestion round.

1. **Resolve the starting best:**
   - the `@run` or the registry's current best (`docs/experiment_log.csv` plus the latest wave's
     `loop_state.json`);
   - its readouts `local_eval/merge_top5` and `local_eval/epoch_<N>` must have `metrics.json`. If
     they don't, offer `/post-train-eval` first;
   - its registry ids.
2. **Copy the structure of `exploop/B3/wave.json`** and fill:
   - `wave`, `track_prefix`, `letters`, `max_iterations` (default 5), `report_dir`
     (`docs/experiments-exploration/<wave lower>-experiments`);
   - `trainer`: entry (`["-m", "b3wave.train"]` or `["main_train.py"]`), `flag_sources`,
     `preflight`;
   - `search_space`: the flags the brief puts in scope, plus coupled groups;
   - `fixed`: the data, dev protocol, eval sets, backbone, MUSAN pool and loudness settings of the
     best;
   - `forbidden`: always `--out_path --train_subset_frac --model_path --resume --label_smoothing
     --local_eval_names`;
   - `lever_queue` and `lever_notes`: from the brief, the latest wave summary §5 and the memory
     notes;
   - `eval`: the profile (`sls_local17`), names (the 17 report sets), `fixed_comparators`;
   - `objective` and `rules`: default to B3's (best = merge `half_B.wf1.wf1` strictly higher;
     ±0.5 band; promote needs CI low > 0 and clean drop ≤ 0.5);
   - the Claude budget;
   - `start.best`. Use `start.adopt` only for an already-trained run that belongs to the wave.
3. **Write `plan_snapshot.md`:** context, base command, the lever table with its targets, the rules,
   and the per-iteration loop (as in `exploop/B3/plan_snapshot.md`).
4. **Show a short summary:** base, levers, rules, budget, expected hours. Write the files only after
   the user approves. Then run `loopctl.sh init <wave>` and `loopctl.sh plan <wave>`, and print
   `scripts/exploop/loopctl.sh start <wave>`.
5. **Do not start the wave** unless the user says so in this session.

If the brief needs code that does not exist yet (a new lever), stop. Propose an rdloop research
cycle instead (`/rd-cycle new <C> "<brief>"`, run by `claude --agent research-lead`): it builds the
new module with default-off flags, regression- and smoke-tests it, and hands it back here through
`new <wave> --from-research <C>`. An interactive phase 0 in plan mode is still possible when the
owner asks for it.

### new `<wave>` --from-research `<C>` (levers built by an rdloop cycle)

1. **Check the cycle:** `scripts/rdloop/rdctl.sh status <C>` must show `AWAITING_USER` (regression
   and smoke passed), and `scripts/rdloop/rdctl.sh verify <C>` must print VERIFIED (the module is
   exactly the code that passed). Otherwise stop and say why.
2. **Read** `rdloop/<C>/handoff/wave_seed.json`, `rdloop/<C>/FINAL_SUMMARY.md` and
   `rdloop/<C>/plan_agreed.md`.
3. **Draft `wave.json` from the latest wave's structure plus the seed:**
   - keep `objective`, `rules`, `gpu` (timeouts, debounce), `disk_gb`, `timeouts`, `retries`,
     `integrity` and `claude` from the latest wave, and apply step 2 of `new` for the rest;
   - `trainer.entry` / `flag_sources` from the seed (`-m <module>.train`);
   - `search_space.allowed` += `allowed_add`, `coupled` += `coupled_add`;
   - `lever_queue` = the seed's queue in the summary's recommended order (keep each lever's
     `expect`, `hypothesis`, `predicted_signature`, `falsifier`); `lever_notes` += the seed's;
   - `early_check.expect` += `expect_add` (the `^<module>: ` header line);
   - `gpu.busy_regex` += `busy_regex_add`; `eval.profile` from the seed; `"smoke": "off"` (rdloop
     already smoke-tested every lever);
   - `start.best` from the seed (the cycle's base; if a newer best exists, say so and ask);
   - add `"_research": <seed._research>` so the wave records which cycle and module sha it runs.
   `rdloop/<C>/handoff/dryrun_wave/wave.json` is the supervisor's own merge of the same inputs and
   already passed `supervisor.py init` + `plan`; diff against it.
4. **`plan_snapshot.md`:** context (the cycle, the module), the lever table from the summary, the
   rules, the per-iteration loop. Then continue with steps 4-5 of `new` (show, write on approval,
   init, plan, print the start command, do not start unasked).
5. After init: `scripts/rdloop/rdctl.sh handed-off <C> <wave>`. The module is frozen while the wave
   uses it; changes go through a new cycle.

## start / status / tail / plan

- **`start`:** `loopctl.sh start <wave>`, only on the user's explicit request. It runs under
  `setsid nohup`, so closing the terminal or Claude does not stop it.
- **`status`:** `loopctl.sh status <wave>`. Exit codes: 0 running, 1 down, 2 blocked, 3 done. Then
  summarise `exploop/<wave>/STATUS.md`:
  - phase and current run;
  - the last epoch rows with Δ against the best at the same epoch;
  - best so far, blockers, Claude spend.
- **`tail [n]`:** recent events.
- **`plan`:** what runs next, with the commands expanded. It changes nothing.

## note / pause / unpause / stop / resume

- **`note "<text>"`:** appends to `user_notes.md`. The next DESIGN step treats it as the highest
  priority. Use it for "try X next", "don't touch the sampler", "stop after this run".
- **`pause` / `unpause`:** no new GPU job starts; running jobs continue.
- **`stop`:** the supervisor exits at its next tick; a running training continues.
- **`resume [--reset-attempts]`:** after BLOCKED, DOWN or a container restart:
  1. read the blocker and its hint in `status` first;
  2. fix the cause (flags only, or the environment);
  3. then resume.
  
  After a container restart, the supervisor reinstalls ffmpeg if it is missing and continues the
  killed training with `--resume`.

## wrapup `<wave>` (after DONE; interactive)

1. Read the wave summary and `wave_table.md`.
2. Update the memory notes: a `<wave>-wave-recipe` note in the style of `b2-wave-training-recipe`
   (outcome per lever, final best, recommended full run, pitfalls learned), and the index.
3. List the changed files (`git status --short`: `scripts/run_<prefix>*.sh`,
   `docs/experiment_log.csv`, the report dir, `exploop/<wave>/`) and suggest a commit. Don't commit
   unless asked.
4. Remind the user:
   - the best `submission.zip` path is in STATUS.md, and they upload it to Codabench themselves;
   - after the score comes back, `/log-experiment` adds `progress_score` to the row.
