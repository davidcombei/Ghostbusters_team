# exploop: unattended training-exploration waves

`exploop` runs a wave of N one-factor training iterations end to end without a live Claude
session. It keeps going when the terminal closes, the SSH connection drops or Claude exits. After a
container restart you continue it with one command.

The background and the design rationale are in `docs/exploration_agent.md`.

## The pieces

| Piece | What it does | How it runs |
|---|---|---|
| `supervisor.py` | State machine: design → render launcher → (smoke) → train → eval → verdict → log → analyze → decide. Waits, polls, resumes, logs epochs. | one process per wave, `setsid nohup`, no LLM |
| `loopctl.sh` | init / plan / start / status / tail / note / pause / stop / resume | your shell, or the `exp-lead` agent |
| `.claude/agents/exp-designer.md` | DESIGN: next one-factor flag change, returned as JSON | `claude -p --agent exp-designer`, fresh context, spawned by the supervisor |
| `.claude/agents/exp-analyst.md` | ANALYZE / SUMMARY: the report plus ranked proposals | `claude -p --agent exp-analyst`, spawned by the supervisor |
| `.claude/agents/exp-lead.md` | interactive research lead: plans waves, starts / steers / wraps up | `claude --agent exp-lead` |
| `.claude/skills/exp-design`, `exp-report`, `exp-loop` | the protocol: one-factor rules, report format, wave control | loaded by the agents; `/exp-design`, `/exp-report`, `/exp-loop` interactively |
| `launcher_template.sh` | renders `scripts/run_<prefix><letter>.sh` from the design JSON | supervisor |
| `profiles/sls_local17.json` | the eval pipeline: the `/post-train-eval` steps + the final-epoch readout + bootstrap CI | supervisor |
| `claude_settings.json`, `guard_hook.py` | deny rules and a PreToolUse guard for the headless steps | `--settings` on every step |
| `schemas/*.json` | structured output of the steps | `--json-schema` |
| `view.py` | read-only snapshot of a wave (phase, live epoch progress, current epoch vs the best run, eval steps, running Claude step, events) as JSON or as a 2-line status bar | Claude Code's status line, the `/exploop` mod, `exp-lead` |
| `claude_mod/` | the `exploop-view` Claude Code mod: `/exploop` live pane + toasts / notifications | installed at local scope (see Watching a wave) |

## A wave's files: `exploop/<wave>/`

| File | What it holds |
|---|---|
| `wave.json` | spec: trainer, search space, lever queue, eval profile, objective, rules, budgets, start |
| `plan_snapshot.md` | the approved plan (agents read this, never `~/.claude/plans/`) |
| `user_notes.md` | your steering notes; the DESIGN step reads them first |
| `loop_state.json` | supervisor state (do not edit while it runs) |
| `STATUS.md` | live page: banner, best, iterations, per-epoch table with Δ vs the best, Claude sessions, commands |
| `events.jsonl` | every launch, epoch, verdict, block |
| `iterNN/` | `design.result.json`, `verdict.json`, `analyze.result.json`, step prompts and outputs, job logs |
| `logs/supervisor.out` | supervisor stdout |

Reports go to `wave.json:report_dir`, registry rows to `docs/experiment_log.csv`, and launchers to
`scripts/run_<prefix><letter>.sh`.

## Daily use

```bash
cd /root/repos/rtc-sdd/Ghostbusters_team
scripts/exploop/loopctl.sh plan   B3     # what runs next, commands expanded (changes nothing)
scripts/exploop/loopctl.sh start  B3     # background supervisor; safe to close the terminal
scripts/exploop/loopctl.sh status B3     # exit 0 running, 1 down, 2 blocked, 3 done
scripts/exploop/loopctl.sh tail   B3 50  # recent events (epochs with Δ, verdicts, ...)
scripts/exploop/loopctl.sh note   B3 "skip the sampler lever; try --specaug next"
scripts/exploop/loopctl.sh pause  B3     # no new GPU job; unpause to continue
scripts/exploop/loopctl.sh stop   B3     # supervisor exits at the next tick; a running training continues
scripts/exploop/loopctl.sh resume B3     # after BLOCKED / DOWN / a container restart
```

Keep `exploop/<wave>/STATUS.md` open in VS Code; it is rewritten every minute. To inspect a Claude
step, run `claude --resume <session id>` (the ids are listed in STATUS.md).

## Watching a wave from Claude Code

Both views read the wave through `view.py` and never touch the supervisor (no lock, no writes), so
they can be added or changed while a wave runs. They follow the wave with the newest heartbeat that
is not DONE; `EXPLOOP_WAVE=<wave>` or `/exploop <wave>` picks another one.

- **Status line**, in every Claude Code terminal session in this repo (`.claude/settings.local.json`,
  `refreshInterval` 10 s). It has two lines:
  ```
  ● B3 RUNNING · it 2/5 B3b · design › render › TRAIN › eval › verdict › log › analyze › decide
  ep 3/8 ▕████░░░░░░▏40% 2.86 b/s · ETA 6.1h · ep2 LocalWF1 91.20 (-0.44 vs B3a@2) ▃▆ · to beat: B3a@ep3 91.39 · best B3a half_B 93.06 · $5.65/150
  ```
  - In EVAL the second line shows the eval step running; in DESIGN / ANALYZE it shows the Claude
    step's last tool call.
  - BLOCKED, WAITING, PAUSED, a DOWN or stale supervisor and a container restart show in red or
    yellow, with the `loopctl.sh` fix.
- **`/exploop` pane** (the `exploop-view` mod in `claude_mod/`). It shows:
  - the supervisor's health and the spend;
  - the phase pipeline;
  - the current step: epoch progress and ETA, the eval checklist, or the Claude step;
  - a *current vs best* card: the best's name and half_B readouts, the LocalWF1 to beat at the epoch in
    progress, and per-epoch LocalWF1, F1noisy, EERnoisy and DevLoss, each with its Δ vs the best at the
    same epoch, plus both sparklines;
  - the iterations table and the recent events.

  New events become toasts (phase change, epoch with Δ vs the best, new best, iteration done, a dead
  trainer); BLOCKED, DONE and a supervisor going down also send a desktop notification. Use
  `/exploop close` to hide the pane and `/exploop auto` to follow the live wave again.
- **Snapshot JSON** for scripts and agents: `python3 scripts/exploop/view.py snapshot [--wave B3]`.

LocalWF1 is the per-epoch score on the dev mini sets, from train.log. half_B WF1 is the objective, and
it exists only after EVAL. The views label the two and never compare one with the other.

Install the mod (once per clone; it is then read from this folder, so after an edit run `/reload-plugins`):
```bash
claude plugin marketplace add ./scripts/exploop/claude_mod
claude plugin install exploop-view@exploop --scope local
```

## A new wave

Interactively, start `claude --agent exp-lead` (or any session) and type
`/exp-loop new B4 "from @SLS_setup/exp/<best run> explore ..."`. The lead does the following:
1. drafts `wave.json` (copying the structure of `exploop/B3/wave.json`) and `plan_snapshot.md`;
2. asks only what it can't infer;
3. writes the files once you approve, then runs `loopctl.sh init B4` and `plan`.

You start the wave.

A wave only changes CLI flags. New levers need code first: an interactive phase 0 in plan mode, in
a separate module whose new flags default to the old behaviour (as `SLS_setup/b3wave` did). Then
draft the wave over those flags. Another trainer or eval pipeline needs only a different
`trainer` block and eval profile.

## Guarantees and limits

- **What survives what:**
  - closing the terminal, dropping SSH, a Claude crash or exit: the supervisor runs on;
  - `kill -9` of the supervisor: the training continues, and `loopctl resume` re-attaches (pid +
    `/proc` start time);
  - a container restart kills everything; `resume` reinstalls ffmpeg if it is missing and continues
    the training with `--resume`.
- **GPU:**
  - the supervisor launches a GPU job only when the GPU is idle (memory < 1 GB and no
    trainer or eval process), checked twice 10 minutes apart before a training run;
  - Claude steps cannot launch GPU jobs: deny rules, the guard hook, and a before/after diff of
    protected files.
- **What the supervisor never does:**
  - delete run outputs (the trainer's own top-7 pruning is the only deletion);
  - kill a process it did not start;
  - commit;
  - upload to Codabench;
  - read or analyse progress-split scores (it only checks that `submission.zip` was built).
- **When it blocks:**
  - any failure past its retry cap leaves the wave **BLOCKED**, and the supervisor exits;
  - STATUS.md says why and what to do; after fixing, run `loopctl resume <wave> [--reset-attempts]`;
  - a busy GPU or low disk is WAITING, not BLOCKED.
- **The early check replaces per-iteration smoke runs** (`wave.json: "smoke": "off"`):
  - in its first ~10 minutes the real run must print the expected header lines and must not show
    a Traceback, an OOM or codec `failed (` lines;
  - its projected time must fit `max_run_hours`;
  - otherwise the supervisor stops its own run and the designer redesigns;
  - `"smoke": "first_use"` or `"always"` adds a 2 % one-epoch smoke run.

## Tests

```bash
PY=/root/.conda/envs/rtc-sdd/bin/python
$PY -m unittest discover -s scripts/exploop/tests -p "test_lib.py"     # seconds: validator, parsers, verdicts, profile
$PY -m unittest scripts.exploop.tests.test_integration -v              # ~3 min CPU: full fake wave + kill / block / restart tests
$PY -m unittest scripts.exploop.tests.test_view -v                     # seconds: view.py snapshot + status line, read-only on exploop/B3
claude plugin test scripts/exploop/claude_mod                          # the /exploop mod: baseline, toasts, notifications, pane
```

The integration tests drive the real supervisor with a fake trainer, eval and claude
(`tests/fake_*.py`) on a scratch copy of the registry. Set `EXPLOOP_TEST_DIR=<dir>` to keep the
scratch waves.
