# The exploration agent (exploop): why it exists, what it encodes, how to use it

Written 2026-10-10 from the project's Claude Code transcripts (25 sessions, 2026-09-17 to 2026-10-10),
the plans in `~/.claude/plans/` and the run directories. The operator guide is
`scripts/exploop/README.md`.

## 1. What happened in the two exploration loops

| | B2 wave (session e43ae511) | B3 wave (session 0863b758) |
|---|---|---|
| Brief | B0 vs B1 post-mortem → "at most 5 iterations … only training parameters … `run_B1.sh`-style script … `/post-train-eval` … compare with the best" | "5 iterations starting from B2f … only CLI arguments … a new agent per iteration for the shell script … log … a markdown per iteration … display every epoch" |
| Plan approval | 5 plan rejections over ~13 h, each adding a constraint | 4 questions answered, approved first time |
| How it ran | one live interactive session for ~25 h. Monitor was re-armed every 30 min (47 times), with a background `kill -0` watcher and a fresh analysis subagent per run | the same, but the per-epoch watcher was a background `until` loop |
| Outcome | 5 iterations B2a→B2e; half_B merge 84.95 → 92.34; wave summary + `run_F1.sh` | **died after launching B3a.** The watcher grepped `it/s` while the trainer prints `batch/s`, so it never fired. At 01:12 UTC the terminal / SSH client dropped and three Claude sessions ended in the same second. The nohup trainer finished all 8 epochs at 05:41, but nothing evaluated, logged or continued it. |
| Friction | the plan file was later overwritten by another plan (plan-path reuse); Claude co-scheduled twice on a GPU the user had said to wait for; a subagent ranked a seed repeat first; the `--ce_weights` direction was wrong in the rule table | everything after the launch was lost |

**Lesson:** an in-session loop is only as durable as the terminal. The B2 wave succeeded because
nothing interrupted a 25-hour session, not because the method could survive an interruption.

## 2. How the experiment owner prompts (what the agent must absorb)

**Shape of a request:**
- context ("We have implemented …", "Based on the current best results …");
- `@/root/…/SLS_setup/exp/<run>` and `@docs/...` references;
- a target number (Codabench 92.49, "around 90% at least");
- the verb chain **investigate → find the gaps / misconfigurations → propose / come up with a plan**;
- where the deliverable goes;
- often "ask before starting".

**Style:**
- terse follow-ups ("Ask again", "Where are the changes ?");
- conceptual checks ("Is it worth?");
- persona prompts ("act as a senior machine learning researcher");
- frequent typos. Read the intent; never ask for a restatement.

**How constraints arrive:**
- incrementally, as plan-rejection feedback (about 35 rejections over all sessions);
- each rejection is a durable rule, not a one-off.

**What had to be re-explained (now encoded once):**

| Recurring instruction (examples of the user's words) | Where it lives now |
|---|---|
| "only make changes to the training parameters … no code changes at all" | exp-design skill, exp-designer agent, `wave.json search_space` + the supervisor's one-factor validator, deny rules / guard hook (no edits under `SLS_setup/`, `scripts/`) |
| one factor per iteration, "based on what each previous experiment produced and behaved" | exp-design procedure (largest remaining error → lever), the analyst's ranked proposals passed to the designer |
| prefer configuration changes over seed repeats | exp-design anti-patterns, the supervisor re-ranks proposals (config first), validator `kind=seed_repeat` |
| "not run for more than 10 epochs", "keep only top 7 checkpoints" | validator (`max_epochs`, `--keep_topk_by_wf1 7`); the supervisor forces `--keep_epochs <final>` |
| "a new bash script for each experiment as the run_B1.sh" | `launcher_template.sh` (same shape as `run_B3a.sh`), rendered and argv-checked by the supervisor |
| "run the correct claude skill post-train-eval" | `profiles/sls_local17.json` = the skill's steps 1–5 + final-epoch readout + CI (kept in sync; the skill says so) |
| "compare with the baseline or the best new experiment" | VERDICT vs the current best on both readouts; best updated in DECIDE |
| "spawn a new subagent that will analyze … document everything inside docs/experiments-exploration … proper name timestamp" | ANALYZE step = fresh `exp-analyst` per run, report `<YYYYMMDD_HHMMSS>_<track>.md`, exp-report skill |
| "keep logging the experiments correctly in experiment_log.csv" | LOG phase: `log_experiment.py` (dry run first, WARNING → block), two rows (merge + final), `b3_log_ci.py` |
| "after each epoch display … the result so far" | STATUS.md per-epoch table with Δ vs the best at the same epoch; `loopctl tail`; the status line and the `/exploop` pane / epoch toasts (`view.py`) |
| who launches runs ("I will do that manually" / "you will loop through the experiments") | you start a wave (`loopctl start`); inside it the supervisor launches; the exp-lead agent starts a wave only on request |
| wait for a free GPU | the supervisor's GPU gate (idle twice, 10 min apart); Claude steps cannot launch GPU jobs |
| "keep the same behaviour without those new flags", separate B3 module, no new worktree | exp-lead standing rules: code changes only in an interactive phase 0 |
| never type numbers, never use progress scores | every agent and skill; deny rule + guard on progress score files |
| resilience to container restarts (memory note) | `loopctl resume`: ffmpeg reinstall + `--resume`; job re-identification by pid + `/proc` start time |

## 3. Architecture

```
loopctl.sh start <wave> ──► setsid nohup supervisor.py   (no LLM; state on disk; restart-safe)
  per iteration: DESIGN ─► RENDER ─► [SMOKE] ─► WAIT_GPU ─► TRAIN (+early check) ─► EVAL ─► VERDICT ─► LOG ─► ANALYZE ─► DECIDE
                 claude -p                                                                                   claude -p
                 exp-designer                                                                                 exp-analyst
  end: SUMMARY (exp-analyst) ─► DONE ─► /exp-loop wrapup (interactive: memory, changed files)
```

- **The deterministic work runs without an LLM.** Waiting, launching, epoch parsing, resuming,
  evaluation, verdicts and registry rows are all done by the supervisor. Waiting 8 hours costs no
  tokens, and a 10-minute tool timeout can't cut an evaluation short.
- **Judgement steps are short headless Claude runs with a fresh context**, which matches the
  "new agent per run" brief. They return schema-checked JSON, run under deny rules plus a guard
  hook with no MCP servers, and the supervisor diffs protected files before and after each step.
  Each step has a USD cap and a timeout, and the wave has an overall cap.
- **Every phase checks its outputs on disk.** A supervisor killed at any point picks up where it
  stopped.

## 4. How to use it

| You want to… | Do |
|---|---|
| plan a new wave from a terse brief | `claude --agent exp-lead`, then `/exp-loop new B4 "from @SLS_setup/exp/<run> explore …"` |
| run it | `scripts/exploop/loopctl.sh start B4`, then close the laptop |
| watch it | the Claude Code status line (phase, epoch progress, current epoch vs the best, best, spend) and `/exploop` (live pane + toasts), both read-only through `scripts/exploop/view.py`; or `loopctl.sh status B4`, `loopctl.sh tail B4`, `exploop/B4/STATUS.md` |
| steer it | `loopctl.sh note B4 "try --specaug next"`, `pause` / `unpause`, `stop` |
| recover after a container restart or a block | `loopctl.sh status B4` (reason + hint), then `loopctl.sh resume B4` |
| analyse a hand-launched run with the same rigour | `/exp-report <run> --ref <eval dir>` |
| decide the next run by hand | `/exp-design <run or wave>` |
| finish | `/exp-loop wrapup B4`; upload the zip to Codabench yourself |

**Other kinds of research:** a wave is declarative. Change the `trainer` block (entry point, flag
sources, preflight), the `search_space`, the eval profile (`scripts/exploop/profiles/*.json`), the
`objective` (metrics key, direction) and the `rules`; the supervisor has no B3-specific code.
Research that needs new code follows the existing two-step pattern:
1. an interactive phase 0 that adds the new levers as default-off flags in their own module;
2. a wave over those flags.

## 5. First wave: B3, adopted mid-flight

`exploop/B3/` adopts the B3 plan (copied to `plan_snapshot.md`). The best is B2f (merge readout
`local_eval/merge_top5`). Iteration 1 is the already-trained B3a, so the wave begins with B3a's
EVAL; iterations 2–5 are designed by the loop. The per-iteration smoke run is off; the early check
replaces it.

```bash
scripts/exploop/loopctl.sh plan B3      # shows B3a's eval commands
scripts/exploop/loopctl.sh start B3
```
