# The rdloop research team: user guide and workbook

> Status: living document · Created 2026-10-11 · Owner: Ghostbusters team.
> Sibling of `docs/exploration_agent.md` (exploop). Operator details: `scripts/rdloop/README.md`.

## 1. What it is and when to use it

**exploop** screens trainer *flags* over code that already exists: one factor per iteration, unattended.
**rdloop** builds the code those flags need. It is the automated "phase 0" before a wave. A team of
agents:
1. audits the current best;
2. searches the state of the art for RTC speech-deepfake detection;
3. debates a plan of 2-4 small levers (architect vs critic) until they agree;
4. stops for your approval;
5. implements the plan as a new isolated module;
6. proves that the old experiments are untouched and that the new code runs, with a regression check
   on the old runs and a smoke session of 30 minutes or less;
7. hands you a summary and asks whether to start an exploop wave over the new levers.

Use rdloop when the next improvement needs **new code**: an architecture, augmentation, loss or
sampler change. Use exploop alone when the levers already exist as flags.

```
 you: brief ──► /rd-cycle new ──► rdctl start
                                     │
   WAIT for running waves ─► AUDIT ─► SCOUT ─► PROPOSE ⇄ CRITIQUE (≤ 3 rounds, LINT between)
   (auditor)   (scout, web)  (architect)  (critic)
                                     │
                         GATE: you approve (or auto) ◄── /rd-cycle approve
                                     │
   SCAFFOLD ─► IMPLEMENT core + 1 step per lever ─► STATIC ─► REVIEW ⇄ FIX
               (engineer, CPU only)                           (critic / engineer)
                                     │
   GPU session: regression on the old runs + smoke (≤ 30 min) ─► COMPAT (exploop dry run)
                                     │
   SUMMARY (auditor) ─► AWAITING_USER ◄── /rd-cycle review: start wave / changes / park
                                     │
              /exp-loop new <W> --from-research <C> ─► exploop wave (you start it)
```

A cycle takes about 3-6 h of agent time, plus any wait for the GPU, and costs roughly $80-150 of
Claude usage (cap: `claude.max_total_usd`, default $150). The research steps run on the CPU and the
API, so they can run while a wave is training. The GPU session waits for an idle GPU and yields to
exploop.

## 2. What each piece does

### 2.1 The runtime (`scripts/rdloop/`)

| Piece | What it does |
|---|---|
| `supervisor.py` | The state machine. One process per cycle, started by `rdctl.sh start` under `setsid nohup`, so it survives closing the terminal or SSH or Claude. It runs the phases below and spawns every agent as `claude -p --agent research-*` with a fresh context, a JSON schema, a budget, per-step permissions and the guard hook. Every phase re-checks its outputs on disk, so a killed supervisor continues where it stopped. |
| `rdctl.sh` | Your command line (section 3). |
| `rdlib.py` | Shared logic: the isolation manifest, LINT of proposals, the debate ledger and its convergence rule, transcript readers, the GPU yield rule, the score comparison, the wave seed, STATUS.md, per-step permission settings. |
| `guard_hook.py` | Pre-tool-use guard of every headless step, with four policies: readonly (auditor, architect), scout (web, no shell), critic (readonly plus web), engineer (writes only inside its module, CPU only, no training, installs or network, read-only git). |
| `schemas/` | What each agent must return. |
| `templates/cycle.json` | Defaults of a new cycle (section 2.5). |
| `view.py` | Read-only snapshot for the `/rdloop` pane and the status line. |
| `claude_mod/` | The `rdloop-view` mod, described in section 2.4. |
| `tests/` | Unit and end-to-end tests: a scratch repo with a fake claude, trainer and eval. |

### 2.2 The agents (`.claude/agents/`)

| Agent | Runs in | Reads | Writes | Returns |
|---|---|---|---|---|
| **research-lead** | your interactive session (`claude --agent research-lead`) | everything | cycle files via `rdctl` | talks to you: kickoff, status, plan gate, final review, handoff |
| **research-auditor** | AUDIT, SUMMARY | registry, waves, reports, readouts, roadmap, ledger | `01_audit.md`, `FINAL_SUMMARY.md` | gaps G1..Gn with evidence; final summary |
| **research-scout** | SCOUT | audit, literature, the web (allowlisted domains) | `02_sota.md` | ranked idea cards with fetched, quoted sources |
| **research-architect** | PROPOSE / REVISE | audit, idea cards, base code, critique, ledger | `debate/proposal_vN.md` | 2-4 lever specs (flags, default equivalence, hypothesis, signature, falsifier, cost, smoke, wave lever); answers every objection |
| **research-critic** | CRITIQUE, CODE REVIEW | proposal, code, reserve, rules, web (citation checks) | `debate/critique_vN.md`, `review/code_review_rN.md` | objections or findings with severity and evidence; alternatives |
| **research-engineer** | IMPLEMENT, FIX | approved plan, base code, progress | the module, `scripts/<prefix>*`, `impl/` | per-part report: files, flags, CPU check results, deviations, needs_user |

### 2.3 The skills (`.claude/skills/`)

None of these can be auto-invoked by a model. You run them as slash commands, and the agents read
them by path.

| Skill | When you use it | Arguments |
|---|---|---|
| `/rd-cycle` | control of a cycle | `new <C> "<brief>" [--auto-gate]`, `status`, `approve`, `review`, `note`, `reopen`, `gate`, `park`, `resume`, `handoff` |
| `/rd-audit` | "audit the current best", "what gaps are left", "what have we tried" | `[<run>]` |
| `/rd-scout` | "what does the literature say about X" | `"<topic>"` |
| `/rd-propose` | "turn these ideas into levers" | `<cycle or topic>` |
| `/rd-critique` | "grill this plan" or "review this module" | `<plan file or module>` |
| `/rd-implement` | the module contract, when you implement a lever by hand | `<cycle>` |

`exp-loop` gained `new <wave> --from-research <C>`, which drafts a wave from a cycle's
`handoff/wave_seed.json`.

### 2.4 Watching it

- **`/rdloop [cycle | auto | close]`**: the live pane. It shows:
  - a header with status, supervisor health, cost, gate mode and base;
  - the phase strip, e.g. `AUDIT › SCOUT › [DEBATE r2/3] › GATE › …`;
  - an **action banner** when you are needed, with the exact command;
  - the **Team** table: each agent's state, step, elapsed time and headline. For the running agent
    it also shows tool-call count, last tool and target, and its last sentence;
  - the debate rounds and open objections, the levers, the STATIC / GPU / COMPAT checks with the
    30-min smoke budget bar, and recent events.

  Hotkeys `1`-`6` open one agent's detail: recent tool calls, searches, fetched pages, files
  written, and the `claude --resume <session>` command. `d` opens the full debate, objections next to
  the architect's responses. `o` returns to the overview, `r` refreshes, `c` closes.
- **Toasts** for phase changes, debate rounds, approvals, implemented or dropped levers, failing
  checks. **Notifications** for "plan ready for you", "cycle done", NEEDS_USER, BLOCKED and
  supervisor DOWN.
- **Status line**: `scripts/statusline.sh` shows the exploop rows as before, plus one rdloop row
  while a cycle runs or needs you. Examples:
  - `rdloop B4 RUNNING · DEBATE r2/3 · critic 4m · 23 tools · last: WebFetch arxiv… · $41/150`
  - `rdloop B4 ⚠ AWAITING_APPROVAL · … · /rd-cycle approve B4`
- **Files**: `rdloop/<C>/STATUS.md` (the status page), `rdctl.sh tail <C>`, `rdloop/<C>/events.jsonl`,
  `rdloop/<C>/logs/supervisor.out`.

### 2.5 `cycle.json` (written by `rdctl.sh new`; edit before `init` if needed)

| Field | Default | Meaning |
|---|---|---|
| `cycle`, `module`, `script_prefix` | `B4`, `b4wave`, `b4_` | the target wave id, the new package, the scripts prefix |
| `base.from_wave` | the latest exploop wave | the base is that wave's `best`. You can also give `base.run_dir` (+ `readouts`) |
| `wait_for_waves` | waves not DONE at `new` | the audit starts after they finish, so the base is final |
| `focus` / `exclude` | `[]` | e.g. `augmentation`, `architecture`, `loss`, or a topic |
| `levers.min` / `levers.max` | 2 / 4 | levers per cycle |
| `limits` | +4 GiB VRAM, ×1.3 epoch time, 400 LOC per lever, 10 h per run | checked by LINT and by the smoke runs |
| `debate.max_rounds` | 3 | critique rounds before open blockers drop a lever and open majors make it contested |
| `gates.plan` | `owner` | `auto` = no stop after agreement (`--auto-gate`) |
| `fix.max_total`, `fix.max_review_rounds` | 6, 3 | fix rounds over the cycle, and review rounds in a row |
| `gpu` | idle < 1 GiB, 120 s debounce, yield to exploop | when the GPU session may start |
| `smoke` | 30 min, 1 % subset, 400 dev lines, peak ≤ 22 GiB | the smoke recipe and budget |
| `regression` | `dev_online_clean sim_matched_v1 sim_heldout_v1 probe_orig probe_noise_only`, tol 1e-4, base checks, equivalence micro-runs | old-run regression (`full: true` re-scores every set of the base's final readout, about 4 min). The sets need clean + matched + heldout, or `local_eval.py` can't print WF1 and exits 1. The two equivalence micro-runs drop `equiv_remove_flags` (`--use_rtc_aug --musan --use_level_aug`), because those augmenters are seeded with the worker pid and no two runs would match |
| `claude` | opus via `--agent`, $150 cap, per-step caps | budget and timeouts |
| `web.domains` | arXiv, ISCA, OpenReview, ACL, IEEE, GitHub, HF, … | the source allowlist for the scout and the critic |

### 2.6 What a cycle produces (`rdloop/<C>/`, plus copies in `docs/research/<C>/`)

| File | Content |
|---|---|
| `01_audit.md`, `audit.json` | the base, everything tried, error anatomy, gaps, constraints, exhausted directions |
| `02_sota.md`, `scout.json` | search log, landscape, idea cards (verified sources), rejected ideas |
| `debate/` | `proposal_vN`, `critique_vN`, `lint_vN.json`, `ledger.json` |
| `plan_agreed.md` / `.json`, `approval.json` | the plan at the gate and your approval |
| `SLS_setup/<module>/`, `scripts/<prefix>*` | the new code |
| `impl/`, `review/` | engineer reports, `progress.md`, `commands.log`, code reviews and the diffs reviewed |
| `tests/` | `static.json`, `gpu.json`, `compat.json`, `facts.json`, `regression/old_scores_compare.json`; smoke runs in `runs/` |
| `handoff/wave_seed.json`, `handoff/dryrun_wave/` | the wave seed and the exploop dry run |
| `FINAL_SUMMARY.md` | what you read at the end |
| `docs/research/lever_ledger.md`, `literature.md` | updated by the supervisor |

## 3. Use cases: what to type

All shell commands run from `/root/repos/rtc-sdd/Ghostbusters_team`. Slash commands go in a session
started with `claude --agent research-lead`.

| I want to… | Type |
|---|---|
| start a cycle with my approval of the plan | `/rd-cycle new B4 "<brief>"`, then "start it" (or `scripts/rdloop/rdctl.sh start B4`) |
| start a fully automatic cycle | `/rd-cycle new B4 "<brief>" --auto-gate` |
| restrict it to augmentation only | add `--focus augmentation` (or "augmentation only" in the brief) |
| keep architecture changes out | add `--exclude architecture` |
| fewer levers | add `--levers-max 2` |
| a different base run | edit `base` in `rdloop/<C>/cycle.json` before `init` (`{"run_dir": ..., "readouts": {...}}`) |
| see what it is doing | `/rdloop B4`, the status line, `/rd-cycle status B4`, or `scripts/rdloop/rdctl.sh status B4` |
| steer it ("don't use neural codecs") | `scripts/rdloop/rdctl.sh note B4 "don't use neural codecs"` |
| approve the plan | `/rd-cycle approve B4` (or `rdctl.sh approve B4`) |
| approve only some levers | `rdctl.sh approve B4 --levers aug_plc,head_asp` |
| send the plan back | `rdctl.sh reopen B4 --at propose "make the PLC lever cheaper; drop the loss lever"` |
| switch to automatic mid-cycle | `rdctl.sh gate B4 auto` |
| a lever needs a package (NEEDS_USER) | install it in the rtc-sdd env yourself, then `rdctl.sh resume B4` (at the gate, approve with `--levers` again) |
| BLOCKED, or after a container restart | `rdctl.sh status B4`, read the hint, fix the cause, then `rdctl.sh resume B4`. A flagged file you changed yourself: `--accept-manifest`. Steps that failed for a reason you fixed: `--reset-attempts` |
| change the code after the summary | `rdctl.sh note B4 "…"` then `rdctl.sh reopen B4 --at implement` |
| change the plan after the summary | `rdctl.sh reopen B4 --at propose "…"` (new debate and a new gate; implemented levers are kept when their spec did not change) |
| hand the levers to a wave | `/rd-cycle review B4`, choose "start the wave", which runs `/exp-loop new B4 --from-research B4`; then `scripts/exploop/loopctl.sh start B4` when you want it |
| shelve it | `rdctl.sh park B4` |
| prove the module is what was tested | `rdctl.sh verify B4` |
| use one skill alone | `/rd-audit`, `/rd-scout "codec robustness"`, `/rd-critique docs/next_steps_plan.md` |

## 4. Rules it enforces, and why

| Rule | Why | How |
|---|---|---|
| older experiments stay reproducible | a broken b3wave would invalidate every B3 comparison | sha256 manifest of every file outside the cycle's scope, checked after each engineer step; old-run scores re-computed and compared with the stored ones |
| new flags default to the base, bit for bit | the wave compares each lever against the current best; off must mean "the base" | rule in the plan (LINT), review checklist, `checks.py`, levers-off micro-run vs the base trainer |
| 2-4 small levers, one component each | each lever is screened as one factor | LINT plus the critic |
| no eval leakage, challenge rules | results must transfer to the hidden eval | critic blockers; the auditor lists the constraints; reserved ops are checked |
| verified sources only | no invented papers or numbers | the scout must WebFetch every source; verification comes from the transcript |
| the owner approves the plan | about 35 earlier plans were rejected; 5 minutes of reading saves hours | gate ON by default |
| no installs, network or commits from agents | your environment, your repo | guard hook, deny rules, `needs_user` |
| the GPU belongs to the waves | B3/B4 training must not OOM | CPU-only steps; the GPU session waits for an idle GPU and yields to exploop |
| budget | API cost | per-step caps and the cycle cap |

## 5. Troubleshooting

| Status | Meaning | Do |
|---|---|---|
| WAITING | waiting for a wave to finish, an idle GPU, an exploop Claude step, or disk | nothing; the reason is on the status line and in STATUS.md |
| AWAITING_APPROVAL | the plan gate | `/rd-cycle approve <C>` |
| NEEDS_USER | something only you can decide (a missing package, debate deadlock, repeated failing fixes) | read `next` in `rdctl.sh status`, then resume, reopen or park |
| BLOCKED | an error with a hint (step failed 3 times, manifest violation, environment: ffmpeg or the data mount) | fix, then `rdctl.sh resume <C> [--reset-attempts]` |
| supervisor DOWN | the process died (container restart?) | `rdctl.sh resume <C>`. Interrupted steps rerun without counting as failed attempts |
| AWAITING_USER | done | `/rd-cycle review <C>` |

Logs: `rdloop/<C>/logs/supervisor.out`; each step's prompt, settings, output and stderr are in
`rdloop/<C>/steps/`; the engineer's commands are in `impl/commands.log`; each test job has a `.log`
in `tests/`. Inspect a step's conversation with `claude --resume <session id>` (shown in the pane's
agent detail).

## 6. Workbook: starting a research iteration

### 6.1 Preconditions

- [ ] The latest wave is DONE (`scripts/exploop/loopctl.sh status B3` exits 3), or you are fine
  with the cycle waiting for it (`wait_for_waves`: the audit starts after the wave ends).
- [ ] `which ffmpeg` works and `ffmpeg -hide_banner -encoders | grep libopus` matches (smoke needs
  the codec stage).
- [ ] At least 50 GB free (`df -h /root`).
- [ ] Budget OK (default cap $150; change with `--budget`).
- [ ] No other cycle is running (`ls rdloop/`; `scripts/rdloop/rdctl.sh status <C>`).

### 6.2 Steps

1. **Open a session** (2 min):
   ```bash
   cd /root/repos/rtc-sdd/Ghostbusters_team && claude --agent research-lead
   ```
2. **Write the brief.** Paste, and adapt:
   ```text
   /rd-cycle new B4 "from the current best after the B3 wave, research SOTA for RTC speech-deepfake
   detection and propose 2-4 flag-gated levers (architecture or augmentation) in SLS_setup/b4wave that
   target the confident noisy-bona false alarms (English, g726/gsm/qq/wechat, SNR 0 dB) and unseen
   generators; small deltas, one component per lever"
   ```
   Tips:
   - name the gap, not the solution;
   - say "augmentation only" or "no architecture changes" to narrow the focus;
   - add `--auto-gate` only if you don't want to see the plan before code is written.
3. **Check the draft.** The lead shows base, module, waits-for, levers, gate and budget. Approve it.
   It then runs `rdctl.sh init B4` (hashes the repo, about 1 s) and `rdctl.sh plan B4`.
4. **Start it:** say "start it", or run `scripts/rdloop/rdctl.sh start B4`.
5. **Watch it:** `/rdloop B4` and the status line. Expect roughly:
   - WAIT, until B3 is done;
   - AUDIT, about 15-30 min;
   - SCOUT, about 30-60 min;
   - DEBATE, 2-3 rounds of about 15-30 min each.
6. **Plan gate.** You get a notification with "plan ready for you". Run `/rd-cycle approve B4`.
   - **Read:** `rdloop/B4/plan_agreed.md`, which has the lever table, contested levers and the
     debate record. Check that each lever names the gap it targets, cites verified sources, has a
     falsifier you believe, and that its cost and smoke estimate are plausible.
   - **Answer:** approve all; approve a subset (`--levers a,b`); or send it back with a note.
7. **While it implements and tests** (about 1-3 h, plus any GPU wait): IMPLEMENT, STATIC, REVIEW,
   FIX, GPU, COMPAT.
   - You are needed only for NEEDS_USER or BLOCKED, and the notification says what to do.
   - The GPU session is about 10 min of regression plus 30 min or less of smoke, and starts only
     when the GPU is free.
8. **Final review.** You get a "cycle done" notification. Run `/rd-cycle review B4`, then check:
   - `FINAL_SUMMARY.md`: what was built and why, and what was dropped;
   - regression: old scores reproduced, levers-off equivalence;
   - per-lever smoke: time factor, peak VRAM, projected run hours;
   - the recommended wave order;
   - `rdctl.sh verify B4` says VERIFIED.
9. **Decide:**
   - **Start the wave.** The lead runs `/exp-loop new B4 --from-research B4`, which drafts
     `exploop/B4/wave.json` and `plan_snapshot.md` and asks you to approve. Then:
     `scripts/exploop/loopctl.sh start B4`, when you want it.
   - **Changes:** `rdctl.sh reopen B4 --at implement` for code, or `--at propose` for the plan.
   - **Park:** `rdctl.sh park B4`.
10. **Wrap up.**
    - Commit `SLS_setup/b4wave/`, `scripts/b4_*`, `docs/research/` and the cycle's markdown
      yourself; the agents never commit.
    - After the wave finishes, fill in the `wave result` column of `docs/research/lever_ledger.md`
      at `/exp-loop wrapup`.
