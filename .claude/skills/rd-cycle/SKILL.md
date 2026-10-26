---
name: rd-cycle
description: Create, start, watch, gate, review and hand off an rdloop research & development cycle (scripts/rdloop/) - a background team (auditor, scout, architect, critic, engineer + the supervisor's tests) that turns a brief into 2-4 default-off levers in a new isolated module, regression- and smoke-tested, ready for an exploop wave. Subcommands - new <C> "<brief>" [--auto-gate], status, approve, review, note, reopen, gate, park, resume, handoff. Invoke as /rd-cycle <subcommand> <C> [...].
disable-model-invocation: true
---

# rdloop: research & development cycles

Run everything from the repo root `/root/repos/rtc-sdd/Ghostbusters_team`. The control script is
`scripts/rdloop/rdctl.sh <cmd> <C>`; the guide with use cases and the workbook is
`docs/research_agent.md`. A cycle lives in `rdloop/<C>/`:

| File | What it is |
|---|---|
| `cycle.json` / `brief.md` | the spec and the owner's brief |
| `user_notes.md` | steering notes, read first by every judgement step |
| `state.json` / `STATUS.md` / `events.jsonl` | supervisor state (never edit while it runs), status page, event log |
| `01_audit.md`, `02_sota.md` (+ `.json`) | gap audit, SOTA idea cards |
| `debate/` | `proposal_vN`, `critique_vN`, `lint_vN.json`, `ledger.json` (objections) |
| `plan_agreed.md` / `.json`, `approval.json` | the plan at the gate, the owner's approval |
| `impl/`, `review/` | engineer reports + progress, code reviews + diffs |
| `tests/` | `static.json`, `gpu.json` (regression + smoke), `compat.json`, `facts.json` |
| `handoff/wave_seed.json` | what `/exp-loop new <W> --from-research <C>` merges into a wave |
| `FINAL_SUMMARY.md` | the owner-facing summary (also copied to `docs/research/<C>/`) |
| `isolation/` | sha256 manifest of every file outside the scope + frozen copy of untracked files |

Cycle ids are the target wave (`B4` -> module `SLS_setup/b4wave/`, scripts `scripts/b4_*`).

## new `<C>` "`<brief>`" [--auto-gate] [--levers-max N] [--focus ...] [--exclude ...]

1. **Resolve the defaults** (ask only what can't be inferred, in one AskUserQuestion round):
   - base = the best of the latest exploop wave (`exploop/*/loop_state.json` `best`); its final readout
     and checkpoint must exist;
   - `wait_for_waves` = waves not DONE (the base must be final before the audit);
   - module `b<N>wave` must not exist yet; prefix `b<N>_`;
   - focus / exclude from the brief (e.g. "augmentation only" -> `--focus augmentation`);
   - gate: owner unless the brief says automatic (`--auto-gate`).
2. **Show a summary**: base, module, waits for, levers min-max, gate mode, budget (default $150), and
   the brief as it will be written. Write only after the owner approves:
   `scripts/rdloop/rdctl.sh new <C> --brief "<brief>" [--auto-gate] [--from-wave W] [--levers-max N] [--focus ..] [--exclude ..]`
3. `rdctl.sh init <C>` (manifest + frozen copy; nothing runs) and `rdctl.sh plan <C>`.
4. Print `scripts/rdloop/rdctl.sh start <C>`. **Do not start** unless the owner says so in this session.

## status `<C>`

`rdctl.sh status <C>` (exit 0 running, 1 down, 2 blocked / needs you, 3 waiting for your review) and
`python3 scripts/rdloop/view.py snapshot --cycle <C>`. Summarise from the JSON (never type numbers):
phase; who is running and what it is doing (`team[].last_tool/last_target/last_text`); debate round +
open objections; lever states; tests; blockers; cost. Live view: `/rdloop <C>` and the status line.

## approve `<C>` (the plan gate, status AWAITING_APPROVAL)

1. Read `rdloop/<C>/plan_agreed.md` (+ `.json`) and `debate/ledger.json`.
2. Present: the lever table (flags, gap, sources, cost, smoke estimate, contested / needs-install),
   the debate outcome (rounds, objections raised and how they were resolved), what was dropped.
3. AskUserQuestion: **approve all agreed levers** / **approve a subset** (list them) / **send back**
   (collect the note). Then:
   - approve: `rdctl.sh approve <C>` or `rdctl.sh approve <C> --levers a,b`;
   - send back: `rdctl.sh reopen <C> --at propose "<note>"` (the note goes to user_notes.md).
   Approving a lever that needs a package checks the import again; install it first.

## review `<C>` (the end, status AWAITING_USER)

1. Read `FINAL_SUMMARY.md`, `tests/facts.json`, `tests/gpu.json`, `handoff/wave_seed.json`;
   `rdctl.sh verify <C>` must say VERIFIED (the module is the code that passed).
2. Present: ideas implemented and why (gaps, sources), regression (old scores reproduced, levers-off
   equivalence), smoke per lever (time factor, peak VRAM, full-run ETA), dropped ideas, recommended
   wave order and hours, risks, cost.
3. AskUserQuestion: **start the exploration wave** / **request changes** / **park**:
   - start: follow `.claude/skills/exp-loop/SKILL.md` -> `new <W> --from-research <C>` (it drafts the
     wave from the seed and asks for approval; the wave still starts only on the owner's word), then
     `rdctl.sh handed-off <C> <W>`;
   - changes: `rdctl.sh note <C> "..."` then `rdctl.sh reopen <C> --at implement` (code changes) or
     `--at propose` (plan changes; re-debate and a new gate);
   - park: `rdctl.sh park <C>`.

## note / gate / reopen / pause / stop / resume / park

- `note "<text>"`: highest-priority steering for every later step.
- `gate <C> owner|auto`: switch the plan gate (applied by a running supervisor at its next tick).
- `reopen <C> --at propose|implement ["note"]`: back to the debate (new gate) or to a FIX round of the
  code; restarts the supervisor. Refused at implement while an exploop wave uses the module.
- `pause` / `unpause`: no new GPU job; `stop`: the supervisor exits, jobs continue.
- `resume [--reset-attempts] [--accept-manifest]`: after BLOCKED / NEEDS_USER / a container restart.
  Read the blocker and hint first. `--accept-manifest` only when the owner confirms the flagged
  files were changed by them.

## Rules

- The owner starts cycles and waves; never commit; never type metric numbers.
- One cycle at a time per module; the module is frozen after handoff while its wave runs.
