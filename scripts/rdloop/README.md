# rdloop - research & development cycles (operator guide)

rdloop turns a brief into new, tested levers for the RTC-SDD trainer: a gap audit, a SOTA search, an
architect / critic debate, an owner gate, an isolated module implemented by an engineer agent, a
code review, regression on the old runs, a <= 30 min smoke session and an exploop dry run. The
output is a module (`SLS_setup/<bNwave>/`) plus `rdloop/<C>/handoff/wave_seed.json`, which
`/exp-loop new <W> --from-research <C>` turns into an exploration wave.

User guide with use cases and the step-by-step workbook: **`docs/research_agent.md`**.

## Files

| Path | Role |
|---|---|
| `supervisor.py` | the state machine, one process per cycle (`run`), plus `new init status plan tail approve gate reopen park unblock verify handed-off alive` |
| `rdlib.py` | phases, write scopes, isolation manifest, code fingerprint, LINT, debate ledger, transcripts, yield rule, score comparison, wave seed, STATUS.md, per-step settings (imports `scripts/exploop/exploop_lib.py` read-only) |
| `rdctl.sh` | the CLI (`setsid nohup` start, resume, note, pause, stop, ...) |
| `guard_hook.py` | PreToolUse hook of every headless step (python 3.8): readonly / scout / critic / engineer policies |
| `schemas/*.json` | structured output of each step (`--json-schema`) |
| `templates/cycle.json` | defaults of a new cycle |
| `view.py` | read-only snapshot (`snapshot`) and status-line row (`line`) |
| `claude_mod/` | the `rdloop-view` mod: `/rdloop` pane, toasts, notifications (marketplace `rdloop`) |
| `tests/` | unit (`test_rdlib`, `test_guard_hook`, `test_view`) and end-to-end (`test_integration`, fake claude / trainer / eval on a scratch repo) |

## Phases

`WAIT_WAVES -> AUDIT -> SCOUT -> PROPOSE -> LINT -> CRITIQUE -> (PROPOSE ...) -> GATE -> SCAFFOLD ->
IMPLEMENT (core, then one step per lever) -> STATIC -> REVIEW <-> FIX -> GPU (regression + smoke) ->
COMPAT -> SUMMARY -> AWAITING_USER`

| Phase | Who | Done when |
|---|---|---|
| WAIT_WAVES | supervisor | every wave in `wait_for_waves` is DONE; the base is resolved (`state.base`) |
| AUDIT | research-auditor | `01_audit.md` + `audit.json` (>= 2 gaps) |
| SCOUT | research-scout | `02_sota.md` + `scout.json` (>= 4 idea cards); sources marked verified from the transcript's WebFetch calls |
| PROPOSE | research-architect | `debate/proposal_vN.{md,json}` |
| LINT | supervisor | mechanical checks (lever count, new flag names, defaults, regexes, limits, smoke budget, responses to open objections); a failure returns to PROPOSE without using a round |
| CRITIQUE | research-critic | `debate/critique_vN.{md,json}`; ledger updated; agreed = no open blocker / major (blockers without evidence are downgraded); after `debate.max_rounds` open blockers drop their lever, open majors make it contested |
| GATE | owner (default) or auto | `plan_agreed.{md,json}`; `approval.json` with the plan's sha (owner) |
| SCAFFOLD | supervisor | module dir with `train.py` copied from the base module + version line |
| IMPLEMENT | research-engineer | `impl/implement_<part>.json`; manifest checked after every step |
| STATIC | supervisor | manifest, py_compile, `--help` lists the flags, `<module>.checks cpu`, `<base>.checks all`, exploop unit tests |
| REVIEW / FIX | research-critic / research-engineer | no open blocker / major finding; FIX -> STATIC -> REVIEW (of the new diff) |
| GPU | supervisor | regression R1 (old checkpoint re-scored = stored scores), R2 (base GPU checks), R3 (module GPU checks), R4 (levers-off micro-run = base micro-run, both without the pid-seeded RTC / MUSAN / level augmenter so they are deterministic); smoke S0, S_<lever>, O_<lever>, E (eval + merge path); <= 30 min of smoke |
| COMPAT | supervisor | `handoff/wave_seed.json`; every lever passes exploop's one-factor validator; `supervisor.py init` + `plan` on `handoff/dryrun_wave/`; smoke runs follow the run contract |
| SUMMARY | research-auditor | `FINAL_SUMMARY.md` + `summary.json`; ledger / literature / `docs/research/<C>/` updated |

Statuses: INITIALISED, RUNNING, WAITING, PAUSED, AWAITING_APPROVAL, NEEDS_USER, AWAITING_USER,
BLOCKED, STOPPED, PARKED, HANDED_OFF. The supervisor exits on AWAITING_USER / NEEDS_USER / BLOCKED /
PARKED; it keeps running (cheap ticks) in AWAITING_APPROVAL so an approval continues at once.

## Safety model

- **Isolation manifest** (`isolation/manifest.json`, built at `init`): sha256 of every tracked and
  untracked file outside the cycle's scope, and size + mtime of every file under `SLS_setup/exp/`
  (new files in old runs are fine; changed / deleted ones are not; runs an exploop wave is writing are
  skipped). Checked after each engineer step, at STATIC and at SUMMARY; a violation blocks the cycle
  (nothing is reverted automatically; untracked originals are in `isolation/frozen_untracked.tar.gz`).
- **Guard hook + deny rules** (per step, `steps/<step>.settings.json`): seatbelts. The engineer writes
  only inside the module dir, `scripts/<prefix>*`, `rdloop/<C>/impl|scratch/`, its `$TMPDIR`; no
  training / eval / launch, no GPU (CUDA hidden), no installs, no network, read-only git.
- **Steps run with** `--strict-mcp-config` (no MCP tools), a USD cap, `nice 10` for the engineer, the
  rtc-sdd python first on PATH, `HF_HUB_OFFLINE=1`.
- **GPU**: the session starts only when the GPU is idle (memory < 1 GB, no training / eval process,
  debounce) and no exploop wave is in DESIGN / RENDER / SMOKE / TRAIN / EVAL; exploop's own gate then
  sees rdloop's jobs as busy. Module writes wait while an exploop Claude step runs.
- **Every pass is tied to the module's code sha**: any edit re-runs the later phases;
  `rdctl.sh verify <C>` proves the module is the code that passed.

## Commands

```
scripts/rdloop/rdctl.sh new <C> --brief "..." [--auto-gate] [--from-wave W] [--levers-max N] [--focus ..] [--exclude ..] [--budget USD]
scripts/rdloop/rdctl.sh init|plan|start|status|tail <C>
scripts/rdloop/rdctl.sh note <C> "text"
scripts/rdloop/rdctl.sh approve <C> [--levers a,b]
scripts/rdloop/rdctl.sh gate <C> owner|auto
scripts/rdloop/rdctl.sh reopen <C> --at propose|implement ["note"]
scripts/rdloop/rdctl.sh pause|unpause|stop|park|verify <C>
scripts/rdloop/rdctl.sh resume <C> [--reset-attempts] [--accept-manifest]
scripts/rdloop/rdctl.sh handed-off <C> <wave>
python3 scripts/rdloop/view.py snapshot|line [--cycle C]
```

## Tests

```
/usr/bin/python3 -m unittest scripts/rdloop/tests/test_rdlib.py scripts/rdloop/tests/test_view.py      # py3.8
RDLOOP_TEST_HOOK_PY=/usr/bin/python3 python -m unittest scripts/rdloop/tests/test_guard_hook.py
/root/.conda/envs/rtc-sdd/bin/python -m unittest scripts/rdloop/tests/test_integration.py             # ~7 min, CPU
claude plugin test scripts/rdloop/claude_mod
```
`RDLOOP_TEST_DIR=<dir>` keeps the integration scratch repos for inspection.
