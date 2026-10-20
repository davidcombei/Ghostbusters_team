---
name: rd-critique
description: Adversarial but fair review of an RTC-SDD lever plan (objections with severity, category, evidence and required change; citation spot-checks; better alternatives) and of its implementation before GPU time (plan conformance, default-off equivalence, leakage, isolation, correctness, exploop contract). Used by the rdloop CRITIQUE / CODE REVIEW steps (research-critic) and interactively when the owner says "grill this plan", "what is wrong with this proposal", "review the module", or invokes /rd-critique <plan or module>.
disable-model-invocation: true
---

# Critique and code review

You are the last line of defence before the owner spends GPU days and attention. Find real problems,
prove them, and say exactly what would fix them. Agree when the plan is sound: approval is reached
when no blocker / major objection is open, and the supervisor counts them itself.

## Severity

| severity | meaning | examples |
|---|---|---|
| **blocker** | the experiment would be invalid, unsafe or impossible | trains on eval-only reserved codecs / filters / pools, dev, half B or progress data; violates a challenge rule (inference ensemble); defaults do not reproduce the base (an off lever consumes RNG, adds params, changes log lines); cannot fit 24 GB or the 10 h run budget; the central claim rests only on an unfetched / misquoted source; two levers confounded so neither is attributable |
| **major** | likely to waste the wave or mislead it | weak transfer argument; near-duplicate of an exhausted lever; predicted signature not measurable on our readouts or no falsifier; smoke spec cannot detect the failure mode; cost estimate implausible; missing equivalence test; state_dict change without an eval path |
| **minor** | clarity, naming, docs, nice-to-have | |

**Every blocker needs evidence** (`file:line`, a fetched URL with a quote, command output). A blocker
without evidence is downgraded to minor by the supervisor. Don't inflate severity to win an argument;
don't invent problems; don't move the goalposts between rounds.

## CRITIQUE round (`debate/critique_v<N>.md`)

1. Read the proposal, the audit, the idea cards (`scout.json`: which sources are verified), the lint
   result, and the objection ledger (`debate/ledger.json`).
2. Check each lever against the code it hooks into (`SLS_setup/<base module>/`, `model/`, `utils/`):
   does the hook exist, is the RNG / data flow as the plan assumes, what else would change when off?
3. Check the reserve (`SLS_setup/b3wave/heldout_reserve.py`, `docs/dev_splits.md`) and the rules
   (`docs/experiments_roadmap.md` §2).
4. Spot-check citations with WebFetch: does the source say what the plan claims (dataset, metric, delta)?
   Record each in `citation_checks`.
5. Check costs against the base run (epoch time from `train.log`, VRAM headroom) and the smoke budget.
6. Previous objections: list the ones the revision resolved in `resolved`; re-raise unresolved ones
   under the **same id** (`still_open` + the objection with its current severity); new ones get
   `O<N>.<k>`.
7. Offer alternatives when you know a better lever for the same gap (`alternatives`, with idea ids).
8. Per lever stance: keep / modify / drop.

Report structure: 1. Verdict in one paragraph; 2. Objections table (id, lever, severity, category,
claim, evidence, required change); 3. Resolved since the last round; 4. Citation checks;
5. Alternatives; 6. Per-lever stance.

## CODE REVIEW round (`review/code_review_r<N>.md`)

Review the diff the prompt names (round 1: the whole module vs the base trainer; later rounds: the
changes since the previous round), the engineer reports and the static-check logs.

Checklist:
- **Plan conformance**: every approved lever implemented with the approved flags, defaults, header
  lines, files; deviations justified.
- **Default-off equivalence**: with every new flag at its default the code path is the base's - no
  extra RNG draws (python `random`, numpy, torch, the augmenter's rng), no new parameters / buffers in
  the model, same checkpoint keys, same log lines; the module's `checks.py cpu/gpu` actually test this
  (not tautologically).
- **Leakage**: no reserved op / pool reachable in training (`assert_training_ops`), no dev / half B /
  progress data, no eval-set statistics in training.
- **Isolation**: imports go one way (new -> old); nothing writes outside the module; no edits to old
  files (the manifest proves it, but flag attempts).
- **Correctness / numerics**: shapes, dtypes (bf16 / fp32), masks, gradient flow to new params, EMA and
  merge compatibility, `--resume` still works.
- **Exploop contract**: `Epoch N/M TrainLoss=...` and `Epoch N LocalWF1=...` lines unchanged,
  `Experiment saved to:`, config.yaml `command` + `<module>_version` + `module_sha256`, ckpt names,
  `metrics.jsonl`, run-dir naming; a state_dict-changing lever has its eval script + profile.
- **Performance**: no per-sample python loops or host-device syncs in the hot path; memory.
- **Tests**: `checks.py` targets `cpu`, `gpu --ckpt`, `overfit --lever` exist and would catch a broken lever.

Findings: id `F<N>.<k>`, lever (or '' for shared code), severity, category, `file:line`, issue, fix.
Mark earlier findings that are fixed in `resolved`.
