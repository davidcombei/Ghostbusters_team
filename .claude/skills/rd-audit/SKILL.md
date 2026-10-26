---
name: rd-audit
description: Evidence-only gap analysis of the current best RTC-SDD run before a research cycle (what is the base, everything tried so far with outcomes, where the remaining errors are, the gaps worth new levers, the constraints), and the owner-facing final summary at the end of an rdloop cycle. Used by the rdloop AUDIT / SUMMARY steps (research-auditor) and interactively when the owner says "audit the current best", "what gaps are left", "what have we tried", or invokes /rd-audit [<run>].
disable-model-invocation: true
---

# Gap audit and final summary

**The rule:** every number is copied from a file you read in this session, and the text names the
file. Hypotheses are labelled as such. Never read progress-split scores. Tables over prose.

## Where the evidence is

| What | Where |
|---|---|
| Current best + its argv | `exploop/<latest wave>/loop_state.json` `best`; `docs/experiment_log.csv` (`half_b_wf1`, `ci_low/high`, `decision`, `notes`) |
| Readouts on half B | `<run>/local_eval/{merge_top5,epoch_<N>}/metrics.json` (`half_B.wf1.*`, `half_B.sets.<set>.*`, `breakdowns`), `probe_report.json`, `compare_ci*.json` |
| Per epoch | `<run>/metrics.jsonl`, `<run>/logs/train.log` |
| What was tried | wave summaries + per-run reports in `docs/experiments-exploration/**` (§5 what went wrong, §6 verdict, §7 proposals), `exploop/*/iterNN/verdict.json`, `docs/research/lever_ledger.md` (earlier rdloop cycles) |
| Context | `docs/experiments_roadmap.md` (§0-2 conditions + rule-compliance matrix, §14 references), `docs/next_steps_plan.md`, `docs/dev_splits.md` (metric, halves, reserve), `docs/augmentations.md`, `SLS_setup/<module>/README.md` |
| Eval-only reserve | `SLS_setup/utils/heldout_reserve.py`, `SLS_setup/b3wave/heldout_reserve.py` (`RESERVED_*`) |

Use `python3 -I` for JSON / CSV parsing; per-slice tables from `breakdowns` and `scores/<set>.logits`
are fine (local sets).

## AUDIT: `rdloop/<C>/01_audit.md`

Title `# rdloop <C> - gap audit of <base label>`, a provenance blockquote (date, files read), then:

1. **Base**: label, run dir, trainer module, the levers it has on (flag diff vs the previous best),
   both readouts (WF1@0.5, WF1@oracle, F1 clean / noisy, EER noisy) with their files.
2. **What has been tried**: one table over the waves (and earlier cycles): lever, wave / run, Δ half_B
   (merge, final), CI, verdict, report path. Mark each direction **open / exhausted / untested**.
3. **Error anatomy**: where the remaining errors are - per set, per slice (platform / codec / SNR /
   language / src), probes (noise_only / sil_only bona, level flips), confident errors (p >= 0.99),
   the oracle gap (threshold vs ranking). Each row cites the file.
4. **Gaps** G1..Gn (>= 2, ranked): title, evidence rows, size (share of errors / WF1 it carries), causes
   (hypotheses), already tried, `addressable_by` (augmentation / loss / architecture / sampler /
   optimisation / data / eval). Prefer gaps whose evidence is strong and whose size is material.
5. **Constraints**: challenge rules (roadmap §2: e.g. one model at inference, weight averaging only),
   the eval-only reserve (never train on reserved codecs / filters / pools, dev or half B), compute
   (one 24 GB 4090, <= 10 h per 8-epoch run), deadline, data.
6. **Exhausted directions**: what not to re-propose without new evidence, with the report that closed it.

Return `audit.json` fields as the schema says (`best.readout_*` as strings copied from the file).

## SUMMARY: `rdloop/<C>/FINAL_SUMMARY.md`

The owner reads this to decide whether to start a wave. Facts come from `rdloop/<C>/tests/facts.json`
(computed by the supervisor) and the cycle files; copy, don't recompute.

Title `# rdloop <C> - final summary (<module>)`, provenance blockquote, then:

1. **Headline**: what was built, in two sentences; readiness (ready / partial / failed).
2. **Levers implemented**: table - lever, flags (default -> wave value), gap targeted, sources
   (verified), smoke result (time factor vs levers-off, peak VRAM, projected full-run hours), notes.
3. **Debate**: rounds, the objections that changed the plan and how they were resolved, contested points.
4. **Dropped ideas**: idea / lever, stage (debate / gate / implement / review / smoke), reason.
5. **Regression and isolation**: old-run scores reproduced (sets, max |Δlogit|), base checks, levers-off
   equivalence with the base trainer, manifest (files checked, violations = 0), exploop dry run.
6. **Recommended wave**: order of the levers (strongest evidence / biggest gap first), expected hours,
   the falsifier of each, anything the wave spec must fix (e.g. `--batch_size` coupling).
7. **Risks and questions for the owner**.
8. **Files**: module path, plan, audit, SOTA, debate, review, tests, wave seed, cost.
