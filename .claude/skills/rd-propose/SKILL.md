---
name: rd-propose
description: Turn a gap audit and SOTA idea cards into a design plan of 2-4 small, flag-gated, default-off levers for the RTC-SDD trainer (flags, default equivalence, implementation hooks, hypothesis, predicted signature, falsifier, cost, smoke spec, wave lever), and revise it answering a critic's objections. Used by the rdloop PROPOSE / REVISE steps (research-architect) and interactively when the owner says "design levers for <gap>", "turn these ideas into a plan", or invokes /rd-propose <cycle or topic>.
disable-model-invocation: true
---

# Designing a lever plan

A lever is **one component change behind new flags that default to the base behaviour**. The cycle
builds 2-4 of them in one new module; an exploop wave then screens each one as a single factor
against the current best. Small, systematic, attributable: if a lever's effect could not be told apart
from another lever's, merge them or drop one.

## Choosing levers

1. Start from the audit's top gaps (priority 1-2) and the idea cards with **verified** sources.
2. Per lever, one component: an augmentation stage, a loss term, a head / pooling block, an optimiser
   rule, a sampler. Not "augmentation + loss" in one lever.
3. Prefer levers that (a) target the largest gap with the strongest evidence, (b) are cheap to test,
   (c) are independent of each other. Diversity across gaps beats three variants of one idea.
4. Never: training on eval-only reserved ops / pools or on dev / half B / progress data; inference
   ensembles; re-opening an exhausted direction without new evidence; a lever whose only support is an
   unverified source; anything needing packages / weights / data the env lacks without saying so in
   `dependencies` (the owner installs, the lever waits).
5. Respect the cycle limits (`cycle.json` `limits`): extra VRAM, epoch-time factor (current runs are
   already ~8 h), LOC per lever; at most one lever that changes the checkpoint keys (`state_dict_change`).

## Lever spec (the JSON schema fields; every field is required)

- **flags**: `--<area>_<name>` (lower snake case, new: not in the base trainer), type, default, choices,
  help. Multi-flag levers list a `wave_lever.coupled` group.
- **default_equivalence**: why the defaults reproduce the base bit for bit - same RNG draws (an off
  augmentation must not consume random numbers), same forward, same checkpoint keys, same log lines.
- **mechanism, gap_ids, idea_ids, sources** (URLs from 02_sota.md).
- **hypothesis / predicted_signature / falsifier**: measurable on our readouts - which sets / slices /
  probes move and in which direction, what must NOT move (e.g. clean F1), what result kills it.
- **expect_regex**: the header line(s) the trainer prints when the lever is on (e.g. `^AugPLC: on p=`).
- **state_dict_change / params_added / dependencies / files** (inside `SLS_setup/<module>/` or
  `scripts/<prefix>*`).
- **equivalence_test**: the CPU / GPU check that proves lever-off == base (RNG draws, logits, keys).
- **cost**: est_loc, extra_vram_gib, epoch_time_frac (honest; the smoke run measures it).
- **smoke**: minutes_est for a 1-epoch 1 % subset run (levers-off S0 takes ~3-4.5 min; add your
  overhead) and what it must show; the whole smoke session is capped at 30 min.
- **wave_lever**: `set` = the flag values the wave screens (store_true -> `[]`), `targets`.
- **risks**.

## Report `rdloop/<C>/debate/proposal_v<N>.md`

Title `# rdloop <C> - proposal v<N>`, then: 1. Summary table (lever, flags, gap, cost, smoke);
2. One section per lever (motivation with citations and audit evidence, mechanism, flags, default
equivalence, implementation sketch: files / classes / functions / where it hooks into the base code,
predicted signature, falsifier, cost, smoke, risks); 3. Shared changes; 4. Eval impact; 5. Responses to
the critic (v >= 2); 6. Changes since the previous version.

## Revising (v >= 2)

- Answer **every** open objection in `responses`: `accept` (and change the plan), `modify` (partial,
  say what), or `rebut` (with evidence: file:line, URL, measurement). Silence is not an answer.
- Answer every critic alternative in `alternatives_response` (adopt with a full spec, or decline + why).
- You may drop a lever (`dropped` with the reason) or swap one in from the idea cards (full spec).
- Owner notes in the prompt override the critic and your own preferences.
- A LINT failure (from the supervisor) is mechanical: fix exactly what it lists.
