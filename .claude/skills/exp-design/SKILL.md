---
name: exp-design
description: Choose the next iteration of a training-exploration wave - exactly one trainer-flag factor changed against the current best, picked from the previous run's evidence, with hypothesis, predicted signature and falsifier. Used by the exploop DESIGN step (exp-designer) and interactively when the user asks "what should the next run be", "design the next iteration", "propose the next flags", or invokes /exp-design <run or wave>.
---

# Designing the next exploration iteration

One iteration changes **one factor** of the current best's trainer command line. The result must
be attributable to that factor. Inside a wave there are no code changes, only CLI flags.

## Inputs

- **Wave spec** `exploop/<wave>/wave.json`:
  - `search_space`: `allowed`, `coupled` groups (each counts as one factor), `forbidden`, `fixed`
    flags, `max_epochs`, `seed`;
  - `lever_queue` (candidates from the approved plan);
  - `lever_notes` (known pitfalls; obey them);
  - `rules`.
- **Plan snapshot** `exploop/<wave>/plan_snapshot.md`.
- **User notes** `exploop/<wave>/user_notes.md`. They are the highest priority, and the newest note
  wins.
- **Current best:** its argv (the base), run dir and readouts.
- **The latest report** in `docs/experiments-exploration/...`, especially §5 (what went wrong) and §7
  (proposals), plus its `analyze.result.json` (proposals already re-ranked: configuration first).
- **The wave history:** each iteration's change, verdict, Δ and decision. Don't repeat a lever that
  was already tested on the same base unless the dose differs and the evidence asks for it.
- **The trainer's flags:** `grep -n add_argument <trainer source>`, plus their help text and the
  header line each prints at startup.

## Procedure

1. **Find the largest remaining error** in the latest evidence: report §5,
   `largest_remaining_error`, the per-set recall_bona / EER and the slices.
   - Recurring B2/B3 errors: noisy bona recall far below spoof recall; a saturated tail of noisy
     bona at p_spoof ≥ 0.99; English bona behind Chinese bona; the g726/gsm codecs; SNR 0; noise_only
     and sil_only bona read as spoof; dev loss rising from epoch 1.
2. **Map it to a lever** from `lever_queue` or the analyst's proposals.
   - If the lever target was already hit and did not move, prefer a different mechanism over a
     higher dose.
   - If the last iteration was a clear positive, a dose step of the same lever (one flag) is
     legitimate.
3. **Check the constraints:** one factor, all flags in `allowed`, nothing `forbidden`, `fixed` flags
   untouched, `num_epochs ≤ max_epochs`. Then check `lever_notes`, for example:
   - `--ce_weights` order is [spoof, bonafide]: more bona weight means a larger second value;
   - consistency needs `--batch_size 8`;
   - the sampler needs re-derived CE weights.
4. **Compute derived values with python** and paste the command output into `derivation`, for
   example the CE weights for `--sampler balanced` from `train_label.txt`. Never estimate them in
   your head.
5. **Write the falsifiable claims:**
   - `hypothesis`: the mechanism;
   - `predicted_signature`: which metrics or sets move, in which direction, roughly by how much;
   - `falsifier`: what result would refute it, using the wave's ±band rule.
6. **Write `expect`:** regexes of the stdout header lines that prove the flag took effect. Grep the
   trainer source for the exact print, e.g. `^SpecAugment: on`,
   `^Consistency: js lambda_js=1.0`, `^SNR: 0..20 dB`, `^Codec profiles: random`,
   `^Sampler: balanced`. The supervisor stops the run in its first ~10 minutes if they are missing.
7. **Name it:**
   - `track` = `<prefix><letter>_` plus the best's descriptive tokens plus a short lever token plus
     `_s<seed>`, e.g. `B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_codecrand_s1234`;
   - `lever_slug` = the short token (letters, digits, dots) used in registry ids;
   - `change_vs_ref` = one line such as `+ --aug_codec_profiles random`.
8. **`kind`:** `config`. Use `seed_repeat`, where only `--seed` changes, only when no configuration
   lever is left that the evidence supports; say why in `rationale`.

## Anti-patterns (each happened before)

- **Ranking a seed repeat first** while configuration levers remain. The user's brief is to explore
  configurations.
- **Two factors at once.** For example, a new LR together with a new schedule, unless they form
  one coupled group in wave.json.
- **Label smoothing combined with asymmetric `--ce_weights`.** This shifted B1's boundary (oracle
  threshold ≈0.69) and made the dev loss uninformative.
- **A long cosine schedule that never anneals** before early stopping. Keep `--earlystop_epoch 99`
  and the short schedule.
- **Getting the `--ce_weights` direction wrong.** It is [spoof, bona], so 0.19 0.81 means *less*
  bona weight.
- **Typing numbers from memory, or using progress-split scores.**

## Output

- **In the exploop step:** the structured JSON (schema `scripts/exploop/schemas/design.json`) and
  nothing else.
- **Interactively:**
  - a short table: lever, flags, the target error with its evidence and file, the predicted
    signature, the falsifier;
  - two ranked alternatives, one line each;
  - the launcher change. Inside a wave it goes through `loopctl.sh note`; outside one it is a
    `scripts/run_<X>.sh` modelled on `scripts/run_B3a.sh`, which the user launches.
