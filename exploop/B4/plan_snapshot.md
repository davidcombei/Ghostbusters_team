> **Snapshot for the exploop supervisor**, written 2026-10-18 at the `/rd-cycle review B4` handoff (session a9b7f800).
> Agents read this copy, never `~/.claude/plans/`. The full plan is `rdloop/B4/plan_agreed.md` (sha `92c44cd85bba`).
> The final summary of the research cycle is `rdloop/B4/FINAL_SUMMARY.md`; copies of both are in `docs/research/B4/`.

# B4 wave: screen of the rdloop B4 augmentation levers (one flag off B2f each)

## Context

The owner's brief (`rdloop/B4/brief.md`): starting from B2f, find augmentation levers that close the gap between
clean and noisy F1. Locally this is half_B clean F1 on `dev_online_clean` vs noisy F1 on `sim_matched` /
`sim_heldout` / `echo`.

rdloop cycle B4 built `SLS_setup/b4wave/` (module sha `d163f934…`, `rdctl.sh verify B4` = VERIFIED). It adds three
default-off levers that put simulated platform DSP into training. With every lever off, `b4wave.train` is
b3wave/B2f (regression max |Δlogit| 0.0; the levers-off micro-run is identical to b3wave's).

**Reference: B2f** (`SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`).
Half-B WF1 91.98 / 92.14 (merge / final), F1 clean 97.60 / 97.69, noisy 89.57 / 89.76, heldout 87.92 / 88.27
(`rdloop/B4/FINAL_SUMMARY.md` §2).

Base command: B2f's argv (from its `config.yaml`) with the entry `-m b4wave.train`, `--batch_size 16`, 8 epochs.
The supervisor sets `--track` and `--keep_epochs`.

## Arms (`wave.json` `lever_queue` has the full targets, hypotheses, signatures and falsifiers)

| iter | arm | change vs B2f | targets | projected train h |
|---|---|---|---|---|
| a | `apm_ns` | `--aug_apm_p 0.3` | G1: suppressor artefacts on heldout bona (largest share of the WF1 gap) | 4.66 |
| b | `opus_plc` | `--aug_opus_plc_p 0.3` | G1 outcome: packet-loss concealment artefacts; mechanism capped at "directional" | 3.78 |
| c | `dtx_gate` | `--aug_dtx_p 0.3` | G3: pause-rich noisy bona false alarms. **Under-dosed as built**: a null reads "inconclusive (under-dosed)" and is never "refuted" | 3.84 |
| d | conditional | if `opus_plc` is positive on the WF1 CI rule: `--aug_opus_plc_p 0.3 --aug_opus_loss 0 0` (zero-loss control). Otherwise the analyst's best-supported B4 follow-up (B4 flags only) | attribution of an `opus_plc` gain (decoder path vs PLC) | ~3.8 |

Expected: about 14 h for a-c (12.28 h of training plus about 0.6 h of eval and analysis per arm), plus about
4.4 h for d.

## Rules

- **Screen design:** every arm is exactly its `set` on top of B2f's argv. `rules.keep_best` keeps B2f as
  the best for the whole wave, so each verdict and readout is against B2f. Promotion between arms would break the
  plan's placebo contrasts. The best run of the wave is chosen by the owner at wrapup from the verdicts.
- **Objective and verdicts:** as in B3. Merge readout `half_B.wf1.wf1`; ±0.5 band; promote = CI low > 0 and
  clean drop ≤ 0.5; positive = Δ > 0.5 with CI low > 0 and clean drop ≤ 0.5. Both readouts (merge top-5 and final
  epoch) are reported with a bootstrap CI.
- **Search space:** only the 7 B4 flags (3 coupled groups). B3 flags (consistency, sampler, SNR, specaug, LR) are
  out of scope, and `--batch_size` stays 16.
- **Smoke:** off. rdloop smoke-tested every lever on module sha `d163f934…`; any code change needs a new cycle or
  smoke run. The early check expects `^b3wave: ` and `^b4wave: ` and forbids `failed \(`.
- **Mechanism readout (analyst):** `scripts/b4_dsp_slices.py` on the merge and final readouts with `--ref` = B2f's
  same readout, `--placebo` = the other B4 arms that exist, `--lever <id>`. The wave SUMMARY re-runs it with
  `--summary` (final verdicts, placebo count printed; 0 placebo arms caps a verdict at "directional"). Owner rules
  O3.1-O3.3 apply: a run anomaly → "confounded"; escape needs heldout F1 Δ ≥ +0.5 on both readouts; "positive
  outcome" = the WF1 CI rule.
- **G6 labels:** after B4, heldout `ns:*`, `vad:*` and `plc` are "family seen, implementation unseen".
- **Combined "platform chain" run:** only after the wave, as a recommendation, if verdicts and the WF1 CI rule
  allow it.

## Per-iteration loop (the supervisor, no LLM except DESIGN / ANALYZE / SUMMARY)

DESIGN (exp-designer: the next queued arm) → RENDER (`scripts/run_B4<letter>.sh`) → WAIT_GPU → TRAIN (early check)
→ EVAL (`sls_local17`: the 17 report sets, top-5 merge + final epoch, CI) → VERDICT vs B2f → LOG (two rows in
`docs/experiment_log.csv`) → ANALYZE (exp-analyst report in `docs/experiments-exploration/b4-experiments/`) →
DECIDE (the best stays B2f). At the end: SUMMARY, then `/exp-loop wrapup B4`.

## Owner decisions at the handoff (2026-10-18)

- Start the wave as built. The `dtx_gate` under-dose is accepted; it runs last as a placebo arm.
- No B2f replicate arm.
- conda-forge `libopus 1.3.1` installed into the rtc-sdd env, the same version as the system library, as the
  fallback after a container restart.
- 2026-10-18 21:50: screen switched from `new_best_delta` 100 to `rules.keep_best` (exploop_lib.decide). The
  first setting made every decision `drop`; B4a's labels were recomputed to revisit (merge) / promote (final).
- No Claude budget cap for the wave (`claude.max_total_usd` 1000000; the per-step caps stay).
- Defaults kept: the F2.1 `dtx_gate` escape metric is WF1 (plan and code); the per-profile (pooled) `opus_plc`
  zero-loss attribution check is accepted, with the iteration-d control covering it.
