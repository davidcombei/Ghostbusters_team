# ties_merging — TIES-Merging for ModelSLS checkpoints

TIES-Merging ([Yadav et al. 2023, arXiv 2306.01708](https://arxiv.org/abs/2306.01708)) as a second
checkpoint-merging strategy next to the plain mean of `main_eval.merge_checkpoints`, with an
evaluation sweep and the analysis figures of the paper.

Nothing in `main_*.py`, `model/`, `utils/`, `b3wave/`, `b4wave/` or the existing `scripts/*` is
changed. The merged checkpoint is a plain `ModelSLS` state dict, so `scripts/local_eval.py
--model_path`, `scripts/make_submission.py` and `/post-train-eval` work on it unchanged.

## Algorithm

With $T$ checkpoints $\theta_t$ and the base $\theta_{init}$:

| step | what | official code (`prateeky2806/ties-merging`) |
|---|---|---|
| task vectors | $\tau_t = \theta_t - \theta_{init}$ | |
| **Trim** | keep the top-$k$% of $\|\tau_t\|$ (global over the flattened vector), zero the rest | threshold = $(n - k_{keep} + 1)$-th smallest, entries $\ge$ threshold kept; exact zeros never count as kept |
| **Elect** | $\gamma = \mathrm{sign}(\sum_t \hat\tau_t)$ | sum, not majority; an exact cancellation gives 0 (counted, `n_gamma_zero_cancel`) |
| **Disjoint merge** | $\tau_m^p = \frac{1}{|A^p|}\sum_{t \in A^p} \hat\tau_t^p$, $A^p = \{t: \mathrm{sign}(\hat\tau_t^p) = \gamma^p\}$ | `dis-mean`: divide by the number of selected non-zero entries, clamped to $\ge 1$ |
| result | $\theta_m = \theta_{init} + \lambda\,\tau_m$ | defaults $k = 20$, $\lambda = 1$ (paper's no-validation recipe; it searched $\lambda \in [0.8, 1.8]$) |

**Which tensors get TIES** is decided by the base (`base.py`):

| `--base` | $\theta_{init}$ | TIES keys | plain mean (exactly `merge_checkpoints`) |
|---|---|---|---|
| `pretrained` (default) | hub Qwen3-ASR audio tower, read from the cached safetensors (`model.audio_tower.*` → `ssl_model.model.*`, bf16 → f32, layers $\ge$ `n_layers` dropped; bit-identical to a fresh `ModelSLS`) | the 393 `ssl_model.*` tensors | head (`sls.fc`, `fc1`, `fc2`, `first_bn.weight/bias`) and the BN running stats; `num_batches_tracked` mean-then-cast |
| `<ckpt.pth>` | that checkpoint | every float parameter (head included) | BN running stats, `num_batches_tracked` |

The head has no pretrained init (random, never saved), so in the default mode it is averaged the way
the production top-K merge averages it today. BatchNorm statistics are never trimmed or sign-elected.

**Arithmetic.** Thresholds, masks and signs are computed on float32 task vectors; the merged task
vector is accumulated in float64 from the memory-mapped checkpoints and added to the float64 base,
then cast. Hence $T = 1$, $k = 100$, $\lambda = 1$ reproduces the checkpoint bit for bit, and with
$k = 100$ every position kept by all models with one sign equals the plain mean (checked on real
checkpoints: max |diff| = 0). Memory: per ingredient one int8 signed-mask vector (0.3 GiB for the
314M backbone floats) plus a few float32 / float64 accumulators, ~12 GiB for $T = 7$; the process
cgroup on the dev box is 64 GiB. The `--stats` pass needs ~20 GiB and a few minutes (one in-place
sort of $|\tau|$ per ingredient).

## Files

| file | role |
|---|---|
| `merge.py` | core: `Ingredients` (mmap'd checkpoints + base + key partition + flat layout), `trim_mask`, `elect_signs`, `disjoint_merge`, `assemble`, `Merger` (caches $\tau_m$ per $k$ so a $\lambda$ sweep only re-assembles; `plain_mean()` = `merge_checkpoints`), `ties_merge(paths, base, k, lam, ...)`, `cell_tag` |
| `base.py` | `load_base("pretrained" \| ckpt)`, `pretrained_backbone` (safetensors, or `--base_via_model` through `ModelSLS`), `runs_of` / `model_args_from_runs` (model args from the runs' `config.yaml`, warnings when runs disagree, e.g. on the seed) |
| `stats.py` | `analyze(merger, k, k_grid, out_dir, lam)` → `ties_stats.json` + `ties_stats.npz` (aggregates only; see below) |
| `plots.py` | the figures, from the aggregates and a sweep's `summary.csv`; `python -m ties_merging.plots --stats <dir> [--sweep <dir>] [--out <dir>]` |
| `checks.py` | `python -m ties_merging.checks cpu \| real --merge_list <txt> [--with_model]` |
| `../../scripts/ties_merge.py` | merge list → `merged.pth` + `merged.sources.txt` + `ties_config.json` [+ stats] |
| `../../scripts/ties_eval.py` | $(k, \lambda)$ sweep scored on the local sets, `eval_checkpoints.py`-style outputs |
| `../../notebooks/ties_merging_analysis.ipynb` | the paper-style analysis, thin wrapper over `plots.py` |

## Flags

`scripts/ties_merge.py` (run from the repo root or `SLS_setup/`; merge-list paths resolve as in `local_eval.py`):

| flag | default | meaning |
|---|---|---|
| `--merge_list` | required | one checkpoint path per line (`select_topk.py` output, or any list; cross-run lists allowed) |
| `--out` | `<merge_list dir>/ties_<tag>` | output dir; `--force` overwrites `merged.pth` |
| `--base` | `pretrained` | `pretrained` or a checkpoint path (full-model TIES) |
| `--ssl_name`, `--n_layers` | run `config.yaml` | model args for the pretrained base |
| `--base_via_model` | off | build `ModelSLS` for the base instead of reading the safetensors (any SSL front-end) |
| `--k` | 20 | % of entries kept per task vector; 100 = no trim |
| `--lam` | 1.0 | $\lambda$ |
| `--trim_scope` | `global` | `global` (paper) or `tensor` (threshold per tensor) |
| `--merge` | `dis-mean` | `dis-mean` (paper, / $|A^p|$) or `mean_over_T` (/ $T$: sparsify without renormalising) |
| `--no_trim`, `--no_elect` | off | ablations of the Trim / Elect step (paper Fig. 12) |
| `--stats`, `--k_grid` | off, `5 10 20 30 50 70 100` | write the analysis aggregates; $k$ grid of the conflict-vs-$k$ statistic |

`scripts/ties_eval.py` (run from `SLS_setup/`): the same merge flags with **lists** (`--k 10 20 30`,
`--lam 0.8 1.0 1.2`, `--merge dis-mean mean_over_T`), plus `--ablation` (no-trim / no-elect /
mean-over-T cells at the first $k$, $\lambda$), `--single <i> ...` (the $T = 1$ trim curve of
ingredient $i$), `--no_mean` (skip the plain-mean cell), `--save none|best|all` (write `merged.pth`
of the best cell by `--select_on` / of every cell), and the `eval_checkpoints.py` flags (`--names`,
`--sets`, `--select_on half_A.wf1`, `--top_k`, `--device/--devices/--layer_split`, `--batch_size`,
`--num_workers`, `--seed`, `--force`, `--loudness_norm/--no_loudness_norm/--loudness_norm_lufs`;
model args default to the run's `config.yaml`).

## Outputs

`ties_merge.py --out <dir>`: `merged.pth`, `merged.sources.txt` (absolute source paths, itself a
merge list), `ties_config.json` (every setting, base kind + hub snapshot or checkpoint, sources,
package version + sha256, merge summary: kept fraction per model, `shared_frac` = positions kept
by $\ge 2$, `conflict_frac` = sign conflicts among those, `merged_frac`, `n_gamma_zero_cancel`,
$\|\tau_m\|$), and with `--stats`:

| `ties_stats.json` | `ties_stats.npz` | paper figure |
|---|---|---|
| per model: $\|\tau_t\|$, per-layer-group norm and relative norm, $|\tau|$ quantiles, thresholds and kept fraction per $k$, kept fraction per group, exact-zero count | `mag_hist` (log10 $|\tau|$, 90 bins), `concentration` (share of $\sum\tau^2$ in the top-$q$%, $q = 1..100$) | Fig. 3 / 10 |
| `conflict_vs_k`, `conflict_vs_T` | `conflict_vs_k`, `shared_vs_k`, `conflict_vs_T` | Fig. 4 / 14 / 15 |
| `kept_count_groups`, `agreement_bins`: mean $|\theta_m - \theta_{init}|$ for TIES vs plain mean | `kept_count_hist/_mean_abs`, `agreement_hist/_mean_abs` | Fig. 9a / 9b |
| per model `elect_agree_frac`, `contribution_frac`; `cosine`; per group `shared/conflict/merged_frac`, TIES / mean / init norms | `cosine` | — |

`ties_eval.py --out <dir>`: `<tag>/scores/<set>.txt|.logits` and `<tag>/metrics.json` per cell
(`local_eval.py` format + `half_A` + a `ties` block, so `scripts/compare_evals.py`,
`scripts/b3_compare_evals.py --ci 2000` and `scripts/probe_report.py` work on any cell),
`summary.csv` (one row per cell with `kind, k, lam, merge, trim, elect, T` and the
`{full,half_A,half_B}.{wf1,f1_clean,f1_noisy,...}` columns in percent), `best.json`. Tags:
`mean`, `k20_lam1`, `k20_lam1_meanT`, `k20_lam1_notrim`, `k20_lam1_noelect`, `single0_k20_lam1`.

## Workflow

```bash
cd SLS_setup
P=/root/.conda/envs/rtc-sdd/bin/python
RUN=exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623

# 1. one merge + the analysis aggregates (CPU, ~3 min merge + ~5 min stats)
$P -u ../scripts/ties_merge.py --merge_list $RUN/top5_by_wf1/merge_list.txt --stats --out $RUN/ties_k20_lam1
$P -m ties_merging.plots --stats $RUN/ties_k20_lam1          # -> $RUN/ties_k20_lam1/figures/*.png
# or open notebooks/ties_merging_analysis.ipynb (STATS_DIR / SWEEP_DIR at the top, env TIES_STATS_DIR / TIES_SWEEP_DIR);
# headless: (cd .. && jupyter nbconvert --to notebook --execute notebooks/ties_merging_analysis.ipynb --output /tmp/x.ipynb)

# 2. stage 1 (GPU): select on the half-A selection sets (~5 min per cell): k grid, lambda grid, ablation, T = 1 curve
$P -u ../scripts/ties_eval.py --merge_list $RUN/top5_by_wf1/merge_list.txt \
    --k 10 20 30 50 70 100 --lam 1.0 --ablation --single 0 \
    --names dev_online_clean sim_matched_mini sim_heldout_mini --out $RUN/local_eval/ties_stage1
$P -u ../scripts/ties_eval.py --merge_list $RUN/top5_by_wf1/merge_list.txt \
    --k 20 --lam 0.6 0.8 1.0 1.2 1.4 --no_mean \
    --names dev_online_clean sim_matched_mini sim_heldout_mini --out $RUN/local_eval/ties_stage1

# 3. stage 2 (GPU, ~0.5 GPU-h per cell): the best stage-1 cells + mean on the 17 report sets, read on half B
NAMES="dev_online_clean sim_matched_v1 sim_heldout_v1 sim_echo_v1 ..."     # the 17 sets of /post-train-eval
$P -u ../scripts/ties_eval.py --merge_list $RUN/top5_by_wf1/merge_list.txt --k 20 --lam 1.0 \
    --names $NAMES --out $RUN/local_eval/ties_sweep --save best
$P ../scripts/b3_compare_evals.py $RUN/local_eval/ties_sweep/mean $RUN/local_eval/ties_sweep/k20_lam1 --ci 2000

# 4. the best merged.pth goes anywhere a checkpoint goes
$P ../scripts/local_eval.py --model_path $RUN/local_eval/ties_sweep/k20_lam1/merged.pth ...
```

Rules: a merge of checkpoints of the same architecture is one model (weight averaging is allowed
by the rules matrix); the pretrained base has no spoof training. Select on half A, report on
half B, and confirm any gain with the paired-bootstrap CI before adopting it.

## What to expect on same-run checkpoints (B2f top-5, measured)

The paper's analysis is about *different tasks*. Our default list is five EMA epochs of one run:

- The task vectors are nearly parallel (pairwise cosine 0.97–1.0), so after trimming at $k = 20$
  the positions kept by $\ge 2$ models are 23.9 % and **0.000 % of them have a sign conflict**
  (0.25 % at $k = 50$, 8 % at $k = 100$). **Elect is a no-op at the default $k$**; what TIES
  changes is (a) trimming, which resets 70 % of the backbone to the pretrained value, and (b) the
  $1/|A^p|$ renormalisation, which gives a position kept by one model its full value instead of
  $1/5$ of it (`merged_frac` 29 %, $\|\tau_m\| = 4.49$ vs $\|\bar\tau\| \approx 4.48$). The
  `--merge mean_over_T` and `--no_trim` cells separate the two.
- The updates are not heavy-tailed: the top 20 % of entries hold about two thirds of
  $\sum\tau^2$, not "almost all" as for the paper's PEFT / ViT task vectors (AdamW moves every
  weight by about the learning rate). Trimming at 20 % therefore removes real signal; sweep $k$
  upward and read the $T = 1$ trim curve before trusting the default.
- The B5 research plan (`docs/experiments-exploration/b5_wave/..._B5_wave_research_directions_and_implementation_plan.md`,
  deprioritised table) cites SuperMerge: TIES/DARE sparsification was worse out of distribution
  than plain averaging. The sweep always reports the plain mean next to every TIES cell so this is
  settled with our own numbers.

## Checks

| target | what | time |
|---|---|---|
| `cpu` | synthetic state dicts: `module`, `trim` (hand-built masks, $\ge$ convention, ties at the threshold, per-tensor scope), `elect` (incl. exact cancellation), `disjoint` (hand values for dis-mean / mean-over-T / no-elect), `identity` ($T = 1$, $k = 100$ bit-exact; checkpoint base, $\tau = 0$), `mean_equivalence` (no conflicts, $k = 100$ = `merge_checkpoints`; `plain_mean()` bit-exact), `kept_fraction` (tolerance 1e-5), `lambda_cache`, `dtypes_keys` (keys, order, dtypes, int64 buffer, contiguity), `stats_totals` | seconds |
| `real --merge_list <txt>` | first two checkpoints of a real list + the pretrained base: base covers all 393 backbone tensors; duplicate list at $k = 100$ bit-exact to the checkpoint and to `merge_checkpoints`; two epochs at $k = 100$: head bit-exact vs `merge_checkpoints`, backbone max |diff| = 0 where both keep one sign; default $k = 20$ kept fraction and finiteness; checkpoint base identities | ~5 min CPU |
| `real ... --with_model` | also: safetensors base == `ModelSLS(...).state_dict()` bit-exact | + model build |

Last run (2026-10-19, B2f top-5 list): `cpu` 10 passed; `real --with_model` 6 passed.
