# RTC-SDD B5 wave: research directions and implementation plan

> Status: plan only (nothing is implemented yet) · Created 2026-10-19 · Companion to `docs/experiments_roadmap.md` and `docs/next_steps_plan.md`
> Source: deep-research run `wf_441b5a05-8c0` (5 search angles, 22 primary sources, 25 claims put through 3-vote adversarial verification, 25 confirmed, 0 refuted).
> B4 is running on another machine. Before any B5 run starts, update **REF** (§1) and drop any B5 arm that duplicates a B4 arm.

---

## 0. TL;DR

- **The remaining error is bonafide false alarms under unfamiliar processing.** Noisy bonafide recall is about 77%; spoof recall is about 99%.
- **The 2026 literature agrees with our diagnosis.**
  - Detectors read AGC/NS energy flattening and diffuse codec noise as synthesis artifacts.
  - Score calibration cannot fix this, because it leaves EER unchanged. The bona/spoof margin has to come from training.
- **The B3 nulls fit the AI-generated-image-detection lesson.** Generalization fails from too little *coverage* of the processing space, not too little volume. Widening the ranges of existing stages (B3d codec profiles, B3e SNR) adds volume, not coverage.
- **Order of attack:**
  1. **Free, weight-space methods first:** cross-run greedy soup, then WiSE-FT-LP.
  2. **LP-FT warm start**, replacing the random head at a 10× LR.
  3. **New *types* of bonafide-first processing chains** (WebRTC APM / RNNoise / AMR-WB / Opus PLC), using implementations disjoint from the heldout reserve.
  4. **Hard-chain mining** on training bonafide clips.
  5. **Nes2Net-X head.**
  6. **DSU feature-style perturbation.**
- **Deprioritised, with evidence:** SAM, MoLEx, TIES/DARE merging, calibration-only fixes, audio-LLM QA framing.

---

## 1. Context

| Item | Value |
|---|---|
| **REF (provisional)** | **B3a-cjs03**: Qwen3-ASR-1.7B full FT, SLS head, RTC+MUSAN aug, loudness norm −23 LUFS, `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, bs 8 pairs, warmup-cosine 8 ep, LR 2.5e-6 / head ×10, EMA, top-5 merge. Launcher `scripts/run_B3a.sh`. |
| REF local scores (merge) | half_B WF1 **93.06**: clean 97.48, matched 92.28, heldout 90.02, echo 94.61; noisy bonafide recall 77.3% |
| Nulls / negatives | B3b balanced sampler (−0.83), B3c λ=1.0 (−0.77), B3d random Opus profiles (−2.20), B3e SNR 0–20 dB (−1.07); B2f loudness norm vs level aug (null) |
| Deadlines | final eval data 2026-11-09, submission 2026-11-16 (AoE): about 3 weeks of compute for B5 |
| Screen cost | ≈4–5 GPU-h per 8-epoch run (from `gpu_h` in `docs/experiment_log.csv`) |

**Updating REF after B4.** If B4 promotes a run, B5 arms change one factor against *that* run. A3 (soup) and A1 (WiSE-FT) should then also include the B4 checkpoints.

---

## 2. What the research says about our failure mode

| Finding | Source | Consequence for B5 |
|---|---|---|
| Under degradation, errors are mostly bonafide flagged as spoof. Detectors treat flattened energy (AGC/NS) and diffuse codec noise as synthesis artifacts. The most damaging adversarial chains all targeted bonafide audio, and the worst was **reverb + Opus**. | *What Do Deepfake Speech Detectors Actually Hear?* [2606.10912](https://arxiv.org/abs/2606.10912) | Put processing on **bonafide first**, so that processing traces stop predicting the label (D-B1). |
| An SSL-AASIST detector with near-zero in-domain EER rejected 78.7% of In-the-Wild bonafide at its in-domain threshold (HTER 39.5%). z-norm, temperature/shift calibration and embedding-mean alignment leave EER unchanged. | Threshold-transfer audit [2606.21584](https://arxiv.org/abs/2606.21584) | Calibration only moves the operating point. Our own oracle gap is already ≈0.9 F1. Do not spend a run on calibration. |
| Moving the threshold 0.5 → 0.30 gained 1.3 F1 cross-domain, but CORAL / target normalisation **hurt**. Platt scaling leaves EER/AUC unchanged. | [2609.09489](https://arxiv.org/abs/2609.09489), [2507.01750](https://arxiv.org/abs/2507.01750) | Be careful with feature-alignment losses (DANN/CORAL on the roadmap; C5 here). |
| NPR-style low-level artifact detectors drop to 54% balanced accuracy on unseen generators; a UnivFD CLIP linear probe reaches 79%. Training on content-matched reconstructions reaches 92–95% (B-Free, DINOv2-on-reconstructions). | PROBE [2605.24906](https://arxiv.org/abs/2605.24906), B-Free CVPR'25 [2412.17671](https://arxiv.org/abs/2412.17671) | Coverage and content matching matter more than volume. Our offline↔online twins are content-matched pairs: use them more (D-B1 pair-consistent chains). |
| Reconstruction-based fakes still leave frequency-domain mismatches that become shortcuts. Aligning real and fake in both pixel and frequency space gives large cross-generator gains. | DDA [2505.14359](https://arxiv.org/abs/2505.14359) | Apply band-limit / high-frequency changes **equally to both classes**. |
| CELP/ACELP codecs (EVS, Speex) and packet-loss concealment smooth away forgery traces; Opus is the most stable. Partial transfer to held-out codecs and loss rates. | [2508.01467](https://arxiv.org/abs/2508.01467) | Add AMR-WB / SILK / EVS and Opus PLC as new stage *types*, as one part of a compositional chain. |
| An AEC → NS → AGC → VAD front-end produces nonlinear time-frequency distortions; VAD shifts segments in time. | TFCL, ACM MM'26 [2607.17761](https://arxiv.org/abs/2607.17761) | The template for the `rtc_v2` chain. Keep the consistency loss robust to time shifts. |
| Open-source detectors lose about 48% AUC in the wild (Deepfake-Eval-2024). HuBERT is more stable under codec shift than wav2vec2 and WavLM. | [2503.02857](https://arxiv.org/abs/2503.02857) | Prioritise robustness over in-domain fitting. |

**Coverage gaps in the research run.** No verified claims came back on Binoculars/DetectGPT, SWAD/MIRO/Fishr, UnivFD for audio, CodecFake, or AT-ADD 2026 system papers. For those, use the roadmap; this document does not override it.

---

## 3. Protocol (inherited, unchanged)

Taken from `docs/next_steps_plan.md` §1 and the B3 wave.

- **One factor per arm against REF.** 8-epoch screen, seed 1234. For any arm that clears the band, add a second seed.
- **Primary readout:** half_B WF1@0.5 for the top-5 merge **and** for the final-epoch EMA checkpoint.
- **Comparison:** `scripts/b3_compare_evals.py <REF> <cand> --ci 2000`.
- **Adopt** only if the paired-bootstrap ΔWF1 CI excludes 0 **and** clean F1 drops by no more than 0.5. Results inside ±0.5 count as null.
- **Required signature for "real" improvement:**
  - Heldout and echo bonafide recall go **up** while noisy EER goes **down**.
  - A mirror-image move of bonafide recall against spoof recall with flat EER is a threshold shift, and is **not** a win.
  - Check the `noise_only` probe (bona-source → spoof rate) and `sil_only`.
- **`sim_unseen_v1`:** read once, at the end of the wave.
- **Logging:** after each run, `/post-train-eval <run> --baseline <REF eval>`, then `/log-experiment`. Reports go in `docs/experiments-exploration/b5_wave/`.

**Code convention.** Follow b3wave's rule: new behaviour is additive, every new flag defaults to REF behaviour, and checkpoints stay plain `ModelSLS` state_dicts so `select_topk.py`, `local_eval.py`, `main_eval.py`, `make_submission.py` and the skills keep working. Put new flags into `SLS_setup/b3wave/` (or a `b5wave/` package that subclasses it, if the diff is large), and add a smoke test to `b3wave/checks.py` / `scripts/b3_smoke.sh` for each one.

---

## 4. Directions

Each direction lists:
- **H**: hypothesis
- **Sig**: predicted diagnostic signature
- **Evidence**
- **Implementation**
- **Rules**
- **Cost**
- **Kill**: kill criterion

### Tier A: cheap, strongest evidence (do first)

#### A3 · Cross-run greedy model soup (run #0)

- **H:** runs that start from the same Qwen3-ASR weights but use different augmentation, consistency or LR settings sit in one basin. Averaging them flattens the solution and averages out run-specific processing shortcuts.
- **Sig:** small ΔWF1 (+0.2…+0.8) from heldout and echo bonafide recall, with clean flat.
- **Evidence:**
  - Model soups ([2203.05482](https://arxiv.org/abs/2203.05482), ICML'22): the greedy soup beats the best single model ID and OOD at no inference cost (ViT-G 90.94 vs 90.78). Code: `github.com/mlfoundations/model-soups`.
  - Use **plain** averaging. TIES/DARE sparsification gave the worst unseen-task accuracy, and plain task arithmetic beat TIES OOD (SuperMerge [2412.10445](https://arxiv.org/abs/2412.10445)).
- **Implementation:**
  - New `scripts/greedy_soup.py`. Candidates are each run's `top5_by_wf1/merge_list.txt`; each ingredient is expanded to its checkpoint list and the averaging reuses `main_eval.merge_checkpoints`. Candidate pool: B3a, B3c, B2e, B2f, B3e, B3b, plus B4 runs once available.
  - Sort the candidates by half_A WF1. Add one only if **half_A** heldout+echo WF1 does not drop.
  - Selecting on half_A and reporting on half_B keeps the selection noise out of the readout.
  - Write the result to `soup_list.txt` in the `merge_list.txt` format, so `main_eval.py --merge_list` and `/post-train-eval` work unchanged.
  - Sanity-check **before** soup selection: run-level merges with mixed BN statistics. The SLS head has BN2d, and averaging running stats is fine for same-init runs. If WF1 collapses, recompute BN stats on one pass over training data.
- **Rules:** weight averaging of the same model is allowed (rules matrix).
- **Cost:** eval only, about 0.5 GPU-h per candidate scored.
- **Kill:** the greedy soup stops at one ingredient, or ΔWF1 is inside ±0.5.

#### A1 · WiSE-FT-LP: interpolate the encoder toward the pretrained weights, then refit the head (run #1)

- **H:** full fine-tuning has moved the encoder away from Qwen3-ASR's noise-robust ASR features and toward "processing ⇒ spoof". Moving part of the way back restores invariance to unfamiliar DSP.
- **Sig:** heldout and echo EER go down and bonafide recall goes up. Clean F1 may fall slightly at low α. The fixed 0.5 threshold may shift, hence the head refit.
- **Evidence:**
  - WiSE-FT ([2109.01903](https://arxiv.org/abs/2109.01903), CVPR'22) gains +4–6 pp under distribution shift and +1.6 pp ID at zero cost. α=0.5 is the recommended default when nothing is known about the target.
  - WiSE-FT-LP ([2404.16422](https://arxiv.org/abs/2404.16422)) is the variant for backbones **without a zero-shot head**, which is our case: interpolate the backbone, then linear-probe the head on the interpolated backbone.
  - On the audio side, frozen SSL + a light head regularises cross-corpus generalization ([2608.01796](https://arxiv.org/abs/2608.01796)). Our own frozen-vs-FT table shows the same direction for the 0.6B and XLS-R models.
- **Implementation:**
  - New `scripts/wise_ft.py --ft <ckpt or merge_list> --alphas 0.5 0.6 0.7 0.8 0.9 --out <dir>`.
  - **θ_pre:** instantiate `ModelSLS` with the REF args, which loads the hub Qwen3-ASR weights, and take its `ssl_model.*` tensors.
  - **θ_ft:** the REF top-5 merge, via `main_eval.merge_checkpoints`.
  - **Encoder keys:** `(1-α)·θ_pre + α·θ_ft` in float64, cast back. **Head keys (SLS, BN, fc1/fc2):** copied from θ_ft.
  - Variant **"kept"**: score directly with `scripts/b3_local_eval.py`.
  - Variant **"refit"**: for each α ∈ {0.6, 0.8}, run `python -m b3wave.train <REF args> --freeze_ssl --model_path <interp_α.pth> --num_epochs 2 --lr_scheduler cosine --warmup_frac 0.05`, with the same augmentation and consistency.
  - `--freeze_ssl` and `--model_path` already exist in `main_train.py`. Check that `b3wave.train` honours `--freeze_ssl` and keeps BN in train mode for the head.
  - Smoke test: at α=1.0 the output must be bit-identical to θ_ft (`wise_ft.py --check`).
- **Rules:** a single model whose weights are an average of two checkpoints of the same architecture, so allowed. The pretrained weights are Qwen3-ASR, which has no spoof training.
- **Cost:** the "kept" sweep costs 5 evals, about 2 GPU-h. "refit" costs 2 × 2 head-only epochs, about 1.5 GPU-h.
- **Kill:** no α beats REF on half_B, *and* EER on heldout does not move. A falling EER with flat WF1 means re-test with refit before killing.
- **Final stack:** apply WiSE-FT-LP to whichever soup or run wins B5 (row "final" in §5).

#### A2 · LP-FT warm start (arm B5a)

- **H:** with a random SLS head at 10× LR, the large early gradients distort the pretrained features. Starting full FT from a converged head preserves them, which helps OOD channels.
- **Sig:** heldout and echo EER go down. The encoder moves less, which can be measured as the relative L2 distance ‖θ_ft − θ_pre‖ / ‖θ_pre‖ per layer (log it). Clean flat or slightly up.
- **Evidence:** LP-FT ([2202.10054](https://arxiv.org/abs/2202.10054), ICLR'22 oral). Over 10 shift benchmarks, full FT is +2% ID but −7% OOD versus linear probing; LP-FT gets +1% ID and +10% OOD over full FT, and features change orders of magnitude less.
- **Implementation:**
  - **Stage 1:** `b3wave.train <REF args> --freeze_ssl --num_epochs 2 --lr 1e-3 --head_lr_mult 1`, i.e. head-only at a normal head LR. Keep the REF augmentation and consistency, and keep the EMA final checkpoint.
  - **Stage 2:** `b3wave.train <REF args> --model_path <stage1 final.pth> --head_lr_mult 1` (instead of 10); everything else as in REF.
  - Optional sub-arm: `--head_lr_mult 3`. Run it only if B5a is positive.
  - New launcher `scripts/run_B5a.sh` that chains both stages in one run dir (`stage1/`, `stage2/`).
  - New telemetry: per-layer relative weight drift, appended to `metrics.jsonl["b5"]`, computed once per epoch against the hub weights. This also gives a cheap H3 diagnostic for every later run.
- **Rules:** none.
- **Cost:** stage 1 (frozen encoder) ≈0.7 GPU-h, plus one screen ≈4.5 GPU-h.
- **Kill:** ΔWF1 inside ±0.5 **and** no EER drop on heldout or echo.

### Tier B: medium cost, aimed directly at bonafide false alarms

#### B1 · `rtc_v2`: bonafide-first, pair-consistent, compositional DSP chains (arm B5b)

- **H:** the model has learned that processing types it never saw in training mean spoof. Exposing **bonafide** (and spoof, equally) to new processing *types* (NS, AGC, VAD/DTX, AEC residual, PLC, CELP codecs), chained the way an RTC client chains them, decorrelates processing from the label.
- **Sig:**
  - Heldout and echo bonafide recall go up with EER down.
  - The `noise_only` bona-source → spoof rate drops.
  - Spoof recall roughly holds (≤1 pt drop).
  - Clean flat.
  - If **only** bonafide recall moves and spoof recall falls in mirror image with flat EER, it is a prior shift. Lower `--aug_bona_extra_p`.
- **Evidence:** 2606.10912 (the most damaging chains target bonafide; reverb+Opus is worst), TFCL 2607.17761 (AEC→NS→AGC→VAD chain), 2508.01467 (CELP and PLC remove traces), B-Free and DDA (the same processing on both classes, in pairs).
- **Heldout-reserve constraint (critical).**
  - `b3wave/heldout_reserve.py` reserves for the proxy sets:
    - codecs G.722 / G.726 / GSM / speex / G.711
    - ffmpeg `afftdn`, `dynaudnorm`, `anlmdn`, `acompressor`, `compand`, `agate`, `crystalizer`
    - the `utils/sim_ops.py` stages `vad_gate`, `packet_loss`, `bandlimit`, `resample8k`
    - MUSAN music
  - The training chain must use **different implementations**. Then the proxy sets measure transfer to an unseen implementation of the same *kind* of processing, which is exactly the eval situation (5 unseen platforms).

  | Stage | Training implementation (new) | Reserved implementation (proxy only) |
  |---|---|---|
  | Echo + AEC residual | far-end RIR echo (training RIR pool) → WebRTC **AEC3** with a mis-tuned delay / short tail, via `webrtc-audio-processing` Python bindings | `sim_ops.aec_residual` |
  | NS | WebRTC APM NS (levels 0–3), **RNNoise** (`rnnoise` binding), **DeepFilterNet** (`--aug_use_deepfilternet` exists) | `afftdn`, `anlmdn` |
  | AGC | WebRTC **AGC2** (adaptive digital, random target level and max gain) | `dynaudnorm`, `acompressor` |
  | VAD / DTX | **WebRTC VAD** (`webrtcvad`, aggressiveness 0–3) gating, with comfort noise during muted frames | `sim_ops.vad_gate` |
  | Codec | **AMR-WB** (ffmpeg `libvo_amrwbenc` or `opencore-amr`), **SILK** (via Opus `-application voip` at 6–12 kbps), **EVS** only if a licensed reference binary builds | G.722 / G.726 / GSM / speex / G.711 |
  | Packet loss | **Opus in-band PLC/FEC**: decode with dropped packets at 1/5/10/20% (opuslib `decode(..., fec=True)`) | `sim_ops.packet_loss` |
  | Band-limit | Opus bandwidth switching (NB/WB/SWB) inside the encoder | `sim_ops.bandlimit`, `resample8k` |

- **Implementation:**
  - `b3wave/augment.py`: add `RTCAugmenterV2(RTCAugmenterB3)` with `--aug_chain {b3,rtc_v2}` (default `b3`, i.e. REF behaviour). The chain is **acoustic** (noise | echo | reverb, from the existing pools) → `aec` → `ns` → `agc` → `vad` → `codec` → `plc`.
  - Each processing stage is drawn independently with `--aug_v2_stage_p` (default 0.4), so the chains are compositional; there is no fixed order beyond the RTC order.
  - `--aug_bona_extra_p P` (default 0): extra probability that a bonafide clip receives a chain even when the base draw said no. This is "bonafide-first" without changing what spoofs see.
  - **Pair consistency:** when `--consistency_pairs twin`/`both` produces a pair, both views share one `chain_seed`, so they get the **same** stage set but independent noise realisations. Store `chain_seed` in the per-sample `meta` vector.
  - Extend `assert_training_ops` in `b3wave/heldout_reserve.py` with the new op names. Add an explicit allowlist so that a typo cannot silently select a reserved op.
  - New telemetry: stage counts per class per epoch (`Aug/v2/<stage>/<label>`).
  - Smoke tests in `b3wave/checks.py`:
    - `augmenter_v2` is deterministic for a fixed seed.
    - There is no reserved op.
    - Both views of a pair use the same stage set.
    - Output length and level stay within ±1 dB of loudness-norm.
  - Throughput: WebRTC APM and the AMR codec are CPU-bound. Measure clips/s with `--num_workers 8–12`. If the augmentation is slower than the GPU step, pre-render 2–3 chain variants per training clip offline (`scripts/prepare_rtc_aug_data.py` pattern).
  - Launcher: `scripts/run_B5b.sh` = REF + `--aug_chain rtc_v2 --aug_v2_stage_p 0.4 --aug_bona_extra_p 0.2`.
- **Rules:** local WebRTC AEC/NS/AGC/VAD simulation is explicitly allowed (rules matrix), and so are software codecs. DeepFilterNet and RNNoise are neural, but they are speech enhancers applied with the label kept, *not* resynthesis. Still, list them in the organiser email if they are not covered yet.
- **Cost:** about 1.5–2 days of engineering (bindings, chain, tests), then one screen at about 5–6 GPU-h with augmentation overhead.
- **Kill:** heldout bonafide recall does not improve by ≥3 pts **and** EER does not move, or clean F1 drops by more than 0.5.

#### B2 · PROBE-style hard-chain mining (arm B5c = B5b + mining)

- **H:** a few chain configurations cause most of the bonafide false alarms. Oversampling those, using **training** bonafide audio only, is an adversarial augmentation search without any extra training loop.
- **Sig:** larger bonafide-recall gain on heldout and echo than B5b, at the same EER improvement or better.
- **Evidence:** the PROBE benchmark ([2605.24906](https://arxiv.org/abs/2605.24906)) shows that failures concentrate in under-covered parts of the generative space. 2606.10912 finds the most damaging chains target bonafide audio. The roadmap's worst-of-k item (Gontijo-Lopes et al.) is a looser version of the same idea.
- **Implementation:**
  - `scripts/b5_mine_chains.py`: sample about 3k **training** bonafide clips (stratified by lang × src). For each `rtc_v2` stage combination plus parameter bucket (≈60–100 cells), render once and score with REF. Write `chain_weights.json` with a weight per cell proportional to `FA_rate^τ` (τ=1), floored at 0.2× uniform.
  - `--aug_v2_chain_weights PATH` in `RTCAugmenterV2` samples the chain cell from these weights. The default is uniform, i.e. B5b behaviour.
  - Do **not** mine on any dev, proxy or eval audio: training clips only, to keep the proxy sets clean.
- **Rules:** training data plus local simulation only, so fine.
- **Cost:** mining ≈1–2 GPU-h of scoring, then one screen.
- **Kill:** no gain over B5b beyond ±0.5.

#### B3 · Nes2Net-X head (arm B5d)

- **H:** the SLS → BN → maxpool → flatten-FC head is length-locked and position-sensitive, and it compresses features crudely. A multi-scale nested back-end that keeps the full feature dimension gives better-separated representations under shift.
- **Sig:** EER down on every noisy set. It allows full-utterance or multi-crop inference (`--multicrop mean` already exists in `b3wave.eval`).
- **Evidence:**
  - Nes2Net / Nes2Net-X ([2504.05657](https://arxiv.org/abs/2504.05657), IEEE TIFS; v2 Oct 2025): ~26% relative In-the-Wild EER reduction vs SLS on the same wav2vec2 front-end, with 87% less back-end compute. The headline 22% figure in the abstract is on CtrSVDD.
  - Code and pretrained back-ends: `github.com/Liu-Tianchi/Nes2Net`.
- **Implementation:**
  - `b3wave/model.py`: add `ModelNes2NetB5(ModelSLS)` selected by `--head {sls,nes2netx}` (default `sls`). Its input is the softmax-normalised layer-weighted sum of the Qwen3-ASR hidden states, shape (B, T, D). Port the `Nes2Net_X` module from the official repo (MIT license; verify).
  - The state_dict keys differ, so `select_topk.py` / `merge_checkpoints` still work, but `main_eval.py` needs `--head nes2netx` passed through. Add it to `b3wave.eval`.
  - Train with A2 (LP-FT stage 1 for the new head) if B5a was positive. Otherwise use REF optimisation with `--head_lr_mult 10`.
  - Smoke test: forward pass on 2 s, 4.04 s and 8 s inputs, and a state_dict round-trip.
- **Rules:** none (single model).
- **Cost:** about 1 day to port, then one screen.
- **Kill:** ΔWF1 inside ±0.5 **and** no EER gain. Then MHFA+DSU is the only other head to try in B5.

#### B4 · DSU / MixStyle feature-statistics perturbation (arm B5e)

- **H:** unseen platforms change per-utterance feature statistics (EQ, gain, codec colouration). Perturbing or mixing the time-mean and time-std of hidden states in training makes the head invariant to them.
- **Sig:** heldout and echo bonafide recall up, clean ≈ flat or −0.3. Little effect on unseen-generator slices.
- **Evidence:**
  - BUT WildSpoof ([2512.12851](https://arxiv.org/abs/2512.12851)) and BUT ESDD 2026 ([2512.08319](https://arxiv.org/abs/2512.08319)) apply DSU to the **value** stream of an MHFA head with temporal statistics, and report clear gains on unseen domains.
  - MixStyle for single-domain DG in ADD ([2407.03654](https://arxiv.org/abs/2407.03654), top-5 ADD Track 1.2).
  - The evidence for unseen *generators* is null, and in-domain results can drop slightly.
- **Implementation:**
  - `b3wave/model.py`: a `FeatStyle` module applied to the stacked per-layer hidden states **before** SLS fusion, in training mode only. Flags:
    - `--feat_style {none,dsu,mixstyle}` (default `none`)
    - `--feat_style_p 0.5`
    - `--feat_style_alpha 0.1` (Beta parameter for MixStyle)
  - **DSU:** μ, σ over time per (layer, channel); sample μ' = μ + ε·Σ_μ and σ' = σ + ε·Σ_σ, with Σ the batch std of μ and σ.
  - **MixStyle:** shuffle the batch and mix the statistics with λ ~ Beta(α, α). With `--consistency_pairs twin`, the offline↔online twins in the batch give natural cross-channel style pairs.
  - Smoke test: identity in `eval()` mode, and a shape test.
- **Rules:** none.
- **Cost:** about half a day of code, then one screen.
- **Kill:** inside ±0.5, or clean F1 drops by more than 0.5.

### Tier C: higher cost or weaker evidence (only if Tier A/B saturate, or as B6 candidates)

| ID | Direction | Evidence | Sketch | Risk / cost |
|---|---|---|---|---|
| C1 | **First-order MLDG meta-learning** | [2502.10838](https://arxiv.org/abs/2502.10838): avg EER over 6 corpora 8.84 → 5.30 with rank-16 LoRA (α=2) on q/k/v/o, backbone frozen. Most of the gain comes from the meta-objective. | Domains = {offline, P01, P02, each `rtc_v2` chain family}. Each step: meta-train on all but 1–2 domains, take one virtual step, add the held-out-domain loss (first-order). Start with LoRA via `peft` (roadmap §13.6). | ~2× step cost, untested with full FT. Best entry point for a LoRA track. |
| C2 | **Multi-centre one-class loss** (SAMO / OC-softmax) | SAMO [2211.02718](https://arxiv.org/abs/2211.02718) beats OC-softmax [2010.13995](https://arxiv.org/abs/2010.13995) (m_real 0.9, m_fake 0.2, α 20; code `yzyouzhang/AIR-ASVspoof`). | One bonafide attractor per platform or chain cluster, as an auxiliary loss next to CE. | No verified evidence for channel shift. A single-centre OC loss can **raise** channel false alarms. |
| C3 | **Model Ratatouille** | [2212.10445](https://arxiv.org/abs/2212.10445) (ICML'23): SOTA DomainBed by averaging target fine-tunes that start from different auxiliary fine-tunes. | Auxiliary tasks on training audio only: platform-ID, chain-type, SNR bucket. Fine-tune the detector from each, then average. | K+K runs. Email the organisers if unsure about "training-time ensembles". |
| C4 | **Freqmix + EMA-teacher KD** | [2406.09664](https://arxiv.org/abs/2406.09664) (telephony fake speech). | Swap one STFT band across the batch, then ISTFT. The EMA teacher sees the mixed view and distils into the student. | Low–moderate cost. Overlaps with B4. |
| C5 | **Channel-invariant embedding** (CausalCLIP-style) | [2512.13285](https://arxiv.org/abs/2512.13285) | Adversarial or orthogonality loss that removes chain or platform information (from `meta`) from the fused embedding. | 2609.09489 found that CORAL-style alignment hurt, so do a probe first. |
| C6 | **Mi-Dasheng 0.6B backbone probe** | [2512.12851](https://arxiv.org/abs/2512.12851): beat WavLM Base+ out of domain (it was not compared with Qwen3-ASR or w2v-BERT 2.0). | `scripts/layerwise_logreg_probe.py` on frozen features over `sim_heldout`, before any full run. | Probe only. Verify that the training data contains no spoof audio. |

### Deprioritised (with evidence; do not spend B5 compute)

| Idea | Why |
|---|---|
| SAM / ASAM | With strong SSL backbones (XLS-R) the gains are small, mixed or negative. The suggested reason is that pretraining already gives a flat feature space ([Huang et al., Interspeech'25](https://www.isca-archive.org/interspeech_2025/huang25e_interspeech.html)). It also doubles compute. If tried anyway: ρ 0.001–0.005, code `nii-yamagishilab/SAM-AntiSpoofing`. |
| MoLEx (mixture of LoRA experts) | Roughly ties a plain full FT with augmentation, and depends on a fragile orthogonality loss and the chosen rank ([2509.09175](https://arxiv.org/abs/2509.09175)). |
| TIES / DARE merging | Worse OOD than plain averaging or task arithmetic (2412.10445). |
| Calibration, bias or threshold tricks | Leave EER unchanged (2606.21584, 2507.01750), and our oracle gap is about 0.9 F1. |
| Audio-LLM QA framing | ALLM4ADD ([2505.11079](https://arxiv.org/abs/2505.11079)) is strong in-domain, but out-of-domain results stay weak ([2601.00777](https://arxiv.org/abs/2601.00777)). |
| AI-text detector ideas (Binoculars score ratios) | Brittle to surface perturbation: about 81% → 38–44% TPR under substitution and homoglyph attacks ([2609.29935](https://arxiv.org/abs/2609.29935)). The audio counterpart of surface perturbation is channel/DSP, our core problem. Inspiration only. |

---

## 5. Run table

REF = the B4 winner, or B3a. Each arm is seed 1234 first; add seed 2 only for arms that clear the band.

| # | ID | Change vs REF | Depends on | Cost | Primary readout |
|---|---|---|---|---|---|
| 0 | B5-soup | A3 greedy cross-run soup | existing runs (+B4) | eval ≈3 GPU-h | half_B WF1, heldout and echo bonafide recall |
| 1 | B5-wise | A1 WiSE-FT α sweep, "kept" + "refit" at α∈{0.6, 0.8} | REF merge | ≈3.5 GPU-h | heldout and echo EER vs α |
| 2 | B5a | A2 LP-FT warm start, `--head_lr_mult 1` | — | ≈5 GPU-h | half_B WF1, per-layer drift |
| 3 | B5b | B1 `--aug_chain rtc_v2 --aug_bona_extra_p 0.2` | B1 engineering | ≈6 GPU-h | heldout bonafide recall, `noise_only` probe |
| 4 | B5c | B5b + B2 `--aug_v2_chain_weights` | B5b, mining script | ≈1.5 + 6 GPU-h | same as B5b |
| 5 | B5d | Nes2Net-X head, on the best of {REF, B5a, B5b/c} | port | ≈5 GPU-h | EER on all noisy sets, multicrop |
| 6 | B5e | DSU feature style on the current winner | — | ≈5 GPU-h | heldout and echo bonafide recall |
| final | B5-final | WiSE-FT-LP (A1) + greedy soup (A3) over the B5 winners and seeds | all | eval | half_B, then `sim_unseen_v1` once |

The engineering for B1 and B3 can run on CPU while rows 0–2 use the GPUs.

---

## 6. Rule checks and organiser email

- **Allowed as is:** WebRTC APM (AEC3 / NS / AGC2 / VAD), AMR-WB, SILK and Opus PLC are local DSP/codec simulation, which the rules allow explicitly. So are weight averaging (soups, WiSE-FT) and training-time-only methods (MLDG, DSU, mining).
- **Include in the next organiser email** (rtcsddchallenge2027@gmail.com):
  1. RNNoise and DeepFilterNet as label-preserving augmentation stages.
  2. Model Ratatouille, i.e. averaging fine-tunes that start from auxiliary-task fine-tunes.
  3. EVS reference-codec usage, if built.
- **Reserve integrity:** extend `b3wave/heldout_reserve.py` with the new op names and an allowlist. The B5b smoke test must fail if any reserved op is drawn.

---

## 7. Implementation checklist (for the next iteration)

| File | Change | Smoke test |
|---|---|---|
| `scripts/greedy_soup.py` (new) | A3 | 1-ingredient soup == that ingredient |
| `scripts/wise_ft.py` (new) | A1 | α=1 is bit-identical to θ_ft |
| `scripts/run_B5a.sh` (new) | A2 two-stage launcher | `b3_smoke_train.sh` variant with 2 tiny stages |
| `SLS_setup/b3wave/train.py` | `metrics.jsonl["b5"]` per-layer weight drift | logged keys present |
| `SLS_setup/b3wave/augment.py` | `RTCAugmenterV2`, `--aug_chain`, `--aug_v2_stage_p`, `--aug_bona_extra_p`, `--aug_v2_chain_weights` | determinism, pair-shared stage set, reserved-op guard |
| `SLS_setup/b3wave/heldout_reserve.py` | new op names + allowlist | assert fires on a reserved op |
| `scripts/b5_mine_chains.py` (new) | B2 mining on training bonafide clips only | refuses any path under `data/dev_noisy_sim` or `data/augmentation_eval` |
| `SLS_setup/b3wave/model.py` | `--head nes2netx`, `--feat_style` | variable-length forward, eval-mode identity |
| `SLS_setup/b3wave/eval.py`, `scripts/b3_local_eval.py` | pass `--head` through | scores match the training-time local eval |
| `SLS_setup/b3wave/README.md` | document the new flags (default = REF) | — |

---

## 8. References

**Audio, primary to the failure mode**
- 2606.10912 · *What Do Deepfake Speech Detectors Actually Hear?* (2026)
- 2606.21584 · Threshold transfer audit for speech deepfake detectors (2026)
- 2609.09489 · Audio deepfake detection using temporal coherence analysis (2026)
- 2507.01750 · Generalizable detection of audio deepfakes (2025)
- 2607.17761 · TFCL, Time-Frequency Consistency Learning (ACM MM 2026)
- 2508.01467 · Multi-granularity adaptive TF attention under real-world communication degradations (2025)
- 2503.02857 · Deepfake-Eval-2024 (2025)

**Audio, methods**
- 2504.05657 · Nes2Net / Nes2Net-X (IEEE TIFS)
- 2502.10838 · Meta-learned LoRA (MLDG) for generalizable SDD
- 2512.12851 · BUT WildSpoof systems (DSU in MHFA; Mi-Dasheng)
- 2512.08319 · BUT ESDD 2026 systems
- 2407.03654 · MixStyle-based DG (ADD); the fetched v1 page was a DCASE SED system, so verify the exact ID before citing
- 2406.09664 · Freqmix + KD for telephony fake speech
- 2010.13995 · OC-softmax
- 2211.02718 · SAMO
- 2509.09175 · MoLEx
- Huang et al., Interspeech 2025 · SAM for SDD
- 2505.11079 · ALLM4ADD
- 2601.00777 · audio-LLM OOD (IJCB 2025)
- 2608.01796 · multi-backbone SSL linear probes
- 2508.20983 · SAFE challenge system (WavLM + AASIST + RawBoost)

**Vision and text transfer**
- 2109.01903 · WiSE-FT
- 2404.16422 · WiSE-FT-LP
- 2202.10054 · LP-FT
- 2203.05482 · Model soups
- 2212.10445 · Model Ratatouille
- 2412.10445 · SuperMerge (TIES/DARE OOD)
- 2605.24906 · PROBE
- 2412.17671 · B-Free
- 2505.14359 · DDA
- 2512.13285 · CausalCLIP
- 2609.29935 · robust LLM-text detection under contamination
