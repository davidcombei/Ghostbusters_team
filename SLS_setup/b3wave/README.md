# b3wave: the B3-wave training / evaluation package

Additive package for the B3 research wave. **Nothing in `SLS_setup/main_train.py`, `main_eval.py`,
`utils/`, `model/` or the existing `scripts/*.py` is changed**, so every B0 / B1 / B2 run stays
reproducible from the original scripts. Checkpoints are plain `ModelSLS` state_dicts, so
`scripts/select_topk.py`, `scripts/local_eval.py`, `scripts/compare_evals.py`, `main_eval.py`,
`scripts/make_submission.py` and the `post-train-eval` / `log-experiment` skills work on B3 runs
unchanged.

```
cd SLS_setup
python -u -m b3wave.train <every main_train.py argument> [B3 flags]      # trainer (= main_train_b3.py)
python -m b3wave.eval  <main_eval.py arguments> --multicrop mean          # progress-split scoring
python ../scripts/b3_local_eval.py ... --multicrop mean                   # local report (+ multi-crop)
python ../scripts/b3_compare_evals.py <base> <cand> --labels A B --ci 2000  # comparison + bootstrap CI
python ../scripts/b3_log_ci.py --id <registry id> --ci_json <cand>/compare_ci.json
python ../scripts/b3_build_unseen_set.py                                  # frozen sim_unseen_v1
bash   ../scripts/b3_smoke.sh [cpu]                                       # all smoke tests
```

**Reproduction statement.** `python -u -m b3wave.train` with B2e's exact command line
(`exp/B2e_*/config.yaml: command`) prints B2e's header lines and draws the same augmentations:
every flag below defaults to the original behaviour, the augmenter delegates to the original
`_stage` when nothing new is on (bit-identical outputs, `b3wave.checks augmenter`), the model
subclass has the same state_dict keys (`b3wave.checks model`) and the original two-group optimizer
is used unless `--llrd` is set.

## Flags (all new; default = B2e behaviour)

| flag | default | what it changes | module | smoke test |
|---|---|---|---|---|
| `--sampler {shuffle,balanced}` | shuffle | balanced: weighted draw with replacement, `len(dataset)` items/epoch (steps unchanged), a `--sampler_bona_frac` share of bonafide | `sampler.py` | checks sampler; train c |
| `--sampler_bona_frac F` | 0.5 | bonafide share of the draws | `sampler.py` | checks sampler |
| `--sampler_lang_balance` | off | each class drawn half en / half zh | `sampler.py` | checks sampler; train c |
| `--aug_snr_db LO HI` | 5 20 | SNR range of every noise stage (incl. MUSAN) | `augment.py` | train d |
| `--aug_snr_dist {uniform,beta}` | uniform | beta: Beta-shaped draw with mode `--aug_snr_beta_mode` (concentration 4) | `augment.py` | checks augmenter; train d |
| `--aug_snr_beta_mode M` | 8.0 | mode of the beta draw, dB | `augment.py` | train d |
| `--aug_max_stages K` | 1 | now functional: U{1..K} distinct acoustic stages chained (noise -> echo -> reverb -> suppress) | `augment.py` | checks augmenter; train e |
| `--aug_codec_profiles {fixed,random}` | fixed | random: per-clip Opus bitrate {6,8,12,16,24} kbps, frame {20,40,60} ms, VBR mode, packet-loss hint (ffmpeg 4.2: no FEC/DTX options) | `augment.py` | train j |
| `--consistency {none,js,js_emb}` | none | two views per clip in ONE forward pass; CE on both + `lambda_js` JS [+ `lambda_emb` (1 - cos)] | `data.py`, `losses.py`, `model.py`, `train.py` | checks losses; train f |
| `--consistency_lambda L` | 0.3 | JS weight | `losses.py` | train f |
| `--consistency_emb_lambda L` | 0.3 | embedding-cosine weight (`js_emb`) | `losses.py` | train f |
| `--consistency_pairs {twin,aug2,both}` | both | twin = online twin (clips without one dropped), aug2 = same clip twice, both = twin else aug2 | `data.py` | checks data; train f |
| `--consistency_clean_view` | off | view a un-augmented and detached (clean teacher) | `data.py`, `losses.py` | checks losses |
| `--pair_csv PATH` | RTCFake `train_offline_online_pairs.csv` | twin map (prefix `train/` stripped; 37,125 pairs) | `data.py` | checks data |
| `--llrd D` | 1.0 | layer-wise LR decay: layer i at `lr * D^(L-1-i)`, stem at `lr * D^L`, final norm and head unchanged; `param_groups[0]` = top layer | `optim.py` | checks optim; train g |
| `--freeze_bottom_k K` | 0 | freeze the conv stem and the first K layers | `optim.py` | checks optim |
| `--specaug` | off | time/frequency masking of the log-mel input, training only, torch RNG | `model.py` | checks model; train h |
| `--specaug_time N MAX` | 2 40 | N time masks of <= MAX frames | `model.py` | train h |
| `--specaug_freq N MAX` | 2 16 | N mel masks of <= MAX bins | `model.py` | train h |
| `--train_subset_frac F` | 1.0 | stratified (label x lang x src) training subset; smoke tests | `data.py` | checks data; train a-k |
| `--loudness_norm`, `--loudness_norm_lufs` | off / -23 | inherited from the current `main_train.py` (read-time BS.1770 normalisation, train + dev + eval) | `utils/loudness.py` (unchanged) | train i |

Eval-side: `--multicrop {first,mean,median}` and `--multicrop_max` (default 8) in `b3wave.eval` and
`scripts/b3_local_eval.py`; `first` is bit-identical to the original scoring (smoke test 10).

## Telemetry (always on in `b3wave.train`)

- Header: `b3wave <version>`, `Train composition`, `Sampler`, `SNR | Chaining | Codec profiles`,
  `Consistency`, `LLRD`, `SpecAugment`, `Subset` lines; `config.yaml` gains `b3wave_version`,
  `train_info` (composition, pairs found, subset) and `sampler_expected_shares`.
- Per epoch (`logs/train.log`, TensorBoard, `metrics.jsonl["b3"]`): CE / JS / embedding loss
  components, realised bona and en shares of the sampled epoch, augmentation stage counts and the
  SNR histogram (`Aug/*`, from the per-clip `meta` count vector; SNR bins are counted only on the
  B3 augmenter path, i.e. when chaining or the beta draw is on).

## Reserved for the proxy sets (never in training)

`b3wave/heldout_reserve.py` extends `utils/heldout_reserve.py`: codecs G.722 / G.726 / GSM /
speex / G.711; filters anlmdn, acompressor, afftdn, dynaudnorm, vad_gate, packet_loss, bandlimit,
compand, agate, crystalizer; dirs `data/augmentation/musan/music` (unseen-set noise) and
`data/augmentation_eval`. `RTCAugmenterB3` asserts this at start-up (smoke test k).

## Unseen set

`scripts/b3_build_unseen_set.py` renders `data/dev_noisy_sim/unseen_v1` from the half-B sources:
MUSAN music at {-5..25} dB, compand / agate / crystalizer / 8 kHz round trip, speex or G.711
mu-law, packet loss; registered as `sim_unseen_v1` (`role: unseen`) in `configs/local_eval_sets.yaml`.
It never enters WF1 and is meant to be read only at the end of a wave.

## Smoke tests

`bash scripts/b3_smoke.sh` runs: 1 isolation (`git diff` empty, only new B3 paths), 2 CLI, 3-6
`python -m b3wave.checks all` (augmenter RNG reproduction, protocol metadata counts, sampler
shares, loss properties), 7-8 `checks model` / `checks optim` (GPU), 9 bootstrap self-tests on the
B2 evals, 10 multi-crop equivalence, 11 `scripts/b3_smoke_train.sh` (1-epoch 2 % subset runs of
every lever into `SLS_setup/exp_smoke/`), 12 the unseen-set builder with `--limit 24`.
