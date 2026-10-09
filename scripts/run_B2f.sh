#!/usr/bin/env bash
# B2f = B2e with ONE factor changed: the level augmentation is replaced by loudness normalisation,
#      --use_level_aug -> --loudness_norm --loudness_norm_lufs -23 (utils/loudness.py, commit 46461b8).
#      Level aug (training only, p=0.8) sets a random active level (-38..-12 dBFS) and runs a random
#      AGC / compressor / limiter so absolute level is uninformative. Loudness norm is deterministic:
#      every clip (train, dev AND eval) is scaled to -23 LUFS (ITU-R BS.1770) in SpoofAudioDataset._load,
#      before the RTC augmenter, so the model sees one input level distribution at train and test time.
#      It removes the level cue by construction but trains no robustness to AGC / compressor dynamics.
#      Evidence: docs/experiments-exploration/20261008_232331_B2e_*.md (B2e half_B WF1: merge 92.34,
#      final-epoch EMA 92.53; level probes flip 0.30 -> ~0.0x of bona at +10 dB with level aug).
#      Falsification: final-epoch half_B WF1 within +-0.5 of B2e's 92.53 -> null (the two level treatments
#      are equivalent); below -> the stochastic dynamics of level aug matter, keep level aug; above -> promote
#      loudness norm (and consider stacking both as a follow-up screen).
#      Everything else as B2e: 8 epochs, warmup 5%, lr 2.5e-6 / lr_min 5e-7, head LR x10, RTC aug + MUSAN,
#      AdamW wd 0.01, grad clip 1, EMA 0.9995, no label smoothing, no early stop, top-7 ckpts kept.
#      --keep_epochs 8 is bookkeeping only (the final EMA epoch can never be pruned).
#
#   cd /root/repos/rtc-sdd/Ghostbusters_team
#   bash scripts/run_B2f.sh                                        # fresh run
#   bash scripts/run_B2f.sh --resume exp/B2f_qwen17_..._<ts>       # after a container restart
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_B2f.sh                 # pick the GPU
#
# Extra arguments are appended to the main_train.py command line. Training runs under nohup;
# its stdout/stderr go to SLS_setup/<track>_<timestamp>.log (one file per launch), and the
# per-epoch summary is also in SLS_setup/exp/<run>/logs/train.log.
#
# Evaluation of this run MUST pass the same flags (--loudness_norm --loudness_norm_lufs -23) to
# scripts/local_eval.py and main_eval.py; scripts/eval_checkpoints.py reads them from config.yaml.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
TRACK="B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234"
RTCFAKE="/mnt/paml-research/RTCFake/data"

# A container restart wipes the system ffmpeg; the codec stage would then silently pass audio through.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found on PATH: the RTC codec augmentation would silently no-op. Reinstall it first." >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "python not found: $PY (set PY=... to override)" >&2
    exit 1
fi
# --loudness_norm needs pyloudnorm in the run env (imported only when the flag is on).
if ! "$PY" -c "import pyloudnorm" >/dev/null 2>&1; then
    echo "pyloudnorm not importable by $PY: run '$PY -m pip install pyloudnorm==0.2.0' first." >&2
    exit 1
fi

cd "$REPO/SLS_setup"
LOG="${TRACK}_$(date +%Y%m%d%H%M%S).log"

nohup "$PY" -u main_train.py \
    --train_data_path "$RTCFAKE/wav/train" \
    --train_protocol  "$RTCFAKE/train_label.txt" \
    --dev_data_path   "$RTCFAKE/wav/dev" \
    --dev_protocol    ../data/dev_noisy_sim/dev_label_excl_halfB.txt \
    --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 \
    --batch_size 16 --num_epochs 8 --num_workers 64 \
    --use_rtc_aug \
    --aug_noise_dirs ../data/augmentation_train/rnnoise ../data/augmentation_train/esc50 \
    --aug_rir_dirs ../data/augmentation_train/rirs \
    --musan --musan_dir ../data/augmentation/musan \
    --loudness_norm --loudness_norm_lufs -23 \
    --optim adamw --lr 2.5e-6 --head_lr_mult 10 --weight_decay 0.01 \
    --lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 5e-7 \
    --grad_clip 1.0 --ema_decay 0.9995 \
    --local_eval_sets ../configs/local_eval_sets.yaml \
    --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 --keep_epochs 8 \
    --seed 1234 --track "$TRACK" \
    "$@" > "$LOG" 2>&1 &

PID=$!
echo "Started $TRACK (pid $PID)"
echo "Log:     $REPO/SLS_setup/$LOG"
echo "Follow:  tail -f $REPO/SLS_setup/$LOG"
echo "Stop:    kill $PID"
