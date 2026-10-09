#!/usr/bin/env bash
# F1 = the full-length run of the strategy selected by the B2 wave (docs/experiments-exploration/
#      20261008_232331_B2_wave_summary_and_full_run_recommendation.md): B2e's configuration unchanged
#      (Qwen3-ASR-1.7B, RTC aug + MUSAN + level aug, AdamW wd 0.01, lr 2.5e-6 / head 2.5e-5, warmup 5% ->
#      cosine over 8 epochs to 5e-7, grad clip 1, EMA 0.9995, NO label smoothing, no early stop)
#      plus --keep_epochs 8 so pruning can never delete the final EMA checkpoint, which is the one to submit.
#      lr 5e-6 / lr_min 1e-6 (B2c) is equivalent within the measured band (< 0.5 half_B WF1); pick one and
#      keep it for all seeds. Seed 1234 of this exact configuration already exists as B2e.
#
#   cd /root/repos/rtc-sdd/Ghostbusters_team
#   SEED=2345 bash scripts/run_F1.sh                               # one seed per launch (~4 h on a 4090)
#   SEED=3456 bash scripts/run_F1.sh
#   bash scripts/run_F1.sh --resume exp/F1_qwen17_..._<ts>         # after a container restart
#   CUDA_VISIBLE_DEVICES=1 SEED=2345 bash scripts/run_F1.sh        # pick the GPU
#
# After each seed: /post-train-eval SLS_setup/exp/<run> --k 5 --baseline <best eval dir>, then score
# ckpt/epoch_8_*.pth alone with scripts/local_eval.py (17 report sets); submit the final EMA checkpoint
# of the best seed, or the score average of the seeds' final checkpoints (weight averaging across seeds
# only if it beats every seed on half_B). Nothing here uses the progress set for selection.
#
# Extra arguments are appended to the main_train.py command line. Training runs under nohup;
# its stdout/stderr go to SLS_setup/<track>_<timestamp>.log (one file per launch), and the
# per-epoch summary is also in SLS_setup/exp/<run>/logs/train.log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
SEED="${SEED:-2345}"
TRACK="F1_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s${SEED}"
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
    --use_level_aug \
    --optim adamw --lr 2.5e-6 --head_lr_mult 10 --weight_decay 0.01 \
    --lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 5e-7 \
    --grad_clip 1.0 --ema_decay 0.9995 \
    --local_eval_sets ../configs/local_eval_sets.yaml \
    --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 --keep_epochs 8 \
    --seed "$SEED" --track "$TRACK" \
    "$@" > "$LOG" 2>&1 &

PID=$!
echo "Started $TRACK (pid $PID)"
echo "Log:     $REPO/SLS_setup/$LOG"
echo "Follow:  tail -f $REPO/SLS_setup/$LOG"
echo "Stop:    kill $PID"
