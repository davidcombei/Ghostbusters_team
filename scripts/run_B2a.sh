#!/usr/bin/env bash
# B2a = B1 (Qwen3-ASR-1.7B, RTC aug + MUSAN + level aug, AdamW, 10x head LR, warmup -> cosine,
#      grad clip, EMA) with the loss/schedule misconfigurations removed:
#        - no --label_smoothing  (with ce_weights 0.1/0.9 it shifted the boundary and made dev loss meaningless)
#        - 8-epoch schedule that actually anneals (B1: cosine over 30 epochs, early-stopped at 13 at 64% of peak LR)
#        - warmup 5% (B1: 3% of 30 epochs = 0.9 epoch, the best epoch was the warmup epoch)
#        - no early stop on a cosine run; top-7 epochs by local WF1 kept on disk.
#      Iteration 1 of the B2 wave (docs/experiments-exploration/).
#
#   cd /root/repos/rtc-sdd/Ghostbusters_team
#   bash scripts/run_B2a.sh                                        # fresh run
#   bash scripts/run_B2a.sh --resume exp/B2a_qwen17_..._<ts>       # after a container restart
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_B2a.sh                 # pick the GPU
#
# Extra arguments are appended to the main_train.py command line. Training runs under nohup;
# its stdout/stderr go to SLS_setup/<track>_<timestamp>.log (one file per launch), and the
# per-epoch summary is also in SLS_setup/exp/<run>/logs/train.log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
TRACK="B2a_qwen17_rtc_musan_lvl_cos8_s1234"
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
    --optim adamw --lr 3e-5 --head_lr_mult 10 --weight_decay 0.01 \
    --lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 1e-6 \
    --grad_clip 1.0 --ema_decay 0.9995 \
    --local_eval_sets ../configs/local_eval_sets.yaml \
    --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 \
    --seed 1234 --track "$TRACK" \
    "$@" > "$LOG" 2>&1 &

PID=$!
echo "Started $TRACK (pid $PID)"
echo "Log:     $REPO/SLS_setup/$LOG"
echo "Follow:  tail -f $REPO/SLS_setup/$LOG"
echo "Stop:    kill $PID"
