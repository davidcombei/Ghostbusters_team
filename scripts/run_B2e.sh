#!/usr/bin/env bash
# B2e = B2c with ONE factor changed: the whole LR schedule halved, --lr 5e-6 -> 2.5e-6 and
#      --lr_min 1e-6 -> 5e-7 (lr_min is applied as the ratio lr_min/lr, so the cosine shape and the
#      20% floor are identical; head LR 2.5e-5 via --head_lr_mult 10).
#      Iteration 5 (last) of the B2 wave. Evidence (docs/experiments-exploration/20261008_185750_B2d_*.md):
#      the LR series 3e-5 -> 1e-5 -> 5e-6 gave +4.64 / +2.14 half_B WF1 (merge) and is still monotone;
#      B2d showed the ranking only improves once LR <= 1.4e-6 while epochs at higher LR add confidence.
#      Falsification: final-epoch half_B WF1 within +-0.5 of B2c's 92.26 -> LR series turned over, full run at 5e-6.
#      Everything else as B2c: 8 epochs, warmup 5%, RTC aug + MUSAN + level aug, AdamW wd 0.01,
#      grad clip 1, EMA 0.9995, no label smoothing, no early stop, top-7 ckpts kept.
#
#   cd /root/repos/rtc-sdd/Ghostbusters_team
#   bash scripts/run_B2e.sh                                        # fresh run
#   bash scripts/run_B2e.sh --resume exp/B2e_qwen17_..._<ts>       # after a container restart
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_B2e.sh                 # pick the GPU
#
# Extra arguments are appended to the main_train.py command line. Training runs under nohup;
# its stdout/stderr go to SLS_setup/<track>_<timestamp>.log (one file per launch), and the
# per-epoch summary is also in SLS_setup/exp/<run>/logs/train.log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
TRACK="B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234"
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
    --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 \
    --seed 1234 --track "$TRACK" \
    "$@" > "$LOG" 2>&1 &

PID=$!
echo "Started $TRACK (pid $PID)"
echo "Log:     $REPO/SLS_setup/$LOG"
echo "Follow:  tail -f $REPO/SLS_setup/$LOG"
echo "Stop:    kill $PID"
