#!/usr/bin/env bash
# B2d = B2c with ONE change: --num_epochs 8 -> 10 (the screening cap), same warmup 5% -> cosine to 1e-6.
#      Iteration 4 of the B2 wave. Evidence (docs/experiments-exploration/20261008_131754_B2c_*.md):
#      at lr 5e-6 the per-epoch mini WF1 still rose from epoch 4 to 7 (+1.43) and the dev loss was
#      still falling at epoch 8, so the schedule length is the lever to test before the full run.
#      The analysis agent ranked a second seed first (noise floor); the user's brief for the loop is to
#      explore training configurations, so the seed repeat is kept for iteration 5 if this one lands
#      inside the +-1.5 band.
#      Everything else as B2c: lr 5e-6 / head 5e-5, RTC aug + MUSAN + level aug, AdamW wd 0.01,
#      grad clip 1, EMA 0.9995, no label smoothing, no early stop, top-7 ckpts kept.
#
#   cd /root/repos/rtc-sdd/Ghostbusters_team
#   bash scripts/run_B2d.sh                                        # fresh run
#   bash scripts/run_B2d.sh --resume exp/B2d_qwen17_..._<ts>       # after a container restart
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_B2d.sh                 # pick the GPU
#
# Extra arguments are appended to the main_train.py command line. Training runs under nohup;
# its stdout/stderr go to SLS_setup/<track>_<timestamp>.log (one file per launch), and the
# per-epoch summary is also in SLS_setup/exp/<run>/logs/train.log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
TRACK="B2d_qwen17_rtc_musan_lvl_cos10_lr5e6_s1234"
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
    --batch_size 16 --num_epochs 10 --num_workers 64 \
    --use_rtc_aug \
    --aug_noise_dirs ../data/augmentation_train/rnnoise ../data/augmentation_train/esc50 \
    --aug_rir_dirs ../data/augmentation_train/rirs \
    --musan --musan_dir ../data/augmentation/musan \
    --use_level_aug \
    --optim adamw --lr 5e-6 --head_lr_mult 10 --weight_decay 0.01 \
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
