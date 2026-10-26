#!/usr/bin/env bash
# B3a = B2f with ONE factor changed: paired consistency training,
#      + --consistency js --consistency_lambda 0.3 --consistency_pairs both (SLS_setup/b3wave, README.md).
#      Every training clip is seen as TWO views in one forward pass: view a = the clip with its usual
#      RTC/MUSAN augmentation; view b = its online RTC twin (train_offline_online_pairs.csv, 37,125 pairs)
#      when the twin exists, otherwise the same clip augmented a second time, independently. The loss is
#      CE on both views + 0.3 * Jensen-Shannon divergence between the two posteriors, so the decision is
#      pushed to be invariant to the channel / noise / codec nuisance that separates the two views.
#      --batch_size 8 counts PAIRS with consistency on: 8 pairs = 16 clips per forward pass = B2f's GPU
#      memory; steps per epoch double (9,474 instead of 4,737), so the run takes ~2x B2f, ~8 h.
#      Evidence: roadmap T3.1 (P0, +2-5 noisy F1 expected). B2f weaknesses it targets: noisy bona recall
#      ~74 vs spoof 99; 16-20% of noisy bona saturated at p(spoof)>=0.99, unchanged by every LR / schedule
#      change of the B2 wave; noise_only bona 100% called spoof (docs/experiments-exploration/*_B2f_*.md).
#      Falsification (pre-registered): delta top-5-merge half_B WF1 vs B2f merge (91.98)
#        > +0.5 with the paired-bootstrap CI excluding 0 and clean F1 drop <= 0.5 -> promote;
#        within +-0.5 -> null; below -0.5 -> negative.
#      Expected signature: noisy EER down, noise_only bona->spoof down, heldout / matched bona recall up,
#      clean flat.
#      Everything else as B2f: 8 epochs, warmup 5%, lr 2.5e-6 / lr_min 5e-7, head LR x10, RTC aug + MUSAN,
#      loudness norm -23 LUFS, AdamW wd 0.01, grad clip 1, EMA 0.9995, no label smoothing, no early stop,
#      top-7 ckpts kept, seed 1234. --keep_epochs 8 is bookkeeping only (the final EMA epoch can never be
#      pruned). The run dir is exp/${TRACK}_epoch8_bs8_<ts> (bs8 = 8 pairs).
#
#   cd /root/repos/rtc-sdd/Ghostbusters_team
#   bash scripts/run_B3a.sh                                        # fresh run
#   bash scripts/run_B3a.sh --resume exp/B3a_qwen17_..._<ts>       # after a container restart
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_B3a.sh                 # pick the GPU
#
# Extra arguments are appended to the b3wave.train command line. Training runs under nohup;
# its stdout/stderr go to SLS_setup/<track>_<timestamp>.log (one file per launch), and the
# per-epoch summary is also in SLS_setup/exp/<run>/logs/train.log.
#
# Evaluation of this run MUST pass the same flags (--loudness_norm --loudness_norm_lufs -23) to
# scripts/local_eval.py and main_eval.py; scripts/eval_checkpoints.py and the post-train-eval skill
# read them from config.yaml. Checkpoints are plain ModelSLS state_dicts (the consistency loss is
# training-only), so the usual eval scripts work unchanged.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
TRACK="B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234"
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

nohup "$PY" -u -m b3wave.train \
    --train_data_path "$RTCFAKE/wav/train" \
    --train_protocol  "$RTCFAKE/train_label.txt" \
    --dev_data_path   "$RTCFAKE/wav/dev" \
    --dev_protocol    ../data/dev_noisy_sim/dev_label_excl_halfB.txt \
    --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 \
    --batch_size 8 --num_epochs 8 --num_workers 64 \
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
    --consistency js --consistency_lambda 0.3 --consistency_pairs both \
    "$@" > "$LOG" 2>&1 &

PID=$!
echo "Started $TRACK (pid $PID)"
echo "Log:     $REPO/SLS_setup/$LOG"
echo "Follow:  tail -f $REPO/SLS_setup/$LOG"
echo "Stop:    kill $PID"
