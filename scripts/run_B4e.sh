#!/usr/bin/env bash
# B4e = B4a with ONE factor changed: + --aug_tel_p 0.25 --aug_tel_codecs g723_1 amr_nb (telephony codecs on 25% of
#   the codec calls instead of the Opus app profile; b4wave lever tel_codec, SLS_setup/b4wave/README.md)
#  B4a (B2f + --aug_apm_p 0.3, seed 1234) is the best run so far: top-5 merge half_B WF1 92.55, final epoch 8
#    93.15 (registry rows B4a-apmns-screen / -ep8). B4e keeps every B4a flag and the seed, and routes 25% of the
#    codec calls through a telephony codec, G.723.1 (6.3 kbps) or AMR-NB (4.75-12.2 kbps), 8 kHz narrowband,
#    round-tripped in memory with PyAV (port of David's utils/rtc_augment_expert.py::telephony_augm,
#    origin/feature/model-updates). The parent's Opus draws are made either way and the lever draws from its own
#    stream, so the acoustic stages and the WebRTC NS draws are B4a's.
#  Codecs: G.711 / G.722 / G.726 from the same table are reserved for the local proxy sets (sim_unseen_v1 /
#    sim_heldout_v1, b3wave/heldout_reserve.py) and refused at start-up; only G.723.1 and AMR-NB are trained on.
#  Hypothesis: B4a never sees narrowband CELP / ADPCM telephony audio, so it reads codec smoothing and the 4 kHz
#    band edge on bona fide speech as synthesis. Training both classes through G.723.1 / AMR-NB on 25% of the codec
#    calls makes those traces class-neutral, and that transfers to the unseen G.722 / G.726 / GSM implementations
#    in sim_heldout_v1.
#  Expected signature (vs B4a, same readout): sim_heldout_v1 recall_bona up, driving heldout F1 up; sim_matched_v1
#    F1 down by at most 0.5 (Opus apps, now 25% fewer Opus clips); dev_online_clean F1 within 0.5; spoof recall
#    within 0.5 on every set; probe sets steady. Caveat: sim_heldout_v1 also has the bandlimit / resample8k stages,
#    so part of a heldout gain can be narrowband familiarity rather than codec transfer; Codabench decides.
#  Decision rule (pre-registered, top-5 merge half_B WF1 vs B4a merge 92.55, paired bootstrap 2000):
#    positive = delta > +0.5 with CI low > 0 and clean F1 drop <= 0.5; within +-0.5 = null; below -0.5 = negative.
#    Falsifier: heldout F1 delta < +0.5 on both readouts while matched F1 drops > 0.5 -> drop the lever.
#
#   bash scripts/run_B4e.sh                                   # fresh run
#   bash scripts/run_B4e.sh --resume exp/<run dir name>       # after a container restart
#   EXPLOOP_PRINT_ARGV=1 bash scripts/run_B4e.sh              # print the trainer argv, launch nothing
#
# Same shape as scripts/run_B4a.sh. Extra arguments are appended to the trainer command line (later flags win).
# Training runs under nohup; stdout/stderr go to <workdir>/<track>_<timestamp>.log and the per-epoch summary to
# <run dir>/logs/train.log.
set -euo pipefail

PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
TRACK="B4e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_tel25_s1234"
WORKDIR="/root/repos/rtc-sdd/Ghostbusters_team/SLS_setup"

ARGS=(
    --train_data_path /mnt/paml-research/RTCFake/data/wav/train
    --train_protocol /mnt/paml-research/RTCFake/data/train_label.txt
    --dev_data_path /mnt/paml-research/RTCFake/data/wav/dev
    --dev_protocol ../data/dev_noisy_sim/dev_label_excl_halfB.txt
    --ssl_name Qwen/Qwen3-ASR-1.7B
    --n_layers 24
    --batch_size 16
    --num_epochs 8
    --num_workers 64
    --use_rtc_aug
    --aug_noise_dirs ../data/augmentation_train/rnnoise ../data/augmentation_train/esc50
    --aug_rir_dirs ../data/augmentation_train/rirs
    --musan
    --musan_dir ../data/augmentation/musan
    --loudness_norm
    --loudness_norm_lufs -23
    --optim adamw
    --lr 2.5e-6
    --head_lr_mult 10
    --weight_decay 0.01
    --lr_scheduler warmup_cosine
    --warmup_frac 0.05
    --lr_min 5e-7
    --grad_clip 1.0
    --ema_decay 0.9995
    --local_eval_sets ../configs/local_eval_sets.yaml
    --earlystop_metric wf1
    --earlystop_epoch 99
    --keep_topk_by_wf1 7
    --keep_epochs 8
    --seed 1234
    --track "$TRACK"
    --aug_apm_p 0.3
    --aug_tel_p 0.25
    --aug_tel_codecs g723_1 amr_nb
)

if [ "${EXPLOOP_PRINT_ARGV:-0}" = "1" ]; then
    printf '%s\n' "${ARGS[@]}"
    exit 0
fi

if [ ! -x "$PY" ]; then
    echo "python not found: $PY (set PY=... to override)" >&2
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found on PATH: the RTC codec augmentation would silently no-op. Reinstall it first (apt-get install -y ffmpeg)." >&2
    exit 1
fi
if ! "$PY" -c "import pyloudnorm" >/dev/null 2>&1; then
    echo "pyloudnorm not importable by $PY: run '$PY -m pip install pyloudnorm==0.2.0' first." >&2
    exit 1
fi
if ! "$PY" -c "import pywebrtc_audio" >/dev/null 2>&1; then
    echo "pywebrtc_audio not importable by $PY: the apm_ns lever (--aug_apm_p) needs it." >&2
    exit 1
fi
if ! "$PY" -c "import av; av.codec.Codec('g723_1', 'w'); av.codec.Codec('libopencore_amrnb', 'w')" >/dev/null 2>&1; then
    echo "PyAV with the G.723.1 and AMR-NB encoders not importable by $PY: run '$PY -m pip install av==15.1.0' first" >&2
    echo "(the av 19.x wheels have no libopencore_amrnb encoder)." >&2
    exit 1
fi

cd "$WORKDIR"
LOG="${TRACK}_$(date +%Y%m%d%H%M%S).log"

nohup "$PY" -u -m b4wave.train "${ARGS[@]}" "$@" > "$LOG" 2>&1 &

PID=$!
echo "Started $TRACK (pid $PID)"
echo "Log:     $WORKDIR/$LOG"
echo "Follow:  tail -f $WORKDIR/$LOG"
echo "Stop:    kill $PID"
