#!/usr/bin/env bash
# Train the single-condition experts (SLS_setup/main_train_expert.py) one after another on ONE GPU.
#
#   bash Ghostbusters_team/scripts/run_all_experts.sh [extra main_train_expert.py args...]
#
# Every expert gets its own condition (p_noise 0.5) plus, shared by all of them, the RTC/telephony
# codecs (p_codec 0.5), the level augmentation (--use_level_aug: random level -38..-12 dBFS +
# AGC / compressor / limiter, utils/level_augment.py) and read-time loudness normalisation
# (--loudness_norm, -23 LUFS on train AND dev; pass the same flag to main_eval.py).
#
# Env overrides:
#   GPU=0                    the one GPU every expert trains on
#   EXPERTS="office ..."     subset / order of experts (default: all 9)
#   LEVEL_AUG=1              0 drops --use_level_aug
#   LOUDNESS_NORM=1          0 drops --loudness_norm
#   LOUDNESS_LUFS=-23        --loudness_norm_lufs
#   TAG=...                  run tag, prefixed to every exp/ dir name (default: built from the above)
#   NUM_WORKERS=8            DataLoader workers
#   DRY_RUN=1                print the commands, launch nothing
#
# Run dirs land in Ghostbusters_team/SLS_setup/exp/ as
#   <TAG>_expert_<noise>_epoch100_bs16_<timestamp>/
# each run's full stdout/stderr is shown live and also kept in exp/launch_<TAG>_<timestamp>/<noise>.log,
# and exp/launch_<TAG>_<timestamp>/summary.txt gets one line per finished expert.
# A failed expert is reported and the next one still starts.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # RTCSDD_challenge/
cd "$ROOT"

EXPERTS=(${EXPERTS:-office coffee echo rain footsteps keyboard reverb musan_noise musan_music})
GPU="${GPU:-0}"
LEVEL_AUG="${LEVEL_AUG:-1}"
LOUDNESS_NORM="${LOUDNESS_NORM:-1}"
LOUDNESS_LUFS="${LOUDNESS_LUFS:--23}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# p_noise / p_codec drawn independently, 13 codecs (RTC Opus + telephony), dev WF1 selection
DEFAULT_TAG="QWEN17_pn05_pc05_telcodecs"
LEVEL_ARGS=()
if [[ "$LEVEL_AUG" == 1 ]]; then
    LEVEL_ARGS+=(--use_level_aug)
    DEFAULT_TAG+="_lvl"
fi
if [[ "$LOUDNESS_NORM" == 1 ]]; then
    LEVEL_ARGS+=(--loudness_norm --loudness_norm_lufs "$LOUDNESS_LUFS")
    DEFAULT_TAG+="_ln${LOUDNESS_LUFS#-}"
fi
TAG="${TAG:-${DEFAULT_TAG}_wf1}"

DATA=data/data
AUG=$DATA/data/augmentation
SIM=$DATA/data/dev_noisy_sim
OUT=Ghostbusters_team/SLS_setup/exp
STAMP="$(date +%Y%m%d%H%M%S)"
LOG_DIR="$OUT/launch_${TAG}_${STAMP}"
SUMMARY="$LOG_DIR/summary.txt"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "[$(ts)] $*"; [[ -n "${DRY_RUN:-}" ]] || echo "[$(ts)] $*" >> "$SUMMARY"; }

[[ -n "${DRY_RUN:-}" ]] || mkdir -p "$LOG_DIR"
echo "GPU $GPU memory in use right now:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$GPU" | sed 's/^/  GPU /'
note "tag=$TAG  gpu=$GPU  experts(${#EXPERTS[@]})=${EXPERTS[*]}  level/loudness: ${LEVEL_ARGS[*]:-off}"

n_ok=0
n_fail=0
for i in "${!EXPERTS[@]}"; do
    noise="${EXPERTS[$i]}"
    cmd=(python Ghostbusters_team/SLS_setup/main_train_expert.py
        --train_data_path $DATA/wav/train
        --train_protocol $DATA/meta/train_label.txt
        # the first dev set is the clean one (WF1 = 0.3 clean + 0.7 mean noisy)
        --dev_data_path $DATA/wav/dev $SIM/matched_v1 $SIM/heldout_v1
        --dev_protocol $SIM/dev_label_excl_halfB.txt
                       $SIM/matched_v1/protocol_mini.txt $SIM/heldout_v1/protocol_mini.txt
        --expert_noise "$noise"
        --aug_noise_dirs $AUG/rnnoise $AUG/esc50
        --aug_rir_dirs $AUG/rirs
        --musan_dir $AUG/musan
        --aug_p_noise 0.5 --aug_codec_p 0.5
        "${LEVEL_ARGS[@]}"
        --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24
        --device "cuda:$GPU"
        --num_workers "$NUM_WORKERS"
        --out_path $OUT
        --track "$TAG"
        "$@")
    echo
    echo "=================================================================================="
    note "[$((i + 1))/${#EXPERTS[@]}] $noise: START on GPU $GPU -> $LOG_DIR/$noise.log"
    echo "=================================================================================="
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '  %q' "${cmd[@]}"; echo
        continue
    fi
    t0=$SECONDS
    "${cmd[@]}" 2>&1 | tee "$LOG_DIR/$noise.log"
    rc=${PIPESTATUS[0]}
    mins=$(( (SECONDS - t0) / 60 ))
    best="$(grep -E 'Early stopping at epoch|Best dev WF1' "$LOG_DIR/$noise.log" | tail -n 1)"
    run_dir="$(grep -oE 'Experiment saved to: .*' "$LOG_DIR/$noise.log" | tail -n 1 | sed 's/Experiment saved to: //')"
    if (( rc == 0 )); then
        n_ok=$((n_ok + 1))
        note "[$((i + 1))/${#EXPERTS[@]}] $noise: DONE in ${mins} min | ${best:-no early stop (ran all epochs)} | ${run_dir}"
    else
        n_fail=$((n_fail + 1))
        note "[$((i + 1))/${#EXPERTS[@]}] $noise: FAILED (exit $rc) after ${mins} min, see $LOG_DIR/$noise.log"
    fi
done

[[ -n "${DRY_RUN:-}" ]] && exit 0
echo
note "ALL DONE: ${n_ok} ok, ${n_fail} failed. Summary: $SUMMARY"
cat "$SUMMARY"
