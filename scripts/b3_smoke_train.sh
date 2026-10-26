#!/usr/bin/env bash
# Smoke test 11 of the B3 package: 1-epoch runs of b3wave.train on a 2 % stratified subset, one per
# lever, into SLS_setup/exp_smoke/ (never exp/). Prints PASS/FAIL per variant and a summary.
#   bash scripts/b3_smoke_train.sh            # all variants (~4 min each on a 4090)
#   bash scripts/b3_smoke_train.sh a c f      # a subset of the variant letters
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
RTCFAKE="/mnt/paml-research/RTCFake/data"
cd "$REPO/SLS_setup"
mkdir -p exp_smoke
head -400 ../data/dev_noisy_sim/dev_label_excl_halfB.txt > exp_smoke/dev_small.txt

COMMON=(--train_data_path "$RTCFAKE/wav/train" --train_protocol "$RTCFAKE/train_label.txt"
        --dev_data_path "$RTCFAKE/wav/dev" --dev_protocol exp_smoke/dev_small.txt
        --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 --batch_size 16 --num_epochs 1 --num_workers 8
        --use_rtc_aug --aug_noise_dirs ../data/augmentation_train/rnnoise ../data/augmentation_train/esc50
        --aug_rir_dirs ../data/augmentation_train/rirs --musan --musan_dir ../data/augmentation/musan
        --use_level_aug --optim adamw --lr 2.5e-6 --head_lr_mult 10 --weight_decay 0.01
        --lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 5e-7 --grad_clip 1.0 --ema_decay 0.9995
        --local_eval_sets ../configs/local_eval_sets.yaml --local_eval_clean_n 200
        --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 --seed 1234
        --out_path ./exp_smoke --train_subset_frac 0.02)

declare -A EXTRA=(
  [a]=""
  [c]="--sampler balanced --sampler_lang_balance --ce_weights 0.5 0.5"
  [d]="--aug_snr_db -5 25 --aug_snr_dist beta --aug_snr_beta_mode 8"
  [e]="--aug_max_stages 3"
  [f]="--consistency js_emb --batch_size 8"
  [g]="--lr 5e-6 --lr_min 1e-6 --llrd 0.85"
  [h]="--specaug"
  [i]="--loudness_norm --loudness_norm_lufs -23"
  [j]="--aug_codec_profiles random"
  [k]="--aug_music_dirs ../data/augmentation/musan/music"
)
declare -A EXPECT=(
  [a]="Sampler: shuffle|Consistency: none|LLRD: 1|SpecAugment: off|Loudness norm: off|Epoch 1 LocalWF1"
  [c]="Sampler: balanced bona_frac=0.5 lang_balance=True|Epoch 1 B3 sampler bona=|Epoch 1 LocalWF1"
  [d]="SNR: -5..25 dB dist=beta|snr_dist=beta|aug .*snr:|Epoch 1 LocalWF1"
  [e]="Chaining: max_stages=3|max_stages=3 \\(chained, distinct\\)|Epoch 1 LocalWF1"
  [f]="Consistency: js_emb|16 clips per step \\(8 pairs\\)|CE=.* JS=.* Emb=|Epoch 1 LocalWF1"
  [g]="LLRD: 0.85|LLRD groups: 53 groups|Epoch 1 LocalWF1"
  [h]="SpecAugment: time 2x<=40 frames, freq 2x<=16 bins|Epoch 1 LocalWF1"
  [i]="Loudness norm: loudness_norm target=-23 LUFS|Epoch 1 LocalWF1"
  [j]="Codec profiles: random|codec_profiles=random|Epoch 1 LocalWF1"
  [k]="reserved for the local proxy sets"
)
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(a b c d e f g h i j k)
pass=0; fail=0; summary=""
for v in "${VARIANTS[@]}"; do
  LOG="exp_smoke/smoke_${v}.log"
  if [ "$v" = "b" ]; then
    RUN=$(ls -dt exp_smoke/B3smoke_a_* 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then echo "FAIL b: no variant-a run to resume"; fail=$((fail+1)); continue; fi
    "$PY" -u -m b3wave.train "${COMMON[@]}" --track B3smoke_a --num_epochs 2 --resume "$RUN" > "$LOG" 2>&1
    want="Resumed |Epoch 2/2|Epoch 2 LocalWF1"
  else
    # shellcheck disable=SC2086
    "$PY" -u -m b3wave.train "${COMMON[@]}" --track "B3smoke_${v}" ${EXTRA[$v]} > "$LOG" 2>&1
    want="${EXPECT[$v]}"
  fi
  ok=1; missing=""
  IFS='|' read -ra PATS <<< "$want"
  for pat in "${PATS[@]}"; do grep -qE -- "$pat" "$LOG" || { ok=0; missing="$missing [$pat]"; }; done
  if [ "$v" != "k" ] && grep -qE 'Traceback|Error' "$LOG"; then ok=0; missing="$missing [traceback]"; fi
  if [ "$v" = "k" ] && grep -q 'Epoch 1/' "$LOG"; then ok=0; missing="$missing [trained despite reserved dir]"; fi
  if [ $ok -eq 1 ]; then echo "PASS $v"; pass=$((pass+1)); else echo "FAIL $v: missing$missing (see $LOG)"; fail=$((fail+1)); fi
  summary="$summary $v:$([ $ok -eq 1 ] && echo ok || echo FAIL)"
done
echo "smoke_train: $pass passed, $fail failed ->$summary"
[ $fail -eq 0 ]
