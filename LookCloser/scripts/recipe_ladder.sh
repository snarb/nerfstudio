#!/usr/bin/env bash
# LookCloser recipe port -> 4D winner. Ablation ladder at 40k iters, sequential.
# Base = winner (H2, static l21/r4096, 4D l21/r4096, occ warmup/binary 4096, eval-12th).
set -u
REPO=/home/ubuntu/repos/nerfstudio
DATA=/opt/dlami/nvme/temporal_ngp_ds_eval12
OUT=/opt/dlami/nvme/temporal_runs
SCR=/tmp/claude-1004/-home-ubuntu-repos-nerfstudio/73b1ed7a-6406-4798-b25a-37355fbef616/scratchpad
SUMMARY="$SCR/recipe_ladder_summary.txt"
COMMON=(--data "$DATA" --output-dir "$OUT" --seed 42 --method instant-ngp-time
  --hypothesis H2 --static-log2-hashmap-size 21 --static-max-res 4096
  --log2-hashmap-size 21 --max-res 4096 --occ-warmup-steps 4096 --occ-binary-warmup-steps 4096
  --max-num-iterations 40000 --step-interval 8000 --no-update-summary)

echo "=== recipe ladder started $(date -u) ===" > "$SUMMARY"
run_one () {
  local name="$1"; shift
  echo ">>> $name START $(date -u)" | tee -a "$SUMMARY"
  python "$REPO"/LookCloser/scripts/run_temporal_ngp_quiet.py \
    --experiment-name "$name" "${COMMON[@]}" "$@" > "$SCR/${name}.runner.log" 2>&1
  echo ">>> $name DONE $(date -u) exit=$?" | tee -a "$SUMMARY"
  grep -iE "final checkpoint|psnr=|artifacts significant|PSNR=" "$SCR/${name}.runner.log" | tail -8 \
    | sed "s/^/    [$name] /" | tee -a "$SUMMARY"
}

# 1) matched control: winner config, MSE, no distortion, 40k
run_one RECIPE_baseline40k

# 2) loss port: Charbonnier + distortion 0.01 (expected biggest LPIPS win)
run_one RECIPE_charbdist \
  --mm reconstruction-loss-type=charbonnier --mm distortion-loss-mult=0.01

# 3) + spatial capacity toward the static leader: static 3D log2=23/max_res=8192 (4D kept l21/r4096)
run_one RECIPE_charbdist_cap \
  --mm reconstruction-loss-type=charbonnier --mm distortion-loss-mult=0.01 \
  --static-log2-hashmap-size 23 --static-max-res 8192

# 4) + ARM (const-freq fallback; temporal has no frequency grid -> low expectation, but test)
run_one RECIPE_charbdist_cap_arm \
  --mm reconstruction-loss-type=charbonnier --mm distortion-loss-mult=0.01 \
  --static-log2-hashmap-size 23 --static-max-res 8192 \
  --enable-arm --adaptive-coarse-step-size 0.0125 --max-steps-per-ray 1024 \
  --transmittance-threshold 0.0 --adaptive-interval-level-mode midpoint \
  --mm adaptive-max-frequency-level=12.0

echo "=== recipe ladder finished $(date -u) ===" | tee -a "$SUMMARY"
