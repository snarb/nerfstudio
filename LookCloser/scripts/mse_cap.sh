#!/usr/bin/env bash
set -u
REPO=/home/ubuntu/repos/nerfstudio
DATA=/opt/dlami/nvme/temporal_ngp_ds_eval12; OUT=/opt/dlami/nvme/temporal_runs
SCR=/tmp/claude-1004/-home-ubuntu-repos-nerfstudio/73b1ed7a-6406-4798-b25a-37355fbef616/scratchpad
SUMMARY="$SCR/mse_cap_summary.txt"
COMMON=(--data "$DATA" --output-dir "$OUT" --seed 42 --method instant-ngp-time --hypothesis H2
  --occ-warmup-steps 4096 --occ-binary-warmup-steps 4096 --max-num-iterations 40000 --step-interval 8000 --no-update-summary)
echo "=== mse_cap started $(date -u) ===" > "$SUMMARY"
run_one () { local name="$1"; shift
  echo ">>> $name START $(date -u)" | tee -a "$SUMMARY"
  python "$REPO"/LookCloser/scripts/run_temporal_ngp_quiet.py --experiment-name "$name" "${COMMON[@]}" "$@" > "$SCR/${name}.runner.log" 2>&1
  echo ">>> $name DONE $(date -u) exit=$?" | tee -a "$SUMMARY"
  grep -iE "final checkpoint|psnr=|artifacts significant" "$SCR/${name}.runner.log" | tail -8 | sed "s/^/    [$name] /" | tee -a "$SUMMARY"; }
# static 3D capacity l23/r8192, 4D kept l21/r4096, MSE
run_one RECIPE_mse_cap --static-log2-hashmap-size 23 --static-max-res 8192 --log2-hashmap-size 21 --max-res 4096
# + bump 4D dynamic capacity (log2=22, res kept 4096 to avoid time over-resolution), MSE
run_one RECIPE_mse_cap4d --static-log2-hashmap-size 23 --static-max-res 8192 --log2-hashmap-size 22 --max-res 4096
echo "=== mse_cap finished $(date -u) ===" | tee -a "$SUMMARY"
