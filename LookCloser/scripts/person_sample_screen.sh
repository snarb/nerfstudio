#!/usr/bin/env bash
# Person-oversampling screen vs leader, all at 25k iters, identical winner config otherwise.
# Sequential (winner config ~35GB VRAM -> no parallel). Logs a compact summary per run.
set -u
REPO=/home/ubuntu/repos/nerfstudio
RUN=python\ "$REPO"/LookCloser/scripts/run_temporal_ngp_quiet.py
DATA=/opt/dlami/nvme/temporal_ngp_ds_eval12
OUT=/opt/dlami/nvme/temporal_runs
SUMMARY=/tmp/claude-1004/-home-ubuntu-repos-nerfstudio/73b1ed7a-6406-4798-b25a-37355fbef616/scratchpad/person_screen_summary.txt
COMMON=(--data "$DATA" --output-dir "$OUT" --seed 42
  --hypothesis H2 --static-log2-hashmap-size 21 --static-max-res 4096
  --log2-hashmap-size 21 --max-res 4096 --occ-warmup-steps 4096 --occ-binary-warmup-steps 4096
  --max-num-iterations 25000 --step-interval 5000)

echo "=== person-sample screen started $(date -u) ===" > "$SUMMARY"

run_one () {
  local name="$1"; shift
  echo ">>> $name START $(date -u)" | tee -a "$SUMMARY"
  python "$REPO"/LookCloser/scripts/run_temporal_ngp_quiet.py \
    --experiment-name "$name" --no-update-summary "${COMMON[@]}" "$@" \
    > "/tmp/claude-1004/-home-ubuntu-repos-nerfstudio/73b1ed7a-6406-4798-b25a-37355fbef616/scratchpad/${name}.runner.log" 2>&1
  echo ">>> $name DONE $(date -u) exit=$?" | tee -a "$SUMMARY"
  # pull final metrics + artifact + throughput
  local L="/tmp/claude-1004/-home-ubuntu-repos-nerfstudio/73b1ed7a-6406-4798-b25a-37355fbef616/scratchpad/${name}.runner.log"
  grep -iE "final checkpoint|psnr=|artifacts significant|PSNR=" "$L" | tail -6 | sed "s/^/    [$name] /" | tee -a "$SUMMARY"
  # rays/sec median from the train stdout log
  local TS
  TS=$(ls -dt "$OUT/$name"/*/*/train_stdout.log 2>/dev/null | head -1)
  if [ -n "${TS:-}" ]; then
    python "$REPO"/LookCloser/scripts/person_rayspeed.py "$TS" 2>/dev/null | sed "s/^/    [$name] /" | tee -a "$SUMMARY"
  fi
}

run_one PERSAMP_baseline_uniform --method instant-ngp-time
run_one PERSAMP_pf030 --method instant-ngp-time-personsample --ps person-frac=0.3
run_one PERSAMP_pf040 --method instant-ngp-time-personsample --ps person-frac=0.4
run_one PERSAMP_pf020 --method instant-ngp-time-personsample --ps person-frac=0.2

echo "=== person-sample screen finished $(date -u) ===" | tee -a "$SUMMARY"
