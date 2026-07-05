#!/usr/bin/env bash
# Shard the train images across N concurrent fast_freqmap.py workers on one GPU.
# Usage: run_parallel.sh <images_dir> <out_dir> <n_workers> [extra args to fast_freqmap]
set -euo pipefail
IMAGES_DIR="$1"; OUT="$2"; NW="$3"; shift 3
EXTRA="$*"
cd /home/brans/repos/nerfstudio
VENV=.venv/bin/python
mkdir -p "$OUT" /home/brans/freqmap_bench/logs

# Build sorted list of frame_train_*.jpg
mapfile -t ALL < <(ls "$IMAGES_DIR"/frame_train_*.jpg | sort)
N=${#ALL[@]}
echo "total=$N workers=$NW extra=$EXTRA"

# Write a per-worker file list, worker w takes indices w, w+NW, w+2NW ...
for ((w=0; w<NW; w++)); do
  L="$OUT/worklist_$w.txt"; : > "$L"
  for ((i=w; i<N; i+=NW)); do echo "${ALL[$i]}" >> "$L"; done
done

START=$(date +%s)
pids=()
for ((w=0; w<NW; w++)); do
  ( $VENV /home/brans/freqmap_bench/fast_freqmap.py \
      --file-list "$OUT/worklist_$w.txt" --out "$OUT" $EXTRA \
      > "/home/brans/freqmap_bench/logs/worker_$w.log" 2>&1 ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
END=$(date +%s)
DT=$((END-START))
echo "ALL_DONE workers=$NW total_imgs=$N wall=${DT}s per_img=$(python3 -c "print(f'{$DT/$N:.2f}')")s"
