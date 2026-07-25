# Fine-tuning task v2: finish every temporal frame

Work from clean `main` in `/home/brans/repos/nerfstudio`. Process the remaining
frames `007754..008048` (stride 7) in order. Frames `007740` and `007747` are
already accepted and must not be retrained or overwritten.

## Final artifact layout

Intermediate runs may live under `/mnt/data`, but after selecting a winner copy
only the final renderable snapshot into its own frame dataset:

```text
/home/brans/temporal_perframe_stride7_45f/<frame>/
  snapshot/
    config.yml
    lookcloser/final/nerfstudio_models/step-XXXXXXXXX.ckpt
    selection.json
    provenance.json
    validation.json
  render/
    eval_img_0000.png
```

The snapshot config must resolve its checkpoint inside that same `snapshot`
tree and its dataparser to that frame. Prove every promoted snapshot with a
fresh `ns-eval` loaded only from `snapshot/config.yml`; record hashes and copy
the selected fresh `eval_img_0000.png` to the sibling `render` directory.
Never copy intermediate checkpoints into the dataset.

The bootstrap snapshots already exist:

- `/home/brans/temporal_perframe_stride7_45f/007740/snapshot`;
- `/home/brans/temporal_perframe_stride7_45f/007747/snapshot`.

Start with target `007754`, loading parent
`/home/brans/temporal_perframe_stride7_45f/007747/snapshot`; after accepting
`007754`, use its snapshot as the only parent for `007761`, and continue
sequentially. Stop on a failed frame—never skip it or forward an unaccepted
checkpoint.

## Training

Generalize `LookCloser/scripts/run_lookcloser_temporal_finetune.py` to accept
target frame, parent snapshot and seed without changing its fixed recipe. For
each frame launch **three concurrent, not sequential**, independent runs with
seeds `42,43,44`, all from the same accepted parent. Use two concurrent runs
(`42,43`) only when a VRAM preflight or real OOM proves that three do not fit;
do not silently fall back to one.

Across frames load only field/model parameters with
`checkpoint_load_mode=model_parameters_only`. Adam, scheduler, scaler, RNG,
occupancy/frequency grids, FAS state and telemetry must be fresh at local step
0. Keep hash23, batch4096, mixed precision, standard
`lookcloser_frequencies`, FAS1.0, FR0.3, warmup4096, Adam
`lr=0.015, eps=1e-15, weight_decay=0`, and exponential decay to `0.0001` over
300000 updates. Full-evaluate and save every15188 steps; reproduce the runner's
process boundaries through step151880 and continue the best valid candidates
one interval at a time only as needed to confirm the two-interval plateau.

Before each launch freeze and verify that frame's exact JPEG and standard-map
file set/hashes. Never use hash24, FR1.0,
`lookcloser_frequencies_chroma422`, `*_probe`, cached rays, fused Adam, TCNN
JIT, CPU FAS prefetch or independent RNG streams.

## Selection and visual gate

For every evaluation boundary crop `eval_img_0000.png` at
`(left=700, top=100, right=1120, bottom=480)` and save a native-resolution
comparison with the previous accepted frame and the `007740` leader. Fingers
must remain separated and sharp; the chain must remain continuous, gap-free
and unblurred. Numeric success without an explicit visual pass is failure.

Reject visual failures first. Among valid candidates prioritize maximum PSNR;
among checkpoints within inclusive `0.07 dB` of that maximum choose minimum
LPIPS. Report SSIM but never use it to rescue bad PSNR/LPIPS.

Expected quality is approximately the current `007747` result or better:
PSNR `≈29.88+`, SSIM `≈0.676+`, LPIPS `≈0.215` or lower. Hard minimum gates are
PSNR `>=29.840143`, SSIM `>=0.669203`, LPIPS `<=0.219455`, plus the visual gate.

## Metrics

Maintain exactly one final-validation row per accepted frame in
`/home/brans/temporal_perframe_stride7_45f/metrics.csv`. Include frame, winning
seed, parent frame, selected step, PSNR, SSIM, LPIPS, visual verdict, snapshot
checkpoint path and SHA-256. The file already contains `007740` and `007747`;
append/atomically update it after each fresh snapshot validation. Do not report
evaluation loss.

Keep complete boundary metrics, three-view renders, crops, configs, hashes and
train/eval wall timings in the `/mnt/data` campaign. Check VRAM and projected
root-disk space before each frame, never delete source runs, and never let a
copied config retain a temporary run's checkpoint path.
