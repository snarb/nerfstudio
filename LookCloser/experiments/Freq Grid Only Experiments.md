# Freq Grid Only Experiments

## Current result

Config: Frequency Grid enabled; Feature Re-weighting, FAS, Adaptive RM disabled. 3k split, `scene_scale=2.5`, `max_res_base=2048`, 5000 steps, CSV logging enabled.

Metrics: `ns-eval` PSNR `26.1194`, SSIM `0.5716`. Per-render PSNR: `25.9051`, `25.9683`, `26.4780`.

Visual result: bad. Renders in `experiments/lookcloser_freqgrid_only/renders/freqgrid_only_scene2p5_5k` look over-smoothed and worse than expected Instant-NGP quality. Treat PSNR as insufficient here; visual quality and SSIM are the useful signals.

Reference: `instant-ngp-big` has PSNR `22.6693`, SSIM `0.6013`; SSIM and visual quality are better than current Freq Grid Only.

## Already checked

- Training/eval now runs end-to-end.
- Fixed marcher shape bug fixed.
- Distortion loss scalar bug fixed.
- Eval PSNR/SSIM metrics fixed.
- Eval random background made deterministic.
- AABB collider enabled for LookCloser.
- AABB sweep: `scene_scale=2.5` best among `1.0`, `1.5`, `2.0`, `2.5`, `3.0`, `5.0`.
- Hash-grid quick sweep: keep `max_res_base=2048`; `1024` is fast-debug fallback.
- Current processed data has no `lookcloser_frequencies`; Frequency Grid is enabled but not yet meaningfully driving downstream behavior.

## ToDo

- Compare against clean `instant-ngp` and `instant-ngp-big` renders side by side on the same three eval views.
- Check whether `ns-eval` PSNR is inflated by background/crop/side-by-side handling.
- Restore or regenerate `lookcloser_frequencies`; verify maps visually before training.
- Run true Frequency Grid update with valid maps.
- Try fixed sampler improvements common to NGP/LookCloser: samples per ray, background color, distortion loss weight, proposal settings if applicable.
- Use SSIM plus visual crops as early stop criteria, not PSNR alone.
- After Freq Grid Only is sane, enable Feature Re-weighting, then FAS, then Adaptive RM one at a time.

## Next command baseline

Use the same command family as `run_5k_final`: `--logging.csv-writer.enable True`, `nerfstudio-data --eval-mode filename --scene-scale 2.5`, Feature Re-weighting/FAS/Adaptive RM off.
