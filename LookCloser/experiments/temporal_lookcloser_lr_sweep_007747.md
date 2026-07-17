# Temporal LookCloser LR Sweep on 007747

## What was tested

Starting from the static 007740 leader checkpoint, train `007747` with the archived leader
configuration unchanged except for LR/scheduler. Five candidates were run in parallel and evaluated
on the 3 held-out eval views. All checkpoints were retained.

Leader sanity on local `007740`: PSNR `29.617964`, SSIM `0.668449`, LPIPS `0.231121`.

## Results

| Candidate | LR | Scheduler | PSNR | SSIM | LPIPS | Selected checkpoint | Renders |
|---|---:|---|---:|---:|---:|---|---|
| `const5e-4` | 0.0005 | constant | 28.882360 | 0.682507 | 0.337838 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_const5e-4_20260705_184726/nerfstudio_models/step-000151880.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_const5e-4_20260705_184726/renders_selected_step-000151880` |
| `exp1e-3` | 0.001 | exponential to 0.0001 | 28.803164 | 0.678778 | 0.346854 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_exp1e-3_20260705_184718/nerfstudio_models/step-000151880.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_exp1e-3_20260705_184718/renders_selected_step-000151880` |
| `exp5e-4` | 0.0005 | exponential to 0.00005 | 28.049322 | 0.670601 | 0.374229 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_exp5e-4_20260705_184720/nerfstudio_models/step-000151880.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_exp5e-4_20260705_184720/renders_selected_step-000151880` |
| `exp2e-4` | 0.0002 | exponential to 0.00002 | 26.963179 | 0.664628 | 0.389755 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_exp2e-4_20260705_184722/nerfstudio_models/step-000151880.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_exp2e-4_20260705_184722/renders_selected_step-000151880` |
| `exp1e-4` | 0.0001 | exponential to 0.00001 | 25.986790 | 0.655206 | 0.389672 | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_exp1e-4_20260705_184724/nerfstudio_models/step-000151880.ckpt` | `/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_exp1e-4_20260705_184724/renders_selected_step-000151880` |

## Insights

`const5e-4` is the selected default for frame-to-frame temporal transfer. It was best on all tracked
metrics: highest PSNR, highest SSIM, and lowest LPIPS. The sequential chain therefore uses constant
LR `0.0005` with no scheduler unless explicitly overridden.
