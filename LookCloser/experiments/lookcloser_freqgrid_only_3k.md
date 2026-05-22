# LookCloser 3k Frequency Grid Only

## What was tested

First controlled LookCloser training run on the processed 3k `007740` split with only the Frequency Grid path enabled:

- `pipeline.enable_frequency_grid=True`
- `pipeline.model.enable_frequency_grid=True`
- `pipeline.model.enable_feature_reweighting=False`
- `pipeline.datamanager.pixel_sampler.enable_fas=False`
- `pipeline.model.enable_adaptive_ray_marching=False`
- `nerfstudio-data --scene-scale 2.5`
- `pipeline.model.max_res_base=2048`
- `--logging.csv-writer.enable True`

Training ran for 5000 iterations with the fixed 256-sample ray marcher and the AABB collider.

Important caveat: the processed dataset does not currently contain `lookcloser_frequencies`, so the pipeline warns that 2D frequency maps are missing. In this ablation the Frequency Grid module is enabled, but because Feature Re-weighting, FAS, and Adaptive RM are disabled, the learned frequency grid is not yet affecting sampling or feature weighting.

## Results

Checkpoint:

[step-000004999.ckpt](lookcloser_freqgrid_only/outputs/freqgrid_only_scene2p5_3k/lookcloser/run_5k_final/nerfstudio_models/step-000004999.ckpt)

Training CSV:

[metrics_compact.csv](lookcloser_freqgrid_only/outputs/freqgrid_only_scene2p5_3k/lookcloser/run_5k_final/metrics_compact.csv)

Explicit `ns-eval` output:

[freqgrid_only_scene2p5_5k.json](lookcloser_freqgrid_only/metrics/freqgrid_only_scene2p5_5k.json)

| Metric | Value |
| --- | ---: |
| Eval PSNR, 3 held-out images | 26.1194 |
| Eval PSNR std | 0.3142 |
| Eval SSIM, 3 held-out images | 0.5716 |
| Eval SSIM std | 0.0644 |
| Eval rays/sec | 540047.9 |
| Eval FPS | 0.1465 |
| Best eval-batch PSNR during training | 25.8909 at step 4500 |
| Last train PSNR in CSV | 27.0862 at step 4990 |

Per-image PSNR computed from the saved side-by-side eval renders:

| Eval image | PSNR |
| --- | ---: |
| `D004_A014` | 25.9051 |
| `E004_B014` | 25.9683 |
| `I004_D014` | 26.4780 |

Saved rendered outputs:

- [eval_img_0000.png](lookcloser_freqgrid_only/renders/freqgrid_only_scene2p5_5k/eval_img_0000.png)
- [eval_img_0001.png](lookcloser_freqgrid_only/renders/freqgrid_only_scene2p5_5k/eval_img_0001.png)
- [eval_img_0002.png](lookcloser_freqgrid_only/renders/freqgrid_only_scene2p5_5k/eval_img_0002.png)

Visual inspection of the final eval renders shows sane geometry and color. Fine floor cracks, cable edges, and small highlights are still smoothed, which is expected before enabling the paper's frequency-aware downstream modules.

## Insights

The training and evaluation path is now working for the minimal LookCloser ablation, and the held-out PSNR target is met with margin: all three eval images are above 25.9 dB.

`scene_scale=2.5` remains the current 3k default after the short AABB sweep and the 5000-step validation run. The next useful experiment is to generate or repair `lookcloser_frequencies`, then enable one downstream module at a time so the Frequency Grid can actually influence training.
