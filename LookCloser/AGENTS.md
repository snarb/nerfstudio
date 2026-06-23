## General Instructions

Your goal is to help implement **"LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene"**.

We use **nerfstudio** (parent dir ./../) as the base codebase and extend it with the LookCloser model.

A rough, unverified baseline implementation is already committed. Treat it critically. Do not assume the current code is correct.

Use the local paper document as the primary reference:

`Paper LookCloser.md`

It contains the full paper text, figures, and links. Always use this local file instead of downloading the paper from the internet.

The main source of truth is measured experimental behavior: metrics, rendered outputs, and visual inspection. The paper is the starting point, but some implementation details or hyperparameters may be incomplete, unclear, or suboptimal for our dataset.

Start from the paper’s design and hyperparameters, then iteratively:
- verify the implementation against the paper;
- fix bugs and inconsistencies;
- improve the code and adapt hyperparameters based on measured results.

At the initial stage, final validation stage, and when debugging unclear problems, render and visually inspect outputs. Save and compare problematic patches visually. Use small/low-resolution crops or thumbnails when possible to avoid wasting tokens.

## Data for preprocessing debugging 

High-res 6K image: /home/ubuntu/repos/look-closer/E004_D014_graded.png
Fast preview: /home/ubuntu/repos/look-closer/E004_D014_HD.jpg

## Training data

For the HD multicamera bounded Instant-NGP baseline, use the processed nerfstudio dataset with the 3 eval views already separated by filename:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`

This dataset should parse as 66 train images and 3 eval images when using `nerfstudio-data --eval-mode filename`.

This is bounded indoor scene.

## Evaluation 

Use PSNR, SSIM, and LPIPS metrics for the evaluation reports. Do not include loss in reports.

After training, save the rendered images from the best checkpoint selected by `eval_all_psnr` (highest PSNR across all eval images; LPIPS as tie-breaker within `0.07 dB`).

### Current bounded Instant-NGP baseline data

Rendered outputs and metrics from the baseline run are documented in:

`./experiments/baseline_bounded_ngp.md`

### Quiet bounded Instant-NGP runner

Use this wrapper for baseline reruns to keep chat/context small:

```bash
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
python /home/ubuntu/repos/nerfstudio/LookCloser/scripts/run_bounded_ngp_quiet.py
```

The runner:

- launches `ns-train` with stdout/stderr redirected to `train_stdout.log`;
- prints only compact status lines (`step=...`, eval loss, PSNR, SSIM);
- monitors `metrics_compact.csv`;
- stops training when eval loss does not improve at an eval boundary;
- runs `ns-eval` on the latest checkpoint and writes renders;
- redirects verbose eval output to `eval_stdout.log`.

Default runner parameters:

- method: `instant-ngp-bounded`
- dataparser: `nerfstudio-data`
- data: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`
- output dir: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs`
- experiment name: `007740_hd_aabb4_multicamera_eval3_ns_focus_scene15`
- eval mode: `filename`
- scene scale: `1.5`
- center method: `focus`
- orientation method: `up`
- auto scale poses: `True`
- downscale factor: `1`
- train rays per batch: `8192`
- eval/save interval: `15188`
- max iterations: `60752`
- compact CSV logger enabled
- local terminal writer disabled

Use `--dry-run` to print the full command without training:

```bash
python /home/ubuntu/repos/nerfstudio/LookCloser/scripts/run_bounded_ngp_quiet.py --dry-run
```

Use `--no-stop-on-no-improve` only when explicitly asked to train all configured iterations regardless of eval loss.

Eval loss is used internally for early stopping.

### Context hygiene for long runs

Do not run long `ns-train` or `ns-eval` jobs in a TTY unless the user needs the full live output. The rich progress bars and config dump waste a lot of context.

Prefer the quiet runner above. If running commands manually, redirect noisy output to files:

```bash
ns-train ... > "$RUN_DIR/train_stdout.log" 2>&1
ns-eval ... > "$RUN_DIR/eval_stdout.log" 2>&1
```

## Architecture

The architecture documentation is stored in ./architecture.md. Update it after each major change, but keep cosine.

## Experiments
Save experiment results in the ./experiments directory. Use one file per topic. 

Experiment Markdown File Structure:

- What was tested: The hypothesis or configuration.

- Results: Tables and rendered crops/images with links.

- Insights: What was learned and next steps.

Keep cosine.

## Environment

Use the following Conda environment:
```bash
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
```
