# Bounded Instant-NGP Experiment Plan

## Guardrails

- Do not tune optimizer/scheduler, camera optimizer, MLP architecture, hash-grid settings, or proposal sampling.
- Keep potentially behavior-changing additions opt-in:
  - `loss_type=mse`
  - `appearance_eval_mode=zero`
  - `distortion_loss_mult=0.0`
  - `use_gradient_scaling=False`
- For scene-scale-only experiments, vary only `scene_scale`; all other train/model/dataparser knobs stay at the anchor defaults.
- Distortion, Huber loss, appearance mean/off, background, and sampling/render knobs are later-stage ablations only.

## Anchor

Documented baseline:

- `scene_scale=1.5`
- `center_method=focus`
- `orientation_method=up`
- `auto_scale_poses=True`
- `train_num_rays_per_batch=8192`
- `background_color=black`
- `near_plane=0.01`
- `alpha_thre=0.0`
- `cone_angle=0.0`
- `render_step_size=null`
- `loss_type=mse`
- `appearance_eval_mode=zero`
- `distortion_loss_mult=0.0`
- no scene contraction

Documented final eval:

| Checkpoint protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Full run, latest checkpoint `step-000060751.ckpt` | 24.417955 | 0.639772 | 0.460250 |

Important: this baseline was evaluated on the latest checkpoint after the full training budget. It was not selected by lowest in-training eval loss.

## Immediate Reproduction Check

Run three `scene_scale=1.5` controls with eval-loss early stop to measure variance and detect any accidental training-side change.

Protocol for each run:

1. Use `run_bounded_ngp_quiet.py`.
2. Pass `--scene-scale 1.5`.
3. Keep the wrapper default early stop enabled. Training stops when the current eval loss is not lower than the previous eval loss.
4. Use `--max-num-iterations 100000` only as a hard ceiling in case eval loss keeps improving.
5. Let the wrapper run final `ns-eval` on the selected best-eval-loss checkpoint.
6. Record final PSNR/SSIM/LPIPS and the wrapper's in-training eval rows from `metrics_compact.csv` for diagnosis.

Commands:

```bash
python LookCloser/scripts/run_bounded_ngp_quiet.py \
  --timestamp repro_scene150_earlystop_A \
  --scene-scale 1.5 \
  --max-num-iterations 100000 \
  --no-update-summary
```

Repeat with suffixes `B` and `C`.

Note: interrupted runs `repro_scene150_latest_A_20260528_074117` and `repro_scene150_earlystop_B_20260528_081354` should be ignored. Clean run A used the original `70000` ceiling; runs B and C should use `100000`.

Acceptance:

- If the three early-stop evals cluster near the expected quality range, continue the staged sweep using the same early-stop plus best-eval-loss checkpoint protocol.
- If they remain far below baseline, stop later sweep stages and inspect training-side changes/config/runtime variance before using those results.

## Reproduction Results

Before clean runs A/B/C, the model-side ablation additions were removed. `git diff -- nerfstudio/models/instant_ngp.py` was empty, so these runs used the original Instant-NGP model code path. The wrapper still exposes extra CLI flags, but none of the removed model flags were passed.

Protocol:

- `scene_scale=1.5`
- eval-loss early stop enabled
- `max_num_iterations=70000` for A, `100000` for B/C as a ceiling only
- final `ns-eval` on the selected best-eval-loss checkpoint

| Run | Timestamp | Selected checkpoint | Eval loss | PSNR | SSIM | LPIPS |
|---|---|---|---:|---:|---:|---:|
| A | `repro_scene150_clean_A_20260528_081832` | `step-000015188.ckpt` | 0.00438475 | 23.715738 | 0.637653 | 0.499827 |
| B | `repro_scene150_clean_B_20260528_083138` | `step-000045564.ckpt` | 0.00410401 | 23.816729 | 0.666949 | 0.470307 |
| C | `repro_scene150_clean_C_20260528_085748` | `step-000030376.ckpt` | 0.00432915 | 23.659456 | 0.647519 | 0.479323 |

Variance:

| Metric | Mean | Std dev | Min | Max |
|---|---:|---:|---:|---:|
| PSNR | 23.730641 | 0.079688 | 23.659456 | 23.816729 |
| SSIM | 0.650707 | 0.014906 | 0.637653 | 0.666949 |
| LPIPS | 0.483153 | 0.015128 | 0.470307 | 0.499827 |

Interpretation:

- The earlier low reproduction attempts were contaminated by temporary model-code changes and should be ignored.
- Clean reproductions cluster tightly in PSNR, but they still do not reproduce the documented baseline PSNR `24.417955`.
- SSIM is comparable or better than the documented baseline `0.639772` in two of three clean runs.
- Continue investigating training/runtime variance before treating later sweep-stage differences as reliable.

## Completed Scene-Scale Sweep

The first scene-scale sweep used early-stop plus best-eval-loss checkpoint selection. Those results are useful for comparing that specific protocol, but they are not directly comparable to the documented baseline latest-checkpoint metric.

Tentative result under that protocol:

| Scene scale | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|
| 1.65 | 24.127241 | 0.620946 | 0.474837 |

Do not carry this forward to later stages until the `scene_scale=1.5` reproduction check is resolved.
