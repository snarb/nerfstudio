# Fresh single-seed LookCloser blur ablations

## What was tested

2026-09-29. In progress; no quality change promoted. Base `630d58bd`, candidate
archive `8389770b`, one seed (42), independently initialized experiments. Maximum
campaign budget: 24 GPU-hours. Main was not changed.

The density bundle is split into activation, inverse-AABB scale, precision and
clipping. SH remains an independent control. The diagnostic runner uses the
standard LookCloser pipeline and `Trainer.train_iteration`, including its normal
AMP/Adam/scheduler and callbacks. Evaluation preserves Python/NumPy/Torch RNG.
Initial field parameter digests and the first 32 sampled batches match across
the six initial density controls.

Synthetic diagnostic: one teacher train camera, two teacher validation cameras,
3000 updates, 2048 rays, fixed128, hash19, LR .01→.001, corrected SH. Valid-pixel
uniform sampling is shared by all controls; this differs from the historical
distillation loop and uses no teacher depth. Existing fitted frequency maps are
shared. Historical numbers are not used as a paired baseline.

Real benchmark: fixed calibrated RGB for DEC5 `000973`, 62 train / **3** eval,
1920×1080; the two additional held-out cameras use identity color profiles and
the same frozen exposure. No eval RGB fits a profile, geometry or frequency map.
The real room remains in training and full-frame evaluation. Native GT-only
detail rectangles are fixed before real training. Full-image frequency maps are
fitted on train images only, 1000 updates/level as in the local paper.

Fight control: `007740`, 66 train / 3 eval, inherited Stage-A model/optimizer
settings and 30376 updates. All quality claims must use fresh paired results.

Runtime: `/home/brans/repos/nerfstudio/.venv`, Torch 2.7.1+cu128, RTX PRO 6000.
The requested Ubuntu Conda is inaccessible. Large artifacts live under
`/home/brans/lookcloser_artifacts/blur_ablation_fresh`; source, requests and compact
evidence live in this repository. [Cleanup log](assets/blur_ablation_fresh/cleanup.jsonl)
records 96.45 GiB of intermediate, unlinked checkpoints owned by `brans`, with
run requests and retained best models. Other users' checkpoints were untouched.

## Results

The synthetic controls reproduce the gray training-view failure. Independent
FP32 and exponential activation controls do not remove it. Inverse-AABB scaling
of **softplus alone** restores color and detail. See the
[common-protocol metrics](assets/blur_ablation_fresh/single_metrics.json).
These are quantized stride-2 diagnostic renders, not native full-scene scores.

| Single-view control, step2000 | Train PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Softplus | 20.233 | .93720 | .22891 |
| Only FP32 density | 20.233 | .93722 | .22885 |
| Exponential + FP32, unscaled | 20.244 | .94077 | .22669 |
| Only inverse-AABB scale, FP16 softplus | **39.721** | **.98948** | **.00500** |
| Scaled softplus + FP32 | 39.668 | .98959 | .00503 |
| Scaled exponential + FP32 | 39.999 | .98942 | .00443 |
| Add exponential clipping | 40.035 | .98944 | .00438 |
| Scaled softplus, legacy SH | 39.360 | .98868 | .00534 |
| Unscaled softplus, AMP scale65536 | 20.233 | .93735 | .22871 |
| Unscaled softplus, no distortion | 20.234 | .93748 | .22837 |

Each panel is GT left / learned RGB right. Unknown teacher background is not
supervised or counted in this table; its arbitrary prediction remains visible.

![Unscaled softplus](assets/blur_ablation_fresh/s0_softplus_train2000.png)

![Only inverse-AABB scale](assets/blur_ablation_fresh/s6_softplus_scaled_fp16_train2000.png)

The [saved-field audit](assets/blur_ablation_fresh/saturation.json) evaluates
1024 fixed valid training rays. Both unscaled softplus and unscaled exponential
have opacity-normalized RGB exactly `[1,1,1]` on all sampled rays at the selected
step1000 checkpoint. They encode a grayscale image through opacity. With scaled
softplus the white-saturation fraction is zero and chroma is nonzero.

Eight numerical/GPU tests pass: legacy softplus identity; FP32 exponentiation
outside autocast; scaling of optical thickness and gradients; clipping;
independent SH addition-theorem reference; and agreement between field-query
and density/occupancy-query paths. The additional check covers FP16 overflow of scaled softplus in tiny world units. Scene-quality acceptance remains pending.

## Insights and next steps

The early synthetic failure is an RGB saturation failure, not evidence that
exponential density is necessary. A density gain prevents this failure in the
small actor AABB. Increased AMP scale and removal of distortion do not fix the
same control. Transfer to real full-scene training is still unverified.

The final selector uses mean full-frame eval PSNR across all held-out cameras,
with LPIPS breaking ties within .07 dB. Matched-step comparisons remain separate.
The primary detail score is the equal-weight mean over face/hair/lipstick
rectangles across the three eval cameras (nine rectangles); the three train
cameras are diagnostic. Acceptance requires ≥.5 dB detail improvement with supporting SSIM/LPIPS and
visible train/eval improvement; fight tolerance is .10 dB PSNR / .005 SSIM /
.01 LPIPS with no new visible defect. No between-seed robustness is claimed.

### Initial real-scene checks

The original full-scene model reproduces blurred train and eval views at 2000
and 4000 updates; facial structure starts to appear after switching from the
256-sample warmup to adaptive marching. The 6000-step frozen-weight
[rendering audit](assets/blur_ablation_fresh/frozen_rendering.json) changes only
integration, with stride-2 face patches. Corrected allocation changes train
PSNR by +.016 dB and eval PSNR by -.014 dB. Uniform4096 does not restore detail.
This rules out a render-only correction as the main fix at this checkpoint.

![GT / original / corrected allocation / fixed256 / fixed1024 / fixed4096](assets/blur_ablation_fresh/frozen_rendering_train.png)

The [real early color audit](assets/blur_ablation_fresh/real_saturation_early.json)
finds no saturated RGB channel among the sampled rays, unlike the synthetic
control. These are distinct failure conditions. The synthetic density result
must not be presented as an already proven explanation of the real failure.

The calibrated input cameras and color profiles predate this experiment; some
held-out views were examined during earlier development. This is a paired
regression benchmark, not an untouched generalization benchmark.

### Fresh fight baseline

| Updates | Eval PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| 15188 | 28.7756 | .65091 | .36464 |
| **30376, selected** | **29.4741** | **.67341** | **.29662** |

Native all-three-view [metrics](assets/blur_ablation_fresh/fight_baseline_metrics.json),
[input hashes](assets/blur_ablation_fresh/fight_input_hashes.json) and
[initialization receipt](assets/blur_ablation_fresh/fight_baseline_identity.json).
Visual review of full frames and fingers shows no lipstick-like collapse.
This is the fresh regression control, not the longer historical Stage-A→FR.3 leader.

![Fight baseline: GT / learned RGB](assets/blur_ablation_fresh/fight_baseline_fingers.png)
