# Three-day EXR HDR quality and cable-continuity campaign

Status: complete. Selected checkpoint: corrected ARM, step98722.

## What was tested

The frozen reference is the native-EXR knee/EAG checkpoint at step75940. Selection remains maximum
PQ `eval_all_psnr`, with LPIPS as tie-breaker inside0.07dB; SSIM is reported. A new secondary gate
measures tolerant edge recall and long missing skeleton runs in five fixed cable/thin-structure
ROIs. It is diagnostic only and cannot override a large regression in the three authoritative
metrics. Every selected candidate must also pass fixed-exposure visual inspection.

The first experiment changes only the inference ray sampler on identical weights. It tests whether
visible cable gaps are integration artifacts or already present in the learned radiance field.

The campaign order was revised after the cable review. The three-day budget is spent in gates:

1. isolate adaptive-ray-marching effects with frozen-weight renders and an equal-length legacy
   allocator control;
2. screen corrected allocation, a denser traversal, and the sample cap during training;
3. tune only losses or batch/LR settings that pass the first boundary;
4. train edge-aware map variants only if the rendering controls leave a residual structural gap;
5. continue the winner, evaluate all three views, and inspect fixed-exposure cable crops.

This ordering treats the frequency map as one input to adaptive sampling, not as the assumed cause
of every thin-structure defect.

## Results

### Frozen-checkpoint ray-sampling ablation

| Renderer | PQ PSNR | PQ SSIM | PQ LPIPS | Edge recall* | Long-gap fraction* | Decision |
|---|---:|---:|---:|---:|---:|---|
| Baseline adaptive | 33.8176 | 0.89838 | 0.22184 | 0.8975 | 0.0879 | Reference |
| ARM max-of-3 frequency query | 33.8197 | 0.89856 | 0.22250 | 0.8958 | 0.0901 | No gain |
| ARM corrected allocator | 33.8237 | 0.89855 | 0.22208 | 0.8985 | 0.0869 | Tiny positive control |
| ARM coarse step /2, cap2048 | 33.8465 | 0.89937 | 0.22432 | 0.8994 | 0.0866 | Small distortion gain, LPIPS regression |
| ARM dense + corrected | 33.8473 | 0.89937 | 0.22432 | 0.8995 | 0.0864 | Small distortion gain, LPIPS regression |
| ARM uniform fallback64 | 27.1144 | 0.78887 | 0.35755 | 0.7932 | 0.1831 | Catastrophic reject |
| Fixed1024 | 32.3761 | 0.88656 | **0.21108** | 0.8852 | 0.1029 | PSNR/continuity reject |
| Fixed2048 | 32.7299 | 0.89586 | 0.21956 | 0.8639 | 0.1251 | PSNR/continuity reject |

`*` The first ablation used the two original broad ROIs. The frozen baseline was subsequently
audited with five cable-focused ROIs: edge recall `0.90758`, long-gap fraction `0.07790`. All future
training candidates use this expanded fixed set.

### Map candidates prepared for downstream screening

| Candidate | Changed cells vs knee | Mean level | Nonempty bins |
|---|---:|---:|---:|
| knee+1 | 100.0% | 9.54 | 15 |
| scene-q75 (`L13`) edge floor | 35.0% | 10.59 | 15 |
| knee/calibrated union on structural cells | 31.3% | 10.29 | 16 |
| global knee/calibrated union | 44.3% | 11.19 | 16 |

These maps have not yet passed downstream training and are not leaders.

### Corrected-ARM continuation

The accepted step75940 weights were resumed with the same optimizer, scheduler, knee maps,
EAG-PQ-DSSIM loss and FR/FAS settings. The only initial change was the corrected ARM interval
allocator. Two additional boundaries improved monotonically:

| Step | PQ PSNR | PQ SSIM | PQ LPIPS |
|---:|---:|---:|---:|
| 83534 | 33.8761 | 0.89883 | 0.21934 |
| 91128 | **34.0280** | **0.89934** | **0.21668** |

The expanded cable gate improved from baseline edge recall `0.90758` / long-gap fraction `0.07790`
to `0.91341 / 0.07280`. The legacy ROI detector marked one dark patch under the eval1 chair as a
serious component. Direct paired-crop inspection shows continuous fingers, trousers and chair legs;
the candidate removes strong GT sensor/noise grain in the dark opening, so this is a detector false
positive rather than a geometry break. The checkpoint is the current intermediate leader.

### Equal-length legacy allocator control

The legacy continuation used the same step75940 weights, optimizer/scheduler state, seed, maps,
loss, ray batch, and two `7594`-step intervals. Only the corrected allocator was disabled.

| Allocator / step | PQ PSNR | PQ SSIM | PQ LPIPS | Edge recall | Long-gap fraction |
|---|---:|---:|---:|---:|---:|
| Legacy / 83534 | 33.8280 | 0.89842 | 0.21954 | — | — |
| Corrected / 83534 | **33.8761** | **0.89883** | **0.21934** | — | — |
| Legacy / 91128 | 34.0050 | 0.89907 | **0.21598** | 0.90978 | 0.07632 |
| Corrected / 91128 | **34.0280** | **0.89934** | 0.21668 | **0.91341** | **0.07280** |

The first boundary proves that the corrected allocation gain is not merely extra training. At the
second boundary the legacy branch catches up in PSNR and has the best LPIPS, but corrected ARM
retains higher PSNR/SSIM and reduces the cable long-gap fraction by another `4.6%` relative to the
equal-step legacy branch. These branches are within the `0.07 dB` perceptual tie window, so neither
is discarded until the denser-ARM screen and visual review are complete.

### Dense corrected-ARM control

Halving the coarse occupancy traversal step to `0.003125` and raising the cap from `1024` to `2048`
was trained for the same two intervals. It reached `33.8321 / 0.89937 / 0.21876` at step83534 and
`33.9837 / 0.89943 / 0.21565` at step91128. The final candidate is only `0.0443 dB` below the
maximum PSNR and therefore wins the aggregate tie-break on LPIPS; it also has the highest SSIM.
However, the five-ROI cable gate regresses to edge recall `0.90774` and long-gap fraction `0.07898`
(worse than the original `0.07790`). The tangled-cable ROI alone reaches `0.09943`. Dense ARM is
therefore rejected for the cable-repair objective despite its attractive aggregate perceptual
metrics.

### Continued corrected leader and paired map control

Continuing the ordinary knee/corrected branch from step91128 to98722 produced a new leader:
`34.0497 / 0.89927 / 0.21336`. Cable edge recall rose to `0.91428` and long-gap fraction fell to
`0.07238`. Relative to the original selected checkpoint this is `+0.2320 dB` PSNR, `-0.00848`
LPIPS, and a `7.1%` relative reduction in long-gap fraction.

The structural edge-floor map was then resumed from the identical step91128 checkpoint for the
identical `7594` updates. It reached `33.9377 / 0.89851 / 0.21242`: LPIPS is slightly lower, but the
`0.112 dB` PSNR deficit lies outside the tie window. It also regresses cable edge recall to
`0.90733` and long-gap fraction to `0.07997`. The edge-floor map is rejected. Threshold-free knee
remains the map policy; raising map levels on structural cells does not repair the cable defect.

### Final stopping-point and visual gate

One more full boundary reached `33.9764 / 0.89928 / 0.21204` at step106316. Its `0.0733 dB` PSNR
deficit is outside the frozen tie window and cable long-gap fraction regressed to `0.07983`. A
midpoint check at step102519 likewise failed (`33.9664 / 0.89915 / 0.21199`, long-gap `0.07568`).
Step98722 is therefore the measured stopping optimum.

All three paired native-EXR renders were independently measured and their `-2/0/+2 EV` sheets were
visually inspected. Cables, cable loops, stand poles, chair legs, floor tape and people remain
connected; no new floaters, clipping, desaturation or over-peak/non-finite output was observed.
The five cable sheets were also inspected at native crop scale. The legacy component detector still
responds to denoising of dark GT grain in one under-chair patch; paired inspection confirms this is
not a structural break.

Final aggregate and per-view PQ metrics:

| View | PQ PSNR | PQ SSIM | PQ LPIPS |
|---:|---:|---:|---:|
| eval0 | 33.8643 | 0.89958 | 0.25137 |
| eval1 | 34.6727 | 0.90599 | 0.22025 |
| eval2 | 33.6120 | 0.89222 | 0.16846 |
| **Mean** | **34.0497** | **0.89926** | **0.21336** |

Final artifacts:

- Checkpoint: `/mnt/data/lookcloser_exr_quality_campaign/runs/hdr_quality_3day_v1/lookcloser/cont_corrected_part2_s42/nerfstudio_models/step-000098722.ckpt`
  (`sha256:95546eca33c3142c3e4e9a940cfcdb5fc916e64d235dd6d885ee7b44a107761a`).
- Paired native EXR renders: `/mnt/data/lookcloser_exr_quality_campaign/runs/hdr_quality_3day_v1/lookcloser/cont_corrected_part2_s42/renders_best_step-000098722/`.
- Fixed-exposure review: `/mnt/data/lookcloser_exr_quality_campaign/runs/hdr_quality_3day_v1/lookcloser/cont_corrected_part2_s42/hdr_review_renders_best_step-000098722/`.
- Cable review and metrics: `/mnt/data/lookcloser_exr_quality_campaign/runs/hdr_quality_3day_v1/lookcloser/cont_corrected_part2_s42/edge_continuity_best_step-000098722/`.

## Insights

Changing only inference sampling does not restore cable continuity. Dense ARM improves global
distortion metrics slightly, while fixed integration trades a large PSNR loss for lower LPIPS and
also worsens the edge gate. Therefore a one-line renderer switch is not the solution. ARM can still
be causal during optimization, so continuation screens use the corrected allocator and compare an
optional PQ edge-consistency term before spending budget on map changes.

The corrected-ARM continuation improves learned geometry, and the effect at the first boundary is
substantially larger than changing the frozen-checkpoint renderer. The equal-length legacy control
confirms that the allocator is causal: corrected ARM wins PSNR and SSIM at both boundaries and the
cable gate at the final boundary. Legacy ARM converges to slightly better LPIPS, so the final choice
still depends on the prescribed perceptual tie rule and visual inspection.

An EAG finite-difference edge term at weight `0.1` reached `34.0001 / 0.89932 / 0.21628` at the same
step. Its LPIPS is lower by `0.00040`, but PSNR is lower by `0.0279 dB`; cable recall is essentially
unchanged (`0.91311`) and long-gap fraction only changes from `0.07280` to `0.07215`. The visual
cable sheets are also indistinguishable at review scale. It is therefore not a meaningful cable
repair and is retained only as a perceptual tie candidate. Reducing feature re-weighting strength
to `0.3` was rejected at the first boundary (`33.8296 / 0.89894 / 0.21991`).

The final conclusion is that cable continuity is governed mainly by the learned trajectory under
adaptive ray allocation. Merely changing the frozen renderer, doubling sampling density, adding a
local edge loss, or raising structural frequency-map cells does not fix it. Correct budget-aware
allocation plus the measured stopping point improves both authoritative metrics and cable
continuity while preserving the scene-adaptive threshold-free knee policy for new EXR scenes.
