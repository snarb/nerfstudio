# DEC5 000899 off-the-shelf geometry without Splatfacto training

## What was tested

The goal was to remove per-frame Splatfacto training from the pure
view-dependent texture renderer while preserving the face, lipstick, hand, and
held object. The room background is not a target. No image, person, or face
mask was used to predict depth, fuse TSDF, select source colour, or render.

The fixed target is held-out camera `nearest_eval_00000` from DEC5 frame
`000899`. Texture always comes from the same 16 evenly distributed train
cameras through hard `nearest-fill` selection; RGB values are never averaged.
The tested geometry sources are:

1. the existing Splatfacto-depth TSDF reference;
2. calibrated PatchMatch geometric depth from 62 train cameras, produced by the verified CUDA
   COLMAP 3.13.0.dev0 build at commit `5509fffe`;
3. pose-conditioned Depth Anything 3 Large 1.1 from 16 and 62 cameras;
4. pose-conditioned MapAnything from 16 cameras;
5. MonoMVSNet's released BlendedMVS checkpoint from 16 cameras, with both 5
   and 16 views supplied to each reference prediction;
6. MVSMamba's released BlendedMVS checkpoint from 16 cameras, with all 16
   views supplied to each reference prediction.

Every candidate depth set was fused into a continuous Open3D TSDF and
ray-traced back into the same source and target cameras. The main TSDF settings
were `voxel_length=0.001`, `sdf_trunc=0.006`, `depth_trunc=4`, and normalized
crop AABB `[-0.15, 0.15]^3`. The stricter COLMAP leader used the previously
validated finer `voxel_length=0.0005`, `sdf_trunc=0.004` pair.

### Fair metric protocol

Candidate-defined support is not used for the comparison: a broken mesh can
otherwise improve its apparent metric by omitting hard pixels. The optional
`--metric-surface-depth-manifest` argument makes every method score the exact
same Splatfacto-TSDF first-hit pixel set while leaving candidate geometry in
control of rendering and visibility. Missing candidate pixels therefore count
as black errors. This option defaults to `None`, so historical rendering is
unchanged.

Metrics are display-domain PSNR, SSIM, and LPIPS. They are restricted to the
fixed surface intersected with one of three rectangles: actor and held object
`[0, 150, 1400, 1030]`, face `[687, 392, 1187, 892]`, and lipstick/hand
`[687, 540, 987, 800]`. No full-frame or room metric is reported.

## Results

### Best configuration for each geometry source

| Geometry source | Geometry cameras | Candidate target coverage | Actor/object PSNR | SSIM | LPIPS | Face LPIPS | Lipstick/hand LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Splatfacto-depth TSDF reference | 62 | 42.25% | **22.9941** | 0.754713 | **0.119565** | 0.114183 | **0.083527** |
| COLMAP PatchMatch -> TSDF | 62 | 41.28% | 22.3303 | **0.786236** | 0.128180 | **0.112860** | 0.083786 |
| Depth Anything 3 Large 1.1 | 16 | 39.61% | 20.0474 | 0.645986 | 0.209388 | 0.189357 | 0.138876 |
| MapAnything | 16 | 54.87% | 19.0066 | 0.627047 | 0.262756 | 0.258410 | 0.236898 |
| MonoMVSNet, 16 views/reference | 16 | 32.90% | 15.1755 | 0.569488 | 0.409213 | 0.270940 | 0.252528 |
| MVSMamba, 16 views/reference | 16 | 22.17% | 12.6006 | 0.498699 | 0.517408 | 0.392817 | 0.352901 |

The common fixed metric surface occupies `42.19%` of the full image before ROI
intersection. MapAnything's larger candidate coverage is false extra geometry,
not more correct actor coverage. Visual inspection agrees with the fixed-support
metrics: it contains large folds and displaced surfaces around hair, hands, and
clothes.

### Camera-count controls

| Method | Change | Actor/object PSNR / SSIM / LPIPS | Result |
|---|---|---|---|
| DA3 Large 1.1 | 16 -> 62 joint input views | `20.0474 / 0.645986 / 0.209388` -> `15.6511 / 0.538273 / 0.430673` | Much worse; false strip across eye and missing hand |
| MonoMVSNet | 5 -> 16 source views per reference | `13.0638 / 0.525620 / 0.478203` -> `15.1755 / 0.569488 / 0.409213` | Better, but still unusably incomplete |

DA3's measured model forward was `1.14 s` for 16 images and `5.81 s` for 62
images at process width 1008 on this GPU. These timings exclude model loading,
depth serialization, TSDF fusion, and mesh ray tracing.

### Visual review

The comparison artifacts are copied to:

`/mnt/data/lookcloser_dec5_5a3_final/000899_offtheshelf_geometry_comparison`

The face review shows that COLMAP retains pores, eyelashes, hair, and the hard
lipstick boundary nearly as well as the Splatfacto-derived surface. Its
remaining errors are small silhouette holes near hair, neck, and hand. DA3
Large keeps local face texture sharp where its mesh is correct, but its actor
boundary is incomplete. MapAnything and MonoMVSNet show large displaced or
missing regions.

### PatchMatch ear-artifact ablation

The initial COLMAP result above used half-resolution (`960`) PatchMatch. A planar hole fill and
different RGB source selectors did not move the detached patch under the woman's left ear, which
localized the defect to geometry rather than texture aggregation. Full-resolution (`1920`)
two-pass PatchMatch repaired the ear/hair surface. A scale-aware connected-component filter then
removed the remaining isolated island: in the clean end-to-end run it had 165 triangles, versus
157,773 triangles in the actor component. The selected generic threshold is the larger of 100
triangles and `0.2%` of the largest component (316 triangles here); it contains no image-space
coordinate or ear-specific rule.

All rows below use the same held-out image, the same independent Splatfacto-derived metric surface,
and the same actor/face/left-ear/lipstick rectangles. They are display-domain metrics and exclude
the room. The original and selected rows were recomputed together rather than copied from runs
with slightly different face boxes.

| PatchMatch / texture configuration | Actor/object PSNR / SSIM / LPIPS | Face PSNR / SSIM / LPIPS | Left-ear PSNR / SSIM / LPIPS | Lipstick/hand PSNR / SSIM / LPIPS |
|---|---|---|---|---|
| Original: `960`, strict TSDF, previous 16-source pool | `22.3292 / 0.785907 / 0.128026` | `22.8196 / 0.748816 / 0.116513` | `21.9858 / 0.731215 / 0.144765` | `22.2919 / 0.851327 / 0.083395` |
| `1920`, strict TSDF, nearest 16 of all 62 | `22.1360 / 0.790894 / 0.129190` | `23.0198 / 0.754475 / 0.113190` | `22.5522 / 0.747698 / 0.125935` | `22.7534 / 0.861605 / 0.078411` |
| **Selected: `1920`, strict TSDF + relative island filter, angular 16** | **`23.8313 / 0.775921 / 0.128127`** | **`26.0281 / 0.734430 / 0.111190`** | **`24.8845 / 0.699793 / 0.100040`** | **`26.0380 / 0.837956 / 0.096828`** |

The selected result improves left-ear PSNR by `+2.90 dB` and LPIPS by `-0.0447` (`31%`) and
removes the visible detached island. Actor/object PSNR improves by `+1.50 dB`, face PSNR by
`+3.21 dB`, and face LPIPS by `-0.0053`. SSIM and lipstick LPIPS do not improve with the angular
pool; this is a real source-view colour/texture trade-off rather than a hidden full-frame gain.
The all-62 texture control is the more balanced SSIM/lipstick variant, while angular-16 is selected
because the stated gate prioritizes the ear and face and is chosen from calibration only.

The angular texture-camera count is not monotonic because the independently selected subsets are
not nested and hard nearest-fill lets the closest retained camera own almost every actor pixel:

| Angular texture pool | Actor/object PSNR / SSIM / LPIPS | Left-ear LPIPS | Observation |
|---:|---|---:|---|
| 8 | `19.3677 / 0.763941 / 0.222363` | `0.322102` | Too sparse; reject |
| 12 | `23.8302 / 0.775234 / 0.129621` | **`0.098479`** | Essentially tied with 16 |
| 16 | `23.8316 / 0.775245 / 0.129635` | **`0.098479`** | Selected coverage/speed compromise |
| 24 (nearest 16 rendered) | `24.3791 / 0.798528 / 0.190812` | `0.227573` | Source-switch structure hurts LPIPS; reject |
| 32 (nearest 16 rendered) | `22.0736 / 0.790560 / 0.130363` | `0.124062` | No advantage over 16 |

The validated CUDA COLMAP build produced all 62 full-resolution geometric maps with mean valid
coverage `38.52%` (minimum `26.72%`). The final TSDF has one connected component after generic
filtering. Relaxed PatchMatch gates, a narrower TSDF band, half+full multiscale fusion and
per-pixel `best-view` source switching were also tested and rejected because they increased holes,
fragmentation, or LPIPS.

## Insights

1. **The practical no-training replacement is calibrated COLMAP PatchMatch,
   not a feed-forward foundation model.** With the same texture renderer its
   lipstick/hand LPIPS is `0.083786`, effectively tied with the
   Splatfacto-depth reference's `0.083527`; face LPIPS is marginally better.
   It removes neural scene training, although dense stereo still runs once per
   frame.
2. **A fixed metric surface is essential.** MonoMVSNet with five views appeared
   to achieve LPIPS `0.1106` on its own surviving surface, but it rendered only
   `26.7%` of the target image. On the common surface it scores `0.4782` and the
   visual holes are correctly penalized.
3. **More nearly redundant views can hurt learned multi-view depth.** DA3 Large
   fails much more severely with 62 cameras than with the evenly spaced 16.
   The result is consistent with a small-baseline, domain-shifted multi-view
   prior becoming mutually inconsistent; it is not evidence that the source
   calibration itself worsened.
4. **Pose alignment is not MapAnything's main failure.** Its predicted rig can
   be similarity-aligned to the supplied camera centres with normalized RMSE
   `0.0136`, yet its geometry remains visibly incorrect. The remaining problem
   is depth/shape accuracy at the actor, not a single global scale mistake.
5. **Generic monocular or benchmark-MVS depth is a useful fallback, not a
   geometry replacement at this quality bar.** DA3 Large is the only learned
   candidate that keeps a mostly coherent face, but it remains far behind
   PatchMatch at the silhouette and held object.
6. **MVSMamba runs, but its released checkpoint does not transfer to this
   capture.** Its official code/checkpoint loaded strictly and its CUDA Mamba
   kernel ran successfully on the current GPU, so the failure is measured
   geometry/domain performance rather than an installation failure. It
   reconstructs only `22.17%` of the target image and is worse than
   MonoMVSNet on all three fixed regions.
7. **The ear defect was a geometry-resolution problem, followed by a tiny disconnected-island
   problem.** Changing the colour compositor could not repair it. Full-resolution two-pass
   PatchMatch repaired the supported surface; a relative topology gate removed the residual island
   without inspecting the held-out image or encoding an ear location.
8. **The selected recipe is frame-reusable, but cross-frame visual validation is still pending.**
   `run_colmap_patchmatch_tsdf.py` exports fixed train-only calibration, forbids masks, builds its
   texture subset from camera geometry, and records every command and input hash. A second-frame
   `000901` canary successfully resolved 62 train cameras and generated the complete dry-run/export
   recipe. Dense MVS on that second frame has not yet been run, so no cross-frame quality claim is
   made here.

The next product-oriented gate is therefore a short camera-path render using
COLMAP PatchMatch -> strict TSDF -> hard view-dependent texture selection. It
must measure temporal holes and source-switch flicker before the Splatfacto
teacher is removed from the per-frame pipeline.
