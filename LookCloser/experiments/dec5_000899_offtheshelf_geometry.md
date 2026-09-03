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
2. COLMAP 4.1.1 calibrated PatchMatch geometric depth from 62 train cameras;
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

The next product-oriented gate is therefore a short camera-path render using
COLMAP PatchMatch -> strict TSDF -> hard view-dependent texture selection. It
must measure temporal holes and source-switch flicker before the Splatfacto
teacher is removed from the per-frame pipeline.
