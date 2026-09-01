# DEC5 frame 000899: depth-surface 4/16/32 camera ladder

## What was tested

- The same held-out eval camera and the same required target train camera
  `E004_B005_1210I7` in every run.
- Nested train subsets of 4, 16, and 32 cameras selected by greedy farthest-point
  sampling on camera-centre directions. The selection does not inspect pixels.
- Full-resolution JPEG RGB, fixed GLOMAP poses/intrinsics, no masks, appearance
  embeddings, camera optimization, U-Net, or LPIPS training loss.
- Bounded depth-Nerfacto with AABB `[-0.15, 0.15]^3`, 128 final samples/ray,
  progressive hash levels, and dense depth rendered by the clean GLOMAP
  Splatfacto teacher.
- Corrected DS-NeRF semantics: `sigma` is a distance standard deviation and the
  Gaussian target is normalized. All other settings and seed are matched.

## Results

### Matched step 4000

| Cameras | Target train PSNR | Train SSIM | Train LPIPS | Eval PSNR | Eval SSIM | Eval LPIPS |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 33.3821 | 0.854534 | 0.294790 | 17.7170 | 0.706393 | 0.554364 |
| 16 | 31.9625 | 0.817723 | 0.339994 | 19.0052 | 0.743530 | 0.518355 |
| 32 | 30.9690 | 0.804547 | 0.379195 | 19.6896 | 0.747249 | 0.525414 |

Visual inspection agrees with the target-train metrics: at equal optimizer
steps, blur increases monotonically from 4 to 16 to 32 cameras. Conversely, the
held-out view improves strongly from 4 to 16 cameras. At 4k, 32 cameras have the
best eval PSNR/SSIM, while 16 cameras have slightly better LPIPS than 32.

### Longer runs

| Cameras | Step | Target train PSNR | Train SSIM | Train LPIPS | Eval PSNR | Eval SSIM | Eval LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 12k | 33.2100 | 0.854176 | 0.308821 | 18.8837 | 0.762585 | 0.474377 |
| 32 | 12k | 32.3658 | 0.836433 | 0.293808 | 19.8429 | 0.772316 | 0.466352 |

At 12k, 32 cameras win all three eval metrics and are visually the least bad
held-out reconstruction. The target train view remains softer than the 16-camera
run by PSNR/SSIM, although its LPIPS is marginally lower. Both eval renders still
contain translucent background/actor layers, so 32 cameras improve coverage but
do not solve the remaining surface ambiguity.

Runs:

- `/dev/shm/standard_nerf_runs/dec5_000899_bounded_nerfacto_geometry/depth-nerfacto/camera4_tight015_denseSplatDSstdnorm001_final128_s42_to4k`
- `/dev/shm/standard_nerf_runs/dec5_000899_bounded_nerfacto_geometry/depth-nerfacto/camera16_tight015_denseSplatDSstdnorm001_final128_s42_to12k`
- `/dev/shm/standard_nerf_runs/dec5_000899_bounded_nerfacto_geometry/depth-nerfacto/camera32_tight015_denseSplatDSstdnorm001_final128_s42_to12k`

## Insights

1. Camera count creates a real optimization trade-off at fixed step count. More
   cameras reduce the number of ray updates per image and make memorizing the
   shared target train view harder.
2. Added angular coverage helps held-out interpolation more than it hurts the
   target train view: eval PSNR improves by `+1.9726 dB` from 4 to 32 cameras at
   4k, even though target-train PSNR falls by `-2.4130 dB`.
3. The 32-camera eval run is still improving from 8k to 12k (`+0.0323 dB` PSNR,
   `+0.00582` SSIM, `-0.01481` LPIPS). It has not crossed the declared SSIM
   plateau gate yet.
4. More views alone do not force a single opaque surface. The remaining eval
   ghosting must be attacked with surface-thickness/visibility constraints rather
   than another camera-count increase.
