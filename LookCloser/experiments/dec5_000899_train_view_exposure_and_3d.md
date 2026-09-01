# DEC5 frame 000899: train-view exposure and 3D diagnostics

## What was tested

- One temporal instant (`000899`) at 1920x1080, converted to JPEG with the shared
  dataset-agnostic converter.
- No masks, appearance embeddings, U-Net, LPIPS training loss, frequency grid,
  frequency-aware sampling, or feature re-weighting.
- The previously blurred train render was traced to physical camera
  `E004_B005_1210I7` (COLMAP image id 19, source `frame_train_00043.jpg`).
- Immutable one-, two-, and four-camera overfit datasets duplicate that exact
  train image into eval. `--preserve-normalization-context` retains every source
  pose solely for `focus/up/auto_scale` calculation, while explicit filename
  lists restrict RGB supervision to the selected cameras. Consequently the
  dataparser transform, scale (`0.14484221414663417`), sparse point cloud, and
  AABB are byte-identical across the camera ladder and the all-62 dataset.
- Every run uses the same LookCloser base field, scalar auto-SfM AABB
  (`+-0.129047513`), 4096 train rays/update, 4096 fixed samples/ray,
  `softplus(+1)`, max hash resolution 8192, and seed 42.
- GLOMAP sparse geometry and the trained all-62 LookCloser density were exported
  in the same Nerfstudio model coordinate system. No train/eval pixels are used
  to colour or alter the geometry diagnostic.

The single-frame JPEG copy for independent work on `dev3` is:

`/home/ubuntu/datasets/dec5_000899_single_frame_jpg/`

It contains 63 verified JPEG files (62 train and one eval), totalling 49,926,412
bytes.

## Results

### Exact train-view ladder

| Train cameras | Step | Mean sampled rays/source pixel | Expected pixels seen at least once | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1k | 1.975 | 86.13% | 28.4197 | 0.938295 | 0.133330 |
| 1 | 3k | 5.926 | 99.73% | 42.1523 | 0.984201 | 0.013033 |
| 1 | 4k | 7.901 | 99.96% | 44.2917 | 0.990903 | 0.007215 |
| 2 | 6k | 5.926 | 99.73% | 43.4588 | 0.987413 | 0.011343 |
| 4 | 2k | 0.988 | 62.76% | 34.9218 | 0.907524 | 0.154323 |
| 4 | 4k | 1.975 | 86.13% | 37.2789 | 0.946323 | 0.072623 |
| 4 | 6k | 2.963 | 94.83% | 38.9555 | 0.963914 | 0.042681 |
| 4 | 8k | 3.951 | 98.08% | 40.3656 | 0.973442 | 0.028171 |
| 4 | 10k | 4.938 | 99.28% | 41.1024 | 0.979513 | 0.020382 |
| 4 | 12k | 5.926 | 99.73% | 42.4730 | 0.983692 | 0.014877 |

The expected-coverage column is the uniform-sampling reference
`1 - exp(-steps * rays_per_batch / (cameras * width * height))`; it is not a
claim that neighbouring camera rays provide no shared 3D information.

The early four-camera step-2k prediction visibly blurs lipstick, facial detail,
hair, shirt texture, and brick seams. The same camera at step 12k is nearly
indistinguishable from ground truth. The model can therefore memorize the
problem camera with fixed calibration and AABB; neither an intrinsically bad
camera nor insufficient hash-grid capacity explains the train blur.

Artifacts:

- Four-camera early prediction:
  `/dev/shm/lookcloser_diagnostics/train4_step2k_render/eval_pred_0000.png`
- Four-camera final GT/pred render:
  `/dev/shm/lookcloser_runs/dec5_000899_E004_B005_train_overfit_ladder/lookcloser/train4_fixednorm_fixed4096_p1_s42_to12k/renders_best_step-000012000/eval_img_0000.png`
- One-camera final GT/pred render:
  `/dev/shm/lookcloser_runs/dec5_000899_E004_B005_train_overfit_ladder/lookcloser/train1_fixednorm_fixed4096_p1_s42_to4k/renders_best_step-000004000/eval_img_0000.png`

For comparison, the all-62 fixed-4096 run at effective step 6k samples only
`0.191` rays/source pixel on average, corresponding to a uniform reference of
about 17.4% of source pixels seen at least once. Even step 16k reaches only
`0.510` samples/pixel (39.9% reference coverage). This explains why increasing
camera count without increasing useful image-ray throughput reproduced the
train-view blur.

### 3D geometry

The GLOMAP sparse cloud contains 134,498 points and forms a coherent human in
all orthographic projections: head, face, hair, arms, and clothing are readily
recognizable. It contains little support for the brick-room background. The
calibration is therefore not a random or grossly malformed 3D reconstruction,
although small local pose errors can still limit novel-view micro-detail.

The all-62 LookCloser effective-6k raw high-density volume is much less
surface-localized. Inside a robust actor box, 57.6% of its selected p99 density
points lie in the SfM-supported region; their median nearest-SfM distance is
0.00681 model units (4.22 diagnostic voxels), and only 5.64% fall within one
voxel. Raw hidden density is not itself render contribution, so this evidence is
paired with the direct target-ray measurement: broad density-weight profiles
remain around the face/lips/hair and are extremely broad on unsupported bricks.

Artifacts:

- Side-by-side GLOMAP versus LookCloser density:
  `/home/brans/lookcloser_temp/dec5_000899_3d_diagnostics/glomap_vs_lookcloser_effective6k.jpg`
- Rotatable combined NeRF/SfM/camera PLY:
  `/home/brans/lookcloser_temp/dec5_000899_3d_diagnostics/lookcloser_all62_effective6k/nerf_sfm_cameras_combined.ply`
- GLOMAP sparse model and cameras:
  `/home/brans/lookcloser_temp/dec5_000899_3d_diagnostics/sfm_glomap_model_space/sfm_cameras_combined.ply`
- Machine-readable NeRF diagnostics:
  `/home/brans/lookcloser_temp/dec5_000899_3d_diagnostics/lookcloser_all62_effective6k/summary.json`

## Insights

1. Gross GLOMAP geometry, AABB placement, dataset normalization, target-camera
   focus, masks, and field capacity are rejected as the primary cause of the
   blurred train view. All are held constant while the prediction progresses
   from visibly blurred to LPIPS 0.0149.
2. The causal bottleneck is the product of two requirements in the current base
   path: high along-ray resolution needs 4096 field queries/ray, while high
   image-space coverage needs many distinct rays. With 62 full-HD views, the
   dense marcher cannot provide both efficiently.
3. Splatfacto's sharp train views are consistent with this diagnosis: its
   rasterized primitives receive dense image supervision without paying 4096
   MLP/hash queries for every supervised pixel ray.
4. Arbitrary raw density away from visible surfaces is a symptom of weakly
   constrained/under-exposed volume, not sufficient proof of rendered ghosting.
   The rendered ladder and per-ray weight profiles are the direct evidence.
5. The next base-model gate is therefore a sampler with substantially fewer
   useful 3D evaluations/ray: stock Nerfacto proposal sampling, followed by
   bounded Instant-NGP occupancy traversal. Both remain free of masks,
   embeddings, and image-space perceptual losses. Only after a sharp base field
   is established should frequency-aware allocation be reintroduced.
