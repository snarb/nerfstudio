# DEC5: 150 real-time meshes, a slow central camera, and near-view texture selection

Final reviewed video: `/mnt/data/dec5_smooth_150_final/video.mp4`.
The final revision fixes the notable 000971 shelf with a matched full-block TSDF
control; all 150 times pass the requested artifact-tolerant viewing gate. Minor
hair/shoulder fringe and skin seams remain; artifact-free geometry is not claimed.

## What was tested

Continuation of the [local geometry / diffusion pilot](dec5_diffusion_mesh_repair.md),
not a replacement of the 150-time-frame objective with a static flythrough.
The pilot's three independently edited synthetic views did **not** establish a
reliable mesh-completion method. Its local cylinder/chin repair is specific to
000973 and is not blindly applied to moving geometry in this video.

This opt-in renderer reuses the existing 150 temporal PatchMatch/TSDF meshes,
000899 through 001197 inclusive, every second source index. Frame 000973 uses
the independently audited full-block TSDF variant selected by the earlier joint
texture workflow. Other frames initially retain the campaign geometry. No new SfM,
PatchMatch, NeRF training, temporal morph, or diffusion RGB is used in that initial
texture pass. The final single-time 000971 repair control below reruns PatchMatch
with identical historical JPEG inputs, then changes only TSDF block activation.

Each frame reads 62 immutable train EXRs into RAM. Exposure is the same fixed
10.570320292648677 multiplier for every camera and time, followed by Reinhard
and sRGB. Previously fitted, time-invariant camera RGB response profiles and
static registration corrections are reused. Time-dependent registration is off.
The three original held-out cameras never provide prediction RGB.

Geometry-surface source labels use the existing graph-labeling texture algorithm;
the final RGB sample always comes from **one** real train image. The 9-pixel
low-pass is only for label costs, not the output RGB. Native texture sampling
checks all four bilinear depth taps. In particular, mesh self-visibility is not
claimed to prove independent observed surface support.

Two source-label recipes were compared:

- Baseline: existing geometry/normal-quality surface labels.
- Candidate: multiply label quality by a calibration-only angular preference,
  `max(exp(-0.5 * (camera_axis_angle / 4 degrees)^2), 1e-5)`. This favors a source
  whose viewing direction resembles the target. Geometry, fixed color profiles,
  projection, visibility, final sampling and RGB averaging policy are unchanged.

The path is a periodic, arc-length-parameterized convex combination of four
central train-camera anchors, with radius 0.3 and a fixed look-at point. It is
not a sequence of interpolations that stop or change velocity at physical cameras.
Two mesh normalization gauges exist in the inventory: each pose is transferred
through calibration coordinates into its own mesh gauge. Rotation matrices must
not be scaled when inverting pose normalization.

The output has 150 distinct chronological source instants at 30 fps: **5 seconds**.
Camera motion is periodic; actor motion is not claimed to loop seamlessly.
There is no duplicated final camera endpoint and no video-frame interpolation.

## Results

### Controlled held-out face-only comparison

The two methods were rendered from the same held-out `F004_B005_1210O9` pose,
using the same fixed GT display transform and GT-defined manual face ROI. GT was
read only after prediction. These are two-time controls, **not** aggregate metrics
for 150 novel viewpoints, which do not have exact-view ground truth.

| Time | Recipe | Face PSNR ↑ | Face SSIM ↑ | Face LPIPS ↓ |
|---|---|---:|---:|---:|
| 000973 | baseline | 26.3241 | 0.871116 | 0.111797 |
| 000973 | near-view preference | 29.3511 | 0.911371 | 0.082150 |
| 001059 | baseline | 27.2139 | 0.893535 | 0.120688 |
| 001059 | near-view preference | 30.0531 | 0.920120 | 0.090256 |

Face metrics only; no full-frame image metrics or loss. The manual ROI remains
independent of rendered geometry, so missing prediction pixels are not masked out.
These controls support reduced view-dependent texture mismatch, not a general
claim that all 150 images improve or that the lipstick backside becomes correct.

Machine-readable controls:

- `/mnt/data/lookcloser_dec5_5a3_smooth_temporal_eval_baseline/face_metrics.json`
- `/mnt/data/lookcloser_dec5_5a3_smooth_temporal_eval_prior/face_metrics.json`

### Camera motion and throughput

| Quantity | Old temporal path | New central path |
|---|---:|---:|
| Angular speed min, degrees/s | 27.8738 | 2.9105 |
| Angular speed median, degrees/s | 42.8483 | 3.0458 |
| Angular speed max, degrees/s | 49.5949 | 3.2080 |

The median camera speed is approximately 14 times lower. This does not slow the
actor or synthesize intermediate temporal frames.

A single initial frame took 23.1 seconds. Four disjoint render workers reused one
96-GB GPU, with CPU EXR loading and source-camera raycasts parallelized. The baseline
supervisor finished its remaining inventory in 1172 seconds after 14 initial
frames had already completed; all four workers exited zero. This is rendering
parallelism, **not concurrent PatchMatch on one GPU**. Worker/PID/GPU memory/free
space/progress checks are logged every 30 seconds. No source or prior output was
removed. Numerical recipes and per-frame hashes are frozen before rendering.

### Initial publications and historical review status

- Baseline 150-frame video: `/mnt/data/lookcloser_dec5_5a3_smooth_temporal_150_v2/smooth_temporal_150.mp4`
- Near-view full-pass workspace: `/mnt/data/lookcloser_dec5_5a3_smooth_temporal_prior_canary`
  (the directory name reflects its initial three-time canary; its immutable
  request contains all 150 source instants).
- Per-frame PNGs, native prediction, camera IDs, raycast depth, source labels,
  result manifests and atomic completion receipts are retained under `frames/`.
- Native-scale `face`, `ear_hair`, `lipstick_hand` panels and nearby real-camera
  comparisons are under `contact_sheets/`; explicit LLM reviews under `visual_reviews/`.

The near-view pass finished all 150 frames in 1142 seconds of four-worker supervision
(three canary frames were already available). Worker exits were all zero, with
no CUDA/OOM error in their logs. Per-frame render time min/median/max was
19.447 / 27.839 / 39.657 seconds. These times include concurrent-worker contention,
not an isolated GPU benchmark. All workers and the review/encoding watcher have exited.

All **150** native face/ear/lip panels were inspected in 38 groups and 21 recorded
review batches; a further 15-frame overview was decoded from the actual MP4.
The independent audit verifies all frame receipts/hashes, existing geometry hashes,
62 distinct non-held-out source cameras, finite depth/render coverage, one fixed
exposure, and the complete chronological video inventory. After inverting the
individual mesh gauges, camera-speed max/min is 1.000166; angular speed remains
2.9105–3.2080 degrees/s, including the periodic camera seam.

The initial version was a **reviewed candidate, not complete geometry-repair success**.
The face/ear/hand motion is coherent and broad skin source boundaries are weaker,
but tan hair/shoulder fringe, polygonal nose/skin seams and metal-edge chips remain.
**000971 is a notable local geometry failure:** a thin, wrong skin-colored triangular
shelf extends beside the tube and across the adjacent lip area. It is explicitly
flagged `fail_local_geometry` in `frames_audit.csv`; the other 149 frames are
`accepted_with_known_artifacts`, not strict artifact-free passes. No frames are
pending or uncertain, and no whole-body/camera collapse was observed. The quality
goal was not claimed fully achieved at that stage; see the final revision below.
A nearby real train image is qualitative context, **not exact target GT**.

Review evidence: [000971 defect and neighboring instants](/mnt/data/lookcloser_dec5_5a3_smooth_temporal_prior_canary/contact_sheets/036_039/lipstick_hand.png),
[later head/ear turn](/mnt/data/lookcloser_dec5_5a3_smooth_temporal_prior_canary/contact_sheets/116_119/face.png),
[MP4-decoded temporal overview](/mnt/data/lookcloser_dec5_5a3_smooth_temporal_prior_canary/encoded_temporal_overview.png).

Selected candidate MP4 SHA-256:
`84ff5df551de824d3c9220fe26c43a90f9cb56f1f98bc5ac57bad8ff3944796e`.
Portable reviewed-video bundle: `/mnt/data/dec5_smooth_150_reviewed`.
It contains MP4, all 150 PNGs, review panels/verdicts and provenance, **not** the
150 mesh binaries. Existing mesh paths/hashes remain in the request. The separate
000973 local-repair GLB remains documented in the linked geometry pilot report.

Thirty targeted tests pass, covering pose normalization/invariant projection,
angular source preference, worker partition coverage, unfinished/duplicate review
rejection, and the existing local-repair/path tests. Only new opt-in helpers,
tests and targeted documentation are committed; unrelated worktree edits and
existing model/single-frame runner defaults are preserved.

### Final revision: 000971 repaired, all 150 real instants retained

The [matched geometry control](dec5_temporal_shelf_repair.md) establishes the
cause and repair of the shelf. Six real train views show no shelf; representative
false faces have 22–25 reliable farther-depth observations and zero near-surface
observations under the stated strict footprint rule. Per-view-block TSDF again
produces the defect on freshly reproduced depth; full-block-union integration
removes it without a new lip hole. No synthetic RGB, hand sculpting or per-frame
texture/exposure change is used for the selected correction.

Final workspace: `/mnt/data/lookcloser_dec5_5a3_smooth_temporal_150_repaired_v3`.
Exactly 149 verified unchanged renders are reused with explicit ancestor-receipt
hashes; the repaired 000971 render is substituted at the identical camera/time.
This is not falsely reported as another 150-frame rerender. All 150 native reviews
remain covered; the changed four-frame group was inspected again. The actual new
MP4 was additionally checked as a 15-image overview and nine consecutive native
lipstick/hand crops around the replacement. The camera path and earlier held-out
face controls are unchanged.

Final audit: 150 unique chronological instants, 150 explicit reviewed results,
zero pending/uncertain, zero remaining **notable** local geometry failures in the
video review and zero catastrophic frames. All 150 are
`accepted_with_known_artifacts`, **not** strict artifact-free passes. Hair/shoulder
fringe, small skin seams and edge shimmer remain accepted under the user's stated
video policy. This completes the reviewed-video delivery scope, not a claim of
perfect unseen backside geometry or completion of the stricter historical campaign.

Final MP4 SHA-256:
`1366ddd220e2b47f5b291251dd2d766969733f740c74aaf579fb0660cd718db2`.
Portable bundle `/mnt/data/dec5_smooth_150_final` includes `video.mp4`, all 150 PNGs,
review panels, decoded transition evidence, manifests and the geometry-control
report. Existing mesh paths/hashes remain in the request; mesh binaries are not
duplicated into this video download bundle.

Final targeted regression suite: **70 passed, 1 intentionally skipped GPU case**.
The source importer also has a tested byte-only provenance-copy fix for NFS;
camera/depth/render numerics and existing model defaults are unchanged.

[Final consecutive decoded transition](/mnt/data/lookcloser_dec5_5a3_smooth_temporal_150_repaired_v3/encoded_transition_032_040/contact.png),
[matched shelf control](/mnt/data/lookcloser_dec5_5a3_shelf_diagnosis_000971/full_block_control/comparison_detail.png).

## Insights

1. A fixed camera color calibration alone cannot remove view-dependent shading,
   highlights, occlusion mistakes or wrong mesh correspondences. The angular
   source preference directly addresses part of this mismatch without averaging
   high-frequency RGB. Its favorable face control does not repair geometry.
2. A small smooth camera trajectory and correct cross-frame pose normalization
   solve distinct problems. Both must be checked; good-looking single snapshots
   cannot establish temporal stability or rule out a normalization jump.
3. The remaining tan fringe is retained geometry/source contamination around hair
   and shoulder silhouettes, not a room-reconstruction target. A confident local
   geometry or train-silhouette support method would be needed to remove it.
   Current source depth tests alone cannot certify or eliminate it.
4. The user's artifact-tolerant video policy is recorded as
   `accepted_with_known_artifacts`, never as `strict_artifact_free=true`.
   Catastrophic visual failures must stay explicit, and incomplete reviews must
   prevent terminal audit. This does not supersede the stricter historical mesh
   campaign gate or imply that every geometric defect has been solved.

## Reproduction

Run from the repository with the nerfstudio virtual environment; these helpers
do not change existing model or single-frame runner defaults.

```bash
python LookCloser/scripts/render_smooth_temporal_mesh_video.py init
python LookCloser/scripts/run_smooth_temporal_workers.py supervise --workers 4
# Isolated near-view request and initial three-time check:
python LookCloser/scripts/temporal_texture_view_prior.py --output OUTPUT_PRIOR
# After visually reviewing the canary, resume the same request across all times:
python LookCloser/scripts/run_smooth_temporal_workers.py supervise --output OUTPUT_PRIOR
python LookCloser/scripts/review_encode_smooth_temporal_video.py sheets --output OUTPUT_PRIOR
python LookCloser/scripts/review_encode_smooth_temporal_video.py encode --output OUTPUT_PRIOR
# Review each saved native face/ear/lip panel, then record actual verdicts:
python LookCloser/scripts/audit_smooth_temporal_video.py record --output OUTPUT_PRIOR --groups 000_003 --notes 'Actual review findings'
python LookCloser/scripts/audit_smooth_temporal_video.py audit --output OUTPUT_PRIOR
```
