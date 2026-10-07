# Luster 000470: opacity and trusted depth

## What was tested

Status: primary and depth experiments in progress; no quality recipe has
been promoted. Branch `lookcloser/000470-opacity-depth`, starting at `f73cc3c5`.
Campaign root: `/home/pavlo/renders/luster_000470_opacity` on `olive-lark`.
Maximum budget: 48 single-device GPU hours, starting with frequency fits.

All 165 undistorted calibrated source cameras are preserved: 162 train;
held-out cameras 012/095/150; train review 011/097/151. The dataset uses area
downsampling to long side 1920 (96 portrait, 69 landscape), raw RGB and studio
background, no source masks or silhouette guards. Intrinsics use actual axis
resize ratios. Normalization uses train cameras only; an AABB encloses their
envelope with 50% axis padding. Train-only frequency maps are being refitted
with the existing progressive 1000-step-per-level recipe.

Planned comparisons: unmasked volume control, hard first-intersection SDF,
strong volume opacity-profile regularization, reviewed semantic transparency,
and PatchMatch depth alone/in combination. The control follows seed42/B4096,
hash23/max8192, corrected FAS, safe exponential and corrected SH, fixed256
warmup4096 then ARM coarse .001, FR1→.3 at20k, full-state base-LR reduction
.01→.002 at32k, maximum40k. Surface models retain the frequency-aware field
and appearance recipe, but use surface search/eikonal in place of volume
occupancy traversal and distortion. This architectural difference is explicit.

The older masked result `31.026 / .9613 / .04042` at40k is historical context,
not the control for this experiment. Full-frame background has changed.

## Results

The initial2000-step primary gates completed; full trajectories are in progress.
Initial RGB contacts and fixed native
face/hair/hand/clothing patches have been visually inspected and retained:

- [Initial views](/home/pavlo/renders/luster_000470_opacity/data/initial_contact.jpg)
- [Initial patches](/home/pavlo/renders/luster_000470_opacity/data/initial_patches.jpg)
- [Data audit](/home/pavlo/renders/luster_000470_opacity/data/audit.json)
- [Campaign budget](/home/pavlo/renders/luster_000470_opacity/campaign.json)

### Initial native comparison at2000

| Representation | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Visual gate |
| --- | ---: | ---: | ---: | --- |
| Unmasked volume | 9.047 | .51568 | .84535 | Failed: held-out face almost gray; train face blurry |
| Hard first-hit SDF | 8.516 | .41153 | .79492 | Failed: opaque fragmented/mottled wrong surfaces |

These are early checkpoints, not final recipe scores. Both are continuing to8000.
All six SDF views have binary alpha with mean1.0 and no intermediate values.
That property alone does not repair wrong geometry. The volume also has mean
accumulated alpha near1.0, despite its blurred reconstruction.

The first32 pixel/RGB batch hashes match across arms, include all162 training
cameras and reach portrait row1919. No depth priors or masks were used.
Runtime parser poses match all165 prepared cameras exactly; intrinsics differ
only by float32 rounding (maximum.000244px). Adam performed1999/1990 updates
in2001 iterations for volume/SDF, so the bad images are not explained by
persistent AMP overflow.

On four fixed train097 skin rays, the volume has termination90%-mass widths
.316–.361 normalized units and expected depths.181–.210. SDF first hits lie
at.748–.800. Three corresponding nearest pixels have trusted PatchMatch ray
depths1.028–1.042; the fourth has no trusted measurement and remains unknown.
These diagnostic subpixel rays use nearest-pixel measured references explicitly;
the reference is not substituted into any rendered output.

- [Face comparison](/home/pavlo/renders/luster_000470_opacity/reviews/gate_002000/face.jpg)
- [Hands](/home/pavlo/renders/luster_000470_opacity/reviews/gate_002000/hand.jpg)
- [Clothing](/home/pavlo/renders/luster_000470_opacity/reviews/gate_002000/clothing.jpg)
- [Hair](/home/pavlo/renders/luster_000470_opacity/reviews/gate_002000/hair.jpg)
- [Termination profiles and measured references](/home/pavlo/renders/luster_000470_opacity/reviews/gate_002000/ray_profiles_with_depth/profiles.png)
- [Runtime camera audit](/home/pavlo/renders/luster_000470_opacity/runtime_camera_audit.json)

Frequency preprocessing finished in16932s (4.70h); the final train-only audit
passed162 maps/162 train/3 eval. Primary runs started after this audit.
The SSH account `coder` cannot traverse `/home/pavlo`; reviewed exports will be
copied to `/var/tmp/lookcloser_000470_delivery`, with a named-user read/traverse
ACL on that task directory. No home-directory access was broadened.

Twenty-three focused opacity/depth/permission behavioral tests pass: first-crossing occlusion, empty rays,
either SDF orientation, exact zero, finite surrogate gradients, interval-split
invariance, empty-gap penalties, independent ray sums, saturated gradients,
two-layer depth cancellation rejection, and invalid-depth exclusion.
A tiny CUDA surface training/evaluation smoke completed through step8 with
finite RGB. Its low-capacity/stride8 outputs validate infrastructure only;
they are not comparable to the HD experiments. Full-state continuation is
checked separately. Optimizer, scheduler, scaler and RNG restoration passed;
the tiny GPU run is not bitwise reproducible (uninterrupted/resumed parameter
values differed). Primary arms use independent pixel/occupancy/frequency RNG
streams to prevent eikonal probes from changing the sampled image sequence.

The broader focused sampler/density/packed-render/RNG suite passed38 tests.
The measured-depth subset also passed tiny CUDA training through16 steps in
both volume and surface modes. Native export produced six HD views and both
fixed review paths encoded successfully on a tiny smoke checkpoint. These
tests validate infrastructure, not reconstruction quality.
No training quality claim follows
from these infrastructure tests.

A semantic CUDA smoke exposed an FP16 alpha input to Nerfacc's FP32 scan.
Alpha and color accumulation now use FP32, with an added half-appearance
regression case. Both forbidden-alpha and permitted-alpha CUDA smokes passed;
the permitted case changed opacity-head parameters through RGB gradients.
Parameter-only surface→semantic initialization preserved exact field hashes
and fresh optimizer/grid/sampler state. These are infrastructure checks only;
the artificial smoke permission grids are excluded from all quality arms.

### PatchMatch build canary

Exact COLMAP commit5509fffe was built locally with CUDA12.8, CPU Ceres2.2 and
GUI/OpenGL disabled. An added `<cassert>` include fixes compilation of the
observation manager; it does not change stereo mathematics. The native
`sm_120` build failed: photometric maps repeated across cameras and all16
geometric maps were zero. Those artifacts are rejected and retained.

Rebuilding the same source with `CMAKE_CUDA_ARCHITECTURES=90-virtual` produced
16 unique, nondegenerate geometric maps, with whole-image coverage3.54–33.33%.
The corrected independent3-camera/1px/1% gate gives face ROI coverage60.13%
on train097 and18.31% on train151. Confidence policy v2 backprojects the actual
nearest source pixel from which the measured depth was read; the old policy
used the continuous projected coordinate and concealed subpixel correspondence
error. The new subpixel regression test passes. The native depth/validity face patches were inspected:
observed depth follows the head/face; holes and rejected pixels remain unknown.
This qualifies the PTX build for the full162-train-camera reconstruction; it
does not certify that all surviving depths are correct.

The MVS workspace changes principal points by−0.5 only to express Nerfstudio
pixel-center rays in COLMAP MVS integer-sample coordinates. No RGB is
undistorted again, resized again, or masked. Depths are in the already
train-normalized coordinate frame, so import applies normalization scale1.

- [PTX canary inventory](/home/pavlo/renders/luster_000470_opacity/depth_canary_ptx90/depth_inventory.json)
- [Face depth/validity review](/home/pavlo/renders/luster_000470_opacity/depth_canary_confidence_v2/face_review.jpg)
- [Confidence manifest](/home/pavlo/renders/luster_000470_opacity/depth_canary_confidence_v2/manifest.json)
- [Rejected native build](/home/pavlo/renders/luster_000470_opacity/depth_canary/canary_failure.json)

The full162-camera PatchMatch reconstruction finished: all162 depth hashes are
unique, no map is empty, and measured whole-frame coverage is1.15–38.62%
(median8.53%). Policy-v2 confidence retains a median6.47% of the whole frame;
the studio background accounts for much of the image area.

| Train camera | Trusted face ROI | Trusted hand ROI | Trusted clothing ROI |
| --- | ---: | ---: | ---: |
| 011 | 63.88% | 3.64% | 1.93% |
| 097 | 68.82% | 57.13% | 38.12% |
| 151 | 55.59% | 49.53% | 63.57% |

Native RGB/measured/trusted face crops, train097 hand, train151 clothing and
the low-coverage train011 hand were visually reviewed. The accepted samples
follow visible surface shape; rejected regions remain unknown. Qualification
allows testing the prior, rather than certifying absolute depth accuracy.

- [Full confidence face review](/home/pavlo/renders/luster_000470_opacity/depth_confidence/face_review.jpg)
- [Full ROI coverage](/home/pavlo/renders/luster_000470_opacity/depth_confidence/roi_coverage.json)
- [Qualification record](/home/pavlo/renders/luster_000470_opacity/depth_confidence/visual_review.json)
- [Full PatchMatch inventory](/home/pavlo/renders/luster_000470_opacity/depth_full/depth_inventory.json)

### Semantic permission review

The first candidate was rejected after visual inspection because polygon cuts
and ornament edges were labeled as possible hair transparency. Revised maps
exclude artificial domain borders and saturated ornament neighborhoods, retaining
only90/165/557 pixels on train011/097/151. The revised overlays were reviewed;
skin/hand vetoes remain. A spatial grid still requires agreement in at least
two visible training cameras and whole-voxel skin vetoes; these2D annotations
alone do not enable alpha learning.
The voxel veto uses the bounding footprint of all eight projected corners:
an unprotected center is insufficient if the voxel overlaps skin. Regression
tests cover this case and unbounded camera-plane footprints.

[Reviewed sparse permissions](/home/pavlo/renders/luster_000470_opacity/semantic_annotations_v2/review.jpg)

### Independent calibration checks

Train-only SIFT matching plus an independently estimated fundamental matrix
selects image correspondences; residuals are then measured against the unchanged
supplied calibration. Pairs010/011 and097/098 have median epipolar residuals
.157px/.134px and93.9%/97.7% within1px. Pair151/152 has only16 independent
inliers and median1.10px. Additional pairs094/151 and151/153 have825/136
independent inliers, median.210px/.222px, and94.1%/94.1% within1px;
they do not support changing camera151 based on the weak151/152 pair.
No calibration or pose optimization has been applied.

The face-arc pivot is frozen from a qualified train097 center depth and used
only for the common evaluation camera path. Every model follows identical
poses; its own learned depth never chooses the trajectory.

## Insights and next steps

The GT head crops show a largely solid styled wig and ornaments. Semantic
transparency must stay conservative around actual strands/fringes; labeling
the entire wig transparent is not justified by these images.

Frequency fits are slower than the earlier masked campaign (first four-worker
fit about329s); elapsed time and GPU/controller/worker state are retained in
`logs/supervision.jsonl`. No fit count is substituted for completion or a
train-only cache audit.

Strong priors use partitioned samples around the trusted interval. Intervals
are not appended on top of existing intervals (which would double optical
thickness). Unknown rays keep normal sampling. Surface priors additionally
supervise signed distance at the measured point and measured free space;
linear-tail penalties preserve correction gradients far from the target.

### Integrator correction and continuous surface trials (2026-10-07)

Legacy ARM with a1024 interval cap could truncate the far part of a ray in the
large unmasked camera-envelope AABB. Three of four inspected skin rays had
near-zero accumulated opacity despite density peaks at trusted depth. The same
checkpoint rendered with4096 or16384 intervals reaches opacity≈1 and the
measured skin surface. Consequently early volume scores, including the depth4
score4.85dB, do not establish model quality and must not rank recipes.
The four legacy volume continuations were stopped and retained. Corrected
volume trials start from the common step4096 state, enable the allocator that
preserves the whole ray, train with4096 intervals and evaluate with16384
intervals/coarse step.0005. They do not inherit legacy evaluation scores.

- [Same-checkpoint sampling probe](/home/pavlo/renders/luster_000470_opacity/reviews/depth4_sampling_probe.json)
- [Measured-depth ray profiles](/home/pavlo/renders/luster_000470_opacity/reviews/depth4_002000_profiles/profiles.png)

The initial implicit-gradient surface trials still used discontinuous
nearest-cell frequency weights for geometry. Their2k held-out PSNR/SSIM/LPIPS
were8.075/.3916/.8054 without depth and9.545/.5240/.6629 with depth4; neither
is promoted. A regression demonstrates that a frequency-cell boundary can
change the signed-field sign without a continuous zero. New isolated trials
use continuous interpolated hash features for signed geometry and the original
frequency reweighting for appearance. Implicit first-hit derivatives and a
physical surface depth objective avoid constraining CDF surrogate thickness.
Defaults preserve archived configurations.27 focused tests and a16-step
measured-depth CUDA smoke pass; these validate execution, not image quality.

No final40k checkpoint or download artifact is available at this stage.
