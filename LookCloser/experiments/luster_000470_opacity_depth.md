# Luster 000470: opacity and trusted depth

## Current measured status

The ordinary no-prior baseline has completed40k and is visually coherent.
The accepted intermediate recipe is ordinary LookCloser plus opacity.1, with
no MVS prior: its newly evaluated38k checkpoint scores20.1719/.816426/.222436
(PSNR/SSIM/LPIPS). It uses unmasked HD RGB, softplus density, legacy SH directions,
hash23/max8192, B4096, FR.3 after20k, box[-1.5,1.5]^3, coarse.00625/cap1024 and
matched near.05 in train/eval. Source calibration is transformed once.
All train-only frequency fits are complete.

The independent density branch passes exact initial native parity and a normal
quality gate, but costs47% more optimization time without an established gain
by itself. Its first2k depth screen narrows trusted train-face termination by
4–7% and improves inner-face PSNR slightly, while face LPIPS also worsens slightly.
It remains experimental. Opacity.3 and all tested hard-boundary/signed-head
initializations are not promoted. Hard opaque40k, semantic transparency quality
training, a stronger successful geometry recipe and repeat robustness remain
incomplete; the full original goal stays active.

The latest accepted38k native eval PNGs and64 orbit/face frames are verified
and copied under `/var/tmp/lookcloser_000470_delivery/accepted_opacity38k_native`.
Both videos decode to32 frames; source eval hashes match. Remaining softness,
wrist noise, studio errors and skirt ghosts mean this is not artifact-free.
Detailed chronological evidence follows; the initial plan below is historical
and has been superseded where later gates identify failures.

## What was tested

Historical initial status: primary and depth experiments in progress; no quality recipe has
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

### Standard baseline gate (2026-10-07)

User-directed priority: reproduce healthy ordinary LookCloser before extending
experimental models. Compare matched2k/8k steps, then add one change at a time;
large early quality regression triggers diagnosis. Time to comparable image
quality is measured, rather than assuming that depth guidance accelerates it.
The prepared continuous-surface40k continuations have not been launched.

The old successful frame report used black-composited masks, background
supervision and a much tighter silhouette-derived AABB. These remain excluded
from the new experiment. Its31dB result is not an unmasked baseline certificate.
The new `standard_preset_008000` uses the existing raw RGB/frequency maps and
already-normalized calibration with ordinary `LookCloserModel`,
`VanillaDataManager` and `Nerfstudio` parser. All model configuration fields
match the current standard preset: softplus density, original SH handling,
FR1/FAS1, historical ARM cap1024/coarse.00625, warmup4096, hash23/max8192.
Standard scene bounds are[-1.5,1.5] on each axis. Parser orientation/centering
and autoscaling are disabled because normalization is already applied. No
source masks, depth manifest, new objective or experimental renderer is used.

- [Exact preset receipt](/home/pavlo/renders/luster_000470_opacity/standard_preset_receipt.json)
- [Baseline request](/home/pavlo/renders/luster_000470_opacity/requests/standard_preset_008000.json)

The previous two incomplete jobs (corrected volume8k and continuous surface
with depth2k) no longer had live processes after the execution-access reset.
Their final metrics are missing; they are interrupted evaluations, not quality
results. Cause is unverified. The completed continuous no-depth2k prototype
scored8.412/.3983/.8161 and is not promoted.

The ordinary preset completed8k in473 seconds including the two native6-view
evaluations (setup excluded). All face/hair/hand/clothing contacts and the
eval012 RGB/alpha panel were inspected. Face/eyes/lips, fingers and garment
are coherent; skin banding/noise, soft embroidery and studio-frame artifacts
remain. This is a learnable baseline for incremental tests, not final quality.

| Standard preset step | All-eval PSNR | SSIM | LPIPS |
| ---: | ---: | ---: | ---: |
| 2000 | 18.482 | .7223 | .4314 |
| 8000 | 18.585 | .7856 | .2853 |

At8k, held-out face PSNR is22.96/19.54/25.48 for012/095/150. Full-frame
metrics include the unmasked studio and cannot be compared to the old masked
31dB score. No recipe is promoted on these scores alone.

- [Native face comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_008000/face.jpg)
- [Hand comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_008000/hand.jpg)
- [Hair comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_008000/hair.jpg)
- [Clothing comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_008000/clothing.jpg)
- [8k metrics](/home/pavlo/renders/luster_000470_opacity/runs/standard_preset_008000/eval_008000/metrics.json)

The next pair resumes the identical ordinary4k checkpoint through8k. Both
forks isolate pixel/occupancy/frequency RNG streams for matched pixel batches;
only the candidate adds trusted-depth strength1 and its local sampling band.
They retain standard bounds, softplus, SH handling, FR/FAS and ARM settings.
Run sequentially to measure comparable single-device wall time. Alpha
regularization and semantic permissions stay disabled.

An opt-in strict volume checkpoint adapter initializes only two bookkeeping
buffers and preserves every field/grid parameter and optimizer state. With
both strengths0, an eight-ray CUDA comparison verifies exact field parameters,
renderer differences below1e-5, matching objectives within numerical tolerance
and maximum hash-gradient difference9.54e-6 (MLP differences0). This is scoped
parity evidence, not a quality test or a claim of bitwise CUDA determinism.
30 focused behavior tests pass, including refusing unmarked checkpoints without
opt-in and retaining strict errors for missing field weights.

[CUDA extension parity](/home/pavlo/renders/luster_000470_opacity/reviews/incremental_standard_parity.json)

### Matched4k→8k depth gates

Both forks preserve standard settings and resume the same ordinary4k state.
The first32 RGB/pixel batches are identical (18747 trusted observations in the
depth arm). Jobs run sequentially; times below exclude setup and include the
same4k additional updates. Evaluation always uses learned RGB without GT depth.

| Variant at8k | All-eval PSNR | SSIM | LPIPS | Training seconds | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Paired standard control | 19.438 | .7802 | .2787 | 182.1 | Current reference |
| Depth strength1, guided partition | 21.473 | .8043 | .2613 | 226.5 | Reject promotion: skin ROI regression |
| Depth strength.25, guided partition | 19.511 | .7848 | .2992 | 228.2 | Reject: blurred face/hands |

Depth1 improves full-frame scores and most clothing crops, but held-out face
LPIPS worsens in all three views (+.0042/+.0099/+.0263); face PSNR worsens on095
and150 (−1.21/−1.42dB). Training face regressions are larger. All6 contacts for
face/hair/hand/clothing were inspected. Lower strength.25 worsens face and hand
PSNR/LPIPS in all6 checked views; native095 face and097 hand visibly blur.
Neither is accepted just because background improves. Strength1 costs24.4%
more training time per4k updates; faster convergence is not established.

- [Depth1 face comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_depth1_008000/face.jpg)
- [Depth1 hand comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_depth1_008000/hand.jpg)
- [Depth1 review](/home/pavlo/renders/luster_000470_opacity/reviews/standard_depth1_008000/review.json)
- [Matched ray receipts](/home/pavlo/renders/luster_000470_opacity/standard_depth_pair_receipt.json)

Code inspection identifies a nonminimal change: the guided branch replaces
ARM with uniform256 intervals plus65 GT-band edges on trusted rays, including
after the warmup. The weaker band also has coarser spacing at a fixed edge count.
Consequently these are not depth-objective-only tests. The next isolated trial
sets `volume_depth_guided_sampling=False`, retaining the ordinary sampler and
adding only the trusted-depth objective. Archived guided defaults stay intact.
31 focused tests pass, including preserving the native renderer with trusted
metadata when guidance is disabled.

Frozen fine rendering is a separate diagnostic axis on both paired checkpoints.
The control's fine result is19.257/.7905/.2740; coarse training-selection scores
remain unchanged. Exports now support ordinary volume configs without SDF
fields. Whole-field density profiles include occupancy-skipped regions, so
their quantile widths are distinct from actual packed renderer-weight widths;
the profiler records both instead of equating them.

### Native-sampler depth and first positive opacity candidate

Keeping native sampling with depth strength1 yields21.122/.7969/.2634 in
216.2 training seconds (276.3 including evaluation). It still fails the
foreground gate: face LPIPS worsens in all3 held-out views, and camera095 face
PSNR drops3.71dB. All6 face/hand/hair/clothing contacts were inspected. Thus
the sampling substitution was not the sole regression cause. This candidate
is retained as diagnostic evidence, not extended to40k.

The next test changes only opacity regularization to.01, without depth input,
using the same4k parent and matched first32 batches. It completes in194.4
training seconds (259.2 including evaluation).

| Paired8k variant | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Ordinary reference | 19.438 | .7802 | .2787 |
| Only opacity.01 | 19.715 | .7996 | .2736 |

Held-out face PSNR changes+1.22/+1.59/−.25dB; hand+1.34/+2.42/−.36dB.
Clothing LPIPS improves in all3 eval views. All6 contact sheets for each ROI
were inspected: no large foreground regression is visible, although skin
noise/softness and small face LPIPS regressions on012/150 remain. This is a
provisional positive candidate, not a final recipe or an artifact-free claim.
It still needs longer paired gates and seed-repeat validation.

- [Opacity face comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity001_008000/face.jpg)
- [Opacity hand comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity001_008000/hand.jpg)
- [Opacity review/metrics](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity001_008000/review.json)
- [Actual renderer profiles](/home/pavlo/renders/luster_000470_opacity/reviews/standard_008000_actual_profiles/profiles.json)

The four inspected skin rays already have a thin dominant peak: actual
renderer90% widths are.0045–.0055 normalized units for the reference and
.0047–.0052 for opacity.01, so this diagnostic does not prove that opacity.01
removed secondary layers. The raw uniform whole-field widths≈1.03 include
occupancy-skipped near-camera density and must not be called visible fog.
Native depth1 moves means near the three measured reference depths and narrows
the dominant peak to.0033–.0038, while image quality of skin regresses.
Geometry and appearance therefore need separate visual assessment.

Stronger only-opacity.1 and weaker native-sampler depth.25 requests are prepared
for subsequent small comparisons; neither was launched at this handoff.
All workers/evaluators exited, GPU is idle, and the goal remains incomplete.
Reviewed GT/reference/opacity comparison images were physically copied into
the private coder-readable delivery directory, without changing home access:

```bash
scp coder@olive-lark:/var/tmp/lookcloser_000470_delivery/incremental_8k/face_comparison.jpg ~/Downloads/
```

This downloads an8k comparison preview, not the requested final hard-opaque40k
artifact. Semantic/full-depth combinations, hard opacity, final40k paths and
seed-repeat verification remain outstanding.

### Stronger opacity, native weak depth and combined8k gate

| Matched4k→8k variant | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Only opacity.1 | 19.651 | .7983 | .2754 |
| Only depth.25, native sampler | 19.839 | .7891 | .2778 |
| Opacity.01 + depth.25, native sampler | 19.421 | .7934 | .2811 |

All6 face/hand/hair/clothing sheets were inspected for each. Opacity.1 has
similar visual behavior to.01 and slightly worse overall scores. Native
depth.25 avoids the severe blur of its guided predecessor, but held-out face
LPIPS still worsens+.0076–.0201 against the ordinary reference. Adding depth.25
to opacity.01 worsens all full-frame metrics; that combination is not promoted.
The new review helper validates shared calibration, matching native GT crops,
six camera identities and first32 pixel/RGB batches before building comparisons.
For continuations from different intermediate parents, it also verifies their
explicit common-checkpoint ancestry; it never auto-accepts visual quality.

The new actual-renderer front-mass diagnostic gives an important qualification
to the metric ranking. On the three skin rays with trusted nearest-pixel depth,
probability mass before the reference interval is1.967–4.079% for control,
1.327–4.517% for opacity.01 (mixed), and.139–.170% for opacity.1. Weak native
depth.25 reduces it to.290–.458%; the combination to.110–.161%. The opacity.1
surface mode stays near the same reference depth. These three rays support a
real reduction in secondary foreground layers, not a scene-wide certificate.
The lower LPIPS of opacity.01 alone is therefore insufficient to select a
final transparency recipe. Both opacity strengths continue to12k alongside
the ordinary control, retaining full optimizer/grid/RNG state.

- [Opacity.1 comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_008000/comparison.json)
- [Native weak depth comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_depth025_native_sampler_008000/comparison.json)
- [Combined comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity001_depth025_008000/comparison.json)
- [Front-layer mass/profiles](/home/pavlo/renders/luster_000470_opacity/reviews/standard_008000_front_mass/profiles.json)

The ordinary12k control scores16.912/.7906/.2764. Face quality improves
(held-out face PSNR23.76/25.73/23.70), while studio/background errors lower
full-frame PSNR. A frozen20-ray background probe compares legacy1024/coarse.00625,
corrected1024 at the same step, and corrected4096/coarse.001. RGB MAE is
.14494/.14494/.14112 respectively; the allocator-only difference is negligible
on these rays. This probe does not explain the global decline as truncation,
and does not establish that every ray is unaffected. The12k comparison keeps
ordinary sampling settings rather than mixing renderer changes with training.

[Frozen12k budget probe](/home/pavlo/renders/luster_000470_opacity/reviews/budget_probe_12k.json)

The next depth experiment separates interval precision from loss influence:
strength1 keeps its original uncertainty interval, while
`depth_prior_loss_mult=.001` reduces the old.01 objective coefficient tenfold.
The historical default is unchanged. A behavioral test confirms proportional
loss scaling with unchanged targets;32 focused tests pass. This request is
prepared but not yet run. It is not evidence of an image-quality improvement.

### Matched12k continuation gate

All three branches continue their own8k states with the same common4k
ancestor. Native12k comparison checks verified ancestry, identical first32
batches and GT crops. All6 ROI contact sheets were visually inspected.

| Pinned12k branch | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Ordinary control | 16.912 | .7906 | .2764 |
| Opacity.01 | 17.537 | .7900 | .2675 |
| Opacity.1 | 18.924 | .7979 | .2596 |

Opacity.1 improves all3 held-out hand LPIPS and hand PSNR by+2.09/+1.30/+3.86dB.
Held-out face PSNR changes−.09/−.56/+3.48dB; face LPIPS improves012/150 but
worsens095 by.0127. Train151 face PSNR falls1.45dB. Remaining skin softness,
noise and studio artifacts are not resolved. Opacity.01 has a larger095 face
regression (−2.67dB). Opacity.1 is now the main provisional branch for further
gates, supported by better12k metrics and its8k front-layer reduction. This
does not prove that its12k front-layer statistics match8k; remeasurement is pending.

Full-frame12k PSNR is below8k for all branches because studio errors increase,
despite clearer foreground details. The historical best-checkpoint selector
therefore still selects8k, and remains authoritative for best-metric exports.
Continuation must use the latest12k state rather than reverting to that best
checkpoint. No long/40k branch or artifact-free recipe is promoted here.

- [Opacity.1 at12k](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_012000/comparison.json)
- [Face at12k](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_012000/face.jpg)
- [Hands at12k](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_012000/hand.jpg)

Reviewed12k comparison previews are copied to
`/var/tmp/lookcloser_000470_delivery/incremental_12k`. All jobs finished at the
visual gate; GPU is idle. Next: the prepared precise-interval/gentle-weight
depth test, then any passing combination and bounded20k comparisons.


### 16k/20k opacity and precise depth with gentler influence

The opacity.1 branch continued from its latest12k state without other changes.
A separate trial added native-sampler depth strength1 and coefficient.001 to
that same12k parent, retaining the original1%/uncertainty interval. Another
matched4k→8k trial tested depth alone with the same coefficient.

| Branch | Step | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| standard_opacity01_020000 | 16000 | 18.599 | 0.8030 | 0.2524 |
| standard_opacity01_020000 | 20000 | 18.230 | 0.7974 | 0.2526 |
| standard_depth1_weight0001_008000 | 8000 | 19.686 | 0.7902 | 0.2776 |
| standard_opacity01_depth1_weight0001_016000 | 16000 | 18.243 | 0.7944 | 0.2566 |

Adding precise depth at16k reduces full-frame PSNR.356dB and worsens LPIPS.00422
against opacity.1 alone. All six face/hand/hair/clothing contacts were inspected.
The095 face rectangle loses5.56dB. A supplemental fixed central-half face crop
from saved8bit RGB loses13.35dB, with visibly darker skin; the background-margin
hypothesis is rejected. Inner-face SSIM worsens in all six views. This branch
is rejected at16k, rather than extended blindly. Depth alone at8k also fails
foreground review despite acceptable full-frame metrics.

- [Fixed inner-face measurements](/home/pavlo/renders/luster_000470_opacity/reviews/face_core_depth16k/metrics.json)
- [Fixed inner-face comparison](/home/pavlo/renders/luster_000470_opacity/reviews/face_core_depth16k/contact.jpg)
- [Full16k comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_depth1_weight0001_016000/comparison.json)

A20×20 fixed integer-pixel grid per face/hand/clothing region on train cameras
011/097/151 probes actual packed renderer weights without sampling overrides.
The ordinary-opacity16k face mode already agrees with trusted PatchMatch:
median relative errors+.025%,+.015%,+.019%;90th-percentile signed errors below
.15%. Median premature face mass is1.714%,.334%,0%; added depth reduces all
three medians to0%, yet changes skin color. Thus lower front-layer mass alone
is not sufficient for visual acceptance. The result supports testing gentler
or bounded influence, rather than assuming wholesale geometry displacement.
These train-region probes do not certify unseen geometry or depth correctness.

- [Opacity16k alignment](/home/pavlo/renders/luster_000470_opacity/reviews/depth_alignment_control16k.json)
- [Added depth16k alignment](/home/pavlo/renders/luster_000470_opacity/reviews/depth_alignment_candidate16k.json)

Two independent next screens preserve that same opacity12k parent: coefficient
.0001 with historical distance cost; or coefficient.001 with `interval_mass`
cost. The latter penalizes off-interval termination mass and missing in-interval
mass without scaling remote layers by distance. Native RGB sampling, targets,
seed and opacity strength stay unchanged. It is opt-in; the historical default
is preserved.25 opacity tests pass, including gradient direction, empty-ray
penalty, invalid-ray freedom, and bounded treatment of remote layers.
The ordinary control is actively continuing12k→20k for matched gates.


### Ordinary control through20k, matched opacity continuation

The ordinary baseline finished with unchanged standard field, native sampling,
calibration and shared4k ancestry. Comparisons verify identical first32 RGB/pixel
batches and six native GT crops. The20k face/hand/hair/clothing sheets were all
visually inspected.

| Branch | Step | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| standard_paired_020000 | 16000 | 17.688 | 0.7949 | 0.2583 |
| standard_opacity01_020000 | 16000 | 18.599 | 0.8030 | 0.2524 |
| standard_paired_020000 | 20000 | 16.645 | 0.7808 | 0.2620 |
| standard_opacity01_020000 | 20000 | 18.230 | 0.7974 | 0.2526 |

Opacity.1 retains a measured advantage at16k and20k. At20k it avoids much of
ordinary095 face/garment darkening and remains the provisional continuation.
The prescribed feature-reweighting change1→.3 happens at20k; this boundary is
not yet evidence of quality after adaptation to that change. Continue latest
20k states to23k/26k, retaining optimizer/scaler/RNG state. Full-frame historical
best selection remains earlier8k;20k preview delivery is explicitly a pinned
comparison rather than the best-metric final artifact.

The20k opacity train-face grid has premature-mass medians1.382%,.431%,0%
(cam011/097/151), p90s4.130%,2.787%,.907%; median90% termination widths
.00314/.00304/.00295 normalized scene units. Small forward layers remain on
some rays. These actual-renderer train measurements replace extrapolation
from four8k rays, but do not certify all views or a hard opaque surface.

- [20k paired evidence](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_20000_matched/comparison.json)
- [20k termination probe](/home/pavlo/renders/luster_000470_opacity/reviews/depth_alignment_opacity20k.json)
- Delivery: `/var/tmp/lookcloser_000470_delivery/incremental_20k`.

The focused opacity/depth-witness/semantic suite passes32 tests. The new
interval-mass experiment is prepared; the separate coefficient.0001 distance
screen is actively training. Neither is promoted without its16k native gate.


The12k→16k historical-distance screen with coefficient.0001 scores
18.257/.7916/.2553, below opacity-only18.599/.8030/.2524. All six fixed
inner-face/hand/hair/clothing rows were inspected. The095 inner-face PSNR
falls14.68dB with visible darkening, despite tenfold gentler influence; this
branch is rejected at the gate. The interval-mass screen is running separately.
A metadata-only control is prepared: keep the measured-depth loader but set
depth strength0, using the same12k parent. This will test the full training
path rather than attributing repeated color drift solely to the objective.

[Gentler16k comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_depth1_weight00001_016000/comparison.json)


### Repeated-control audit: revise causal interpretation

The interval-mass candidate also fails the original16k foreground gate. Before
attributing its darkening to depth, a12k→16k metadata-only control retained
measured-depth loading but set depth strength0. A second repeat removed the
depth manifest as well. Both use the same parent, seed, pixels and current code.

| Run at16k | PSNR | SSIM | LPIPS | 095 inner-face delta vs original control, dB |
| --- | ---: | ---: | ---: | ---: |
| standard_opacity01_depth1_mass_016000 | 18.045 | 0.7894 | 0.2541 | -12.20 |
| standard_opacity01_depth0_metadata_016000 | 18.321 | 0.7964 | 0.2548 | -10.05 |
| standard_opacity01_repeat_016000 | 18.504 | 0.7977 | 0.2527 | -13.67 |

**The earlier causal explanation is not established.** Even the repeat without
depth metadata loses13.67dB on the095 interior while full-frame PSNR changes
only−.094dB. Thus substantial local run variability confounds those single-run
depth comparisons. Candidate non-promotion remains valid, but does not prove
the depth objective itself caused the entire color change. Do not select a
recipe or infer geometry from full-frame scores alone.

At14k the metadata-only control has different hash/MLP weights from the old
control despite identical first32 RGB/pixel batches. A4096-ray same-model,
same-RNG probe isolates metadata: forward errors below4.5e−6 and field gradient
errors below7.7e−6. With deterministic PyTorch enabled, RGB/depth/accumulation
and MLP gradients match exactly; TCNN hash gradients still differ by up to
3.94e−6. This is single-batch evidence, not long-run reproducibility. The flag
is opt-in and restored during export. It does not cover custom TCNN kernels.

- [Metadata-only comparison](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_depth0_metadata_016000/comparison.json)
- [Repeat without depth](/home/pavlo/renders/luster_000470_opacity/reviews/standard_opacity01_repeat_016000/comparison.json)
- [14k weight audit](/home/pavlo/renders/luster_000470_opacity/reviews/depth0_metadata_14k_weight_parity.json)
- [4096-ray probe](/home/pavlo/renders/luster_000470_opacity/reviews/metadata_gradient_parity.json)
- [Deterministic probe](/home/pavlo/renders/luster_000470_opacity/reviews/metadata_gradient_parity_deterministic.json)

Next: ordinary control and opacity.1 continue their latest20k states to23k/26k
with the same deterministic PyTorch setting, allowing adaptation after the
20k FR switch. The ordinary continuation is active. New depth trials remain
unpromoted; no reliable depth gain or training-speed gain is claimed.


### Plain26k gate: improve the control before more depth

The ordinary no-prior continuation in deterministic PyTorch mode scores
17.054/.7749/.2583 at23k and16.401/.7816/.2601 at26k. Held-out095 face PSNR
is16.43/18.16dB; the23k native GT/render face was inspected and has a large
color/brightness error. Thus the ordinary preset itself needs improvement;
its failure cannot be assigned to newly added opacity or depth objectives.
The prepared opacity26k continuation is deferred while repairing this control.

The repository already has `correct_sh_directions`, mapping unit directions
from[-1,1] to TCNN's[0,1] SH input, as in the standard Nerfacto field. It is
False in the literal LookCloser preset. The older masked leader used corrected
SH together with exponential density, but that report does not establish
unmasked quality or isolate the two changes. A matched26k→28k screen now tests
**SH mapping alone**: identical parent, native sampler, softplus density, FR.3,
optimizer/RNG and deterministic mode, no new priors. The ordinary28k control
is active; the corrected-SH request is prepared.

A separate source audit also finds that frequency-grid side-channel rays use
integer coordinates, whereas the ordinary RGB RayGenerator uses pixel centres
(offset.5). This discrepancy remains unchanged in the SH pair so it does not
introduce a second simultaneous change. Correcting it requires its own gate.
No complete or artifact-free recipe is claimed.


### Warm SH screen and cold documented leader baseline

The matched28k ordinary/SH-only continuations score16.801/.7873/.2591 and
16.681/.7900/.2613. All six face-interior/hand/hair/clothing rows were inspected.
Changing SH mapping on the already-trained softplus checkpoint does not repair
095 skin in2k steps; this warm branch is not promoted. It does not establish
that corrected SH is ineffective when trained from initialization.

A review-only photometric audit projects trusted097 central-face points into
095 and nearby train cameras.095 median sRGB differences are+.0076/+.0045/+.0033
(1815 points, target visibility not independently certified); confidence-qualified
096 differs−.0503/−.0275/−.0347,154 differs+.0132/+.0039/+.0051. Source differences
exist but do not by themselves explain the full rendered darkening. No exposure,
pose or held-out color adjustment is fitted.

[Photometric correspondence audit](/home/pavlo/renders/luster_000470_opacity/reviews/face_photometric_correspondence.json)

The historical masked leader used exponential density and corrected SH, whereas
the literal preset retains legacy softplus/raw SH. The next baseline restores
that documented field math **from initialization** with ordinary LookCloser,
VanillaDataManager, built-in parser, scene scale1.5 and unmasked source RGB.
No opacity/depth/semantic priors or hull guard are enabled. Independent RNG
streams and deterministic PyTorch are recorded. This is a baseline restoration,
not a clean two-parameter causal comparison against old differently-seeded stages.
Its preflight verifies the ordinary components and only the declared field/RNG
differences. Cold2k/4k/8k native gates run under active supervision.

[Leader preflight](/home/pavlo/renders/luster_000470_opacity/reviews/leader_unmasked_preflight.json)


The cold unmasked leader scores8.941/.4916/.8706 at2k and8.956/.4663/.8802
at4k. The095 held-out face stays featureless fog, while train097 is a noisy,
blurred shape. Both native gates were visually inspected. The worker was
terminated by visual review (last compact progress7100), preserving2k/4k
checkpoints and `visual_abort.json`. No8k completion or leader-quality gain
is claimed. All training workers are terminal.

A same-checkpoint26k counterfactual changes only inference FR.3→1 on native
095 face: PSNR18.159→19.969, SSIM.8916→.9028, LPIPS.07567→.06624. GT/control/
counterfactual was inspected. This supports testing a separate FR1 continuation,
but does not repair the complete color error or establish whole-view quality.
It is a scoped diagnostic, not a new best checkpoint.

- [Cold leader visual abort](/home/pavlo/renders/luster_000470_opacity/runs/leader_unmasked_008000/visual_abort.json)
- [FR counterfactual](/home/pavlo/renders/luster_000470_opacity/reviews/fr_counterfactual26k/metrics.json)
- [FR comparison](/home/pavlo/renders/luster_000470_opacity/reviews/fr_counterfactual26k/contact.png)

The original goal remains incomplete: no accepted depth improvement, no
40k hard opaque artifact, and no semantic3D quality arm. Next useful gate is
baseline repair with one controlled FR continuation or the separately audited
pixel-centre correction, before further opacity/depth combinations.


### Camera-near density explains a large apparent foreground regression

The FR1 continuation alone scores16.427/.7836/.2617 at28k against ordinary
16.801/.7873/.2591. It is not promoted under the original evaluator; those
scores still contain the subsequently identified camera-near contamination.

On the same ordinary26k checkpoint, changing inference near plane0→.02 raises
095 native-face PSNR18.159→27.211, SSIM.8916→.9259, LPIPS.07567→.05405. Exactly
matching the old train near.01 produces no change: the entire effect cannot be
attributed solely to train/eval mismatch. The harmful layer extends beyond.01.
Near.05 yields27.501/.9271/.05211; further.25/.5 changes very little. Camera095,
097,012,150 origins are inside the standard AABB. All qualified train-depth
pixels have ray distance≥.5744; none is below.05. Unknown depth is not certified
by this audit.

A LookCloser-only opt-in collider now preserves the configured near plane in
eval. Historical defaults and shared nerfstudio collider behavior remain
unchanged. New train/eval near.05 settings can be tested consistently without
source masks or RGB edits.34 focused tests pass, covering inside-camera rays,
unchanged outer-box entry, historical defaults and positive dummy intervals.

| Same ordinary26k weights, native all3 eval views | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Original eval near0 | 16.401 | .7816 | .2601 |
| Eval near.05 | 19.832 | .7989 | .2277 |

All six face/hand/hair/clothing contacts were inspected;095 face and garment
lose the gray overlay. Noise, softness and studio artifacts remain. This is a
renderer-domain diagnosis, not a learned depth/opacity improvement or a final
best-checkpoint export. Existing prior comparisons must be reevaluated with
the same ray bounds before interpreting their foreground regressions.
Cold exponential4k fog persists at near.05 and remains unpromoted.

- [Six-view near comparison](/home/pavlo/renders/luster_000470_opacity/reviews/near_fixed_standard26k/comparison.json)
- [Minimum measured distances](/home/pavlo/renders/luster_000470_opacity/reviews/trusted_near_distance_audit.json)
- [Exact near.01/.02 diagnostic](/home/pavlo/renders/luster_000470_opacity/reviews/near_match_counterfactual26k/metrics.json)
- Delivery: `/var/tmp/lookcloser_000470_delivery/near_fixed_baseline26k`.

Next active check: reevaluate the shared4k→8k ordinary control and native
strength1 depth candidate with the same near.05 evaluator. Do not combine old
near0 and corrected scores in checkpoint selection.


### Shared4k→8k depth trial under corrected near bounds

Both saved branches are reevaluated with native sampling and eval near.05.
The control scores21.254/.7776/.2643; native depth strength1 scores
21.531/.8013/.2617. Six identical GT face/hand/hair/clothing rows were compared
and inspected. The depth branch still has visibly noisier skin, wrinkled/blurry
hands and worse foreground details despite slightly better full-frame scores.
It remains unpromoted at8k. Near-plane repair explains the large gray overlay,
but does not make this strong early depth objective acceptable.

[Corrected-bound depth comparison](/home/pavlo/renders/luster_000470_opacity/reviews/near_fixed_depth1_8k/comparison.json)

All training/evaluation workers are terminal at the review gate. The next
prepared request is `standard_near05_028000`: same ordinary26k parent, train
and eval near.05, no added priors and no old incomparable history. After this
baseline gate, reevaluate gentler/later opacity+depth branches with identical
bounds before choosing the next small change. The complete40k hard-opaque,
semantic and artifact-free recipe requirements remain outstanding.


### Near.05 training gate at28k

The ordinary model resumes26k→28k with train/eval near.05, no new priors and
full optimizer/scaler/RNG state. It scores19.906/.8070/.2339. All six native
face/hand/hair/clothing rows were inspected against26k evaluated at the same
near.05. Quality stays coherent with mixed small changes:095 face PSNR−1.99dB,
LPIPS+.0038;150 face PSNR+.61dB, LPIPS−.0028. This is a usable control rather
than a claim of uniform quality improvement.28k becomes the common parent for
new matched short prior tests. Historical near0 scores are excluded from its
selection history.

[Near.05 longitudinal review](/home/pavlo/renders/luster_000470_opacity/reviews/near05_training28k/comparison.json)

The incremental reviewer now accepts explicit reevaluation directories while
still checking source ancestry, first32 identical pixel/RGB batches, native
GT crops and identical effective ray-bound/sampling settings. The corrected
8k depth comparison passes those checks and remains visually unpromoted.
The late16k opacity control under near.05 scores20.622/.8154/.2358; the matching
gentle-depth candidate is being reevaluated before choosing a fresh branch.


### Late gentle depth after corrected evaluation bounds

The common12k opacity.1 parent branches reevaluated at16k/near.05 score
20.622/.8154/.2358 (opacity only) and20.654/.8132/.2364 (plus depth1,
coefficient.001). Source ancestry, identical first32 pixel/RGB batches and
identical native bounds were verified. All six face/hand/hair/clothing contacts
were inspected. Gray darkening is removed by the near correction; remaining
appearance differences are small, but face LPIPS worsens in all six views.
The branch is not promoted. The earlier large skin-darkening attribution to
this prior alone is superseded by the ray-bound evidence.

[Verified late-depth comparison](/home/pavlo/renders/luster_000470_opacity/reviews/near_fixed_opacity_depth16k_verified/comparison.json)

The fresh near.05 ordinary28k→30k control scores19.839/.8076/.2315. Two separate
same-parent2k screens are prepared: opacity.1 with no depth; or depth strength10
with `interval_mass`, coefficient.0001, native sampling and no opacity penalty.
The depth coefficient totals.001; median trusted face tolerances are
.001109/.001036/.001019 scene units (cam011/097/151), with measured uncertainty
as a floor. This tightens geometry influence rather than penalizing arbitrary
remote distance. Neither screen is accepted before native30k visual review.
Opacity training is active under supervision.

[Tight interval receipt](/home/pavlo/renders/luster_000470_opacity/reviews/tight_depth_interval_receipt.json)


### Fresh matched30k opacity and tight-depth screens

Both independent2k screens start from ordinary near.05 step28k, preserve
optimizer/scaler/RNG, native sampling, pixel batches, calibration and GT. The
verifier confirms ancestry and identical first32 RGB/pixel batches. All six
face/hand/hair/clothing contacts were inspected for both candidates.

| Pinned30k branch | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Control | 19.839 | .8076 | .2315 |
| Opacity.1 only | 20.060 | .8161 | .2266 |
| Tight interval-mass depth only | 19.796 | .8099 | .2318 |

Opacity.1 improves the held-out metrics with small mixed foreground changes:
inner-face PSNR−.273/+.512/−.022dB; train inner faces lose.42/.80/.96dB, with
small LPIPS differences. It is provisional rather than uniformly better.
Tight depth keeps appearance close (inner-face changes+.430/−.103/+.158dB
held-out) and improves or preserves several foreground LPIPS values. Neither
candidate removes all observed noise, softness or studio artifacts.

Actual native-renderer termination is probed on fixed20×20 grids in train
011/097/151 face/hand/clothing regions, using confidence-qualified measurements
and unchanged checkpoint sampling. The following face medians are conditional
on trusted points; they are not a whole-scene or held-out certificate.

| Train face | Control90% width | Opacity90% width | Tight-depth90% width | Control front mass | Opacity front mass | Tight-depth front mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 011 | .003296 | .003113 | .002588 | 2.746% | 2.080% | 2.406% |
| 097 | .002930 | .002771 | .002483 | 1.220% | .349% | 1.157% |
| 151 | .002954 | .002771 | .002563 | 2.190% | .063% | 1.778% |

Widths are normalized scene units. Front mass uses the same reference interval
max(2uncertainty,1% depth) across all probes, whereas the new training objective
uses0.1% with the uncertainty floor. Opacity mainly removes secondary front
layers; tight depth reduces core width by roughly13–21% without a large RGB
regression in this gate. This supports a single additional-depth fork from the
opacity30k state, compared to opacity-only30k→32k. Both requests are prepared;
no combination gain is yet established. The32k LR reduction remains identical.

- [Opacity30k verified comparison](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity01_030000/comparison.json)
- [Tight-depth30k verified comparison](/home/pavlo/renders/luster_000470_opacity/reviews/near05_depth10_mass_030000/comparison.json)
- [Control termination](/home/pavlo/renders/luster_000470_opacity/reviews/near05_control30k_alignment.json)
- [Opacity termination](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity30k_alignment.json)
- [Tight-depth termination](/home/pavlo/renders/luster_000470_opacity/reviews/near05_depth30k_alignment.json)

The exporter now restores selected reevaluation near bounds; an actual loaded
8k camera-ray check verifies near.05. Preview copies are available at
`/var/tmp/lookcloser_000470_delivery/incremental_near05_30k`. They are pinned30k
comparisons rather than a final best-checkpoint or hard40k artifact. All jobs
are terminal at this review gate. The complete original goal remains active.

### Matched opacity and tight-depth continuation, 32k–36k

What was tested: from the same opacity30k checkpoint, retain opacity coefficient
.1 or add confidence-qualified interval-mass depth strength10 with coefficient
.0001 (effective .001). Near.05 applies in train and eval, native sampling stays
unchanged, full optimizer state is restored and the first32 pixel/RGB batches
match. Both branches use the same LR change at32k and deterministic Torch
controls; custom CUDA hash gradients are not guaranteed bitwise deterministic.

| Native held-out gate | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Opacity only,32k | 20.1624 | .813827 | .227709 |
| Opacity+tight depth,32k | 20.1407 | .814202 | .227082 |
| Opacity only,36k | 20.0957 | .816466 | .222051 |

Results: the32k depth addition improves inner-face PSNR in all six review views:
held-out+.365/+.271/+.446dB and train+.149/+.795/+.033dB. Face LPIPS differences
are small and mixed. This is a provisional local gain, not proof of complete
artifact removal or faster convergence. The36k opacity-only gate was inspected
on all six face/hand/hair/clothing contacts against32k. Faces remain coherent;
softness, hand noise and background errors remain. No major foreground collapse
appears. Its PSNR is within.07dB of32k and LPIPS is lower, so the current selector
chooses36k. The matched36k depth branch is under supervision and is not accepted
before its native comparison.

- [Matched32k evidence](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity01_depth10_mass_032000/comparison.json)
- [Opacity32k→36k faces](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity01_036000_gate/face.jpg)
- [Opacity32k→36k hands](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity01_036000_gate/hand.jpg)

### Independent signed-head initialization: rejected smoke gate

What was tested: transfer the ordinary control30k hash encoding and appearance,
then fit a separate zero-initialized signed residual head using learned train-only
termination points. The cache has30,002 accepted points across162 train cameras;
only widths≤.005 enter the head fit. No source masks, held-out targets or MVS
depth are used. Appearance and encoding stay frozen during300 geometry steps.

Results: sampled surface absolute error falls from.1209 to.01458 scene units,
but the native initial render scores4.9193/.101012/.981914 PSNR/SSIM/LPIPS.
The face render is severely fragmented with spurious bright colors. This smoke
test fails the visual gate; it is not a viable opaque40k initialization and will
not be promoted merely because the geometry fitting error decreases.

Insights: copied appearance queried at inaccurate first hits does not preserve
ordinary volume quality. Geometry anchoring alone did not establish correct
first-hit topology. Further hard-surface work needs an explicit ray/first-hit
audit before a long RGB fit. Future initialization diagnostics default to
stride4 previews; primary quality comparisons remain native stride1.

- [Failed native initialization metrics](/home/pavlo/renders/luster_000470_opacity/surface_bootstrap_030000_smoke/initial_surface_metrics.json)
- [Failed face crop](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity01_036000_gate/bootstrap_face.jpg)

The actual first-hit audit uses1,024 fixed train-only learned teacher rays and
verifies regenerated origins/directions against the hashed cache. It proves
that anchor proximity did not establish correct first intersections. A separate
300-step screen adds positive-sign constraints before the teacher point. Its
stride4 face preview remains severely fragmented and fails the gate. Different
preview resolutions are not compared as primary quality metrics.

| Initialization diagnostic | Anchor absolute SDF median | First-hit absolute error median | First hit within.005 | Negative before teacher |
| --- | ---: | ---: | ---: | ---: |
| Original300 steps | .010338 | .389032 | .586% | 91.992% |
| Free-ray penalty1,300 steps | .009403 | .119760 | 10.645% | 20.703% |

These are learned-teacher diagnostics, not measured-depth certification or
held-out validation. The new penalty changes anchor sampling through its extra
random draw in this first screen; subsequent implementation isolates that RNG.
These failed visual results do not justify a long opaque RGB fit.

- [Original first-hit audit](/home/pavlo/renders/luster_000470_opacity/reviews/surface_bootstrap_smoke_first_hit_audit.json)
- [Free-ray first-hit audit](/home/pavlo/renders/luster_000470_opacity/reviews/surface_bootstrap_free300_first_hit_audit.json)
- [Failed free-ray preview](/home/pavlo/renders/luster_000470_opacity/reviews/surface_bootstrap_030000_free300_face_preview.jpg)

### Tight depth36k: narrower surface, foreground tradeoff

What was tested: continue the matched32k pair to36k with unchanged opacity.1,
native sampling, calibrated near.05, and either no depth or interval-mass
depth strength10/coefficient.0001. The verifier passes common ancestry,
identical first32 RGB/pixel batches, calibration, native GT and ray bounds.

| Native36k branch | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Opacity.1 only | 20.0957 | .816466 | .222051 |
| Opacity.1+tight depth | 20.1766 | .815336 | .223967 |

Results: all six face/hand/hair/clothing contacts were inspected. No major
collapse occurs, but face LPIPS worsens in every view: held-out+.01152/+.00200/
+.00205, train+.01285/+.00388/+.00070. Held-out inner-face PSNR changes
+.051/−.161/−.369dB. The earlier uniformly positive32k inner-face result does
not persist. Hand150 improves2.812dB and.01085 LPIPS. Global PSNR gain.081dB
therefore does not establish a better face or an artifact-free scene.

On the same trusted native train-face rays, median90% widths shrink from
.002930/.002740/.002673 to.002405/.002222/.002206 (18–19%). Median relative
mode errors also decrease. This is geometric improvement with an appearance
tradeoff; it is conditional on the trusted train samples. Foreground softness
and wrist artifacts remain. The strong-depth40k request is held. The next
same-parent screen from opacity-only36k reduces only its depth coefficient
from.0001 to.00003, keeping strength10 and the uncertainty interval unchanged.
An ordinary no-prior baseline36k control is also running under supervision.
No convergence-speed claim is made; diagnostic GPU work overlaps these runs.

- [Verified36k comparison and visual verdict](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity01_depth10_mass_036000/comparison.json)
- [Opacity-only termination](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity36k_alignment.json)
- [Tight-depth termination](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity_depth36k_alignment.json)

### Ordinary baseline36k endpoint verification

What was tested: continue the ordinary LookCloser model/manager/parser from
control30k to36k, retaining near.05 in train/eval and adding neither depth nor
opacity regularization. Compare native endpoints to opacity.1 from their
common28k origin. Stage boundaries differ; recorded32-batch samples at28k and
30k match, but this is not a whole-trajectory bitwise replay.

| Native36k endpoint | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Ordinary, no priors | 19.8106 | .808639 | .228245 |
| Opacity.1, no depth | 20.0957 | .816466 | .222051 |

Results: native GT is identical in all six views and all four region contacts
were inspected. Both endpoints preserve a coherent actor. Opacity.1 reduces
horizontal facial defects in012/011 and improves face LPIPS in every review
view; hand differences are mixed and wrist noise remains. The endpoint gain
is.285dB PSNR with lower LPIPS, not proof of an artifact-free recipe. Both
control40k and opacity40k requests are prepared; opacity40k is running under
supervision, followed by the same-parent weaker-depth screen. No source masks
are used. The independent hard-surface and semantic arms remain unfinished.

- [Endpoint metrics, batch evidence and visual review](/home/pavlo/renders/luster_000470_opacity/reviews/near05_baseline_opacity_036000/comparison.json)
- [Ordinary/opacity face contact](/home/pavlo/renders/luster_000470_opacity/reviews/near05_baseline_opacity_036000/face.jpg)

Pinned comparisons are copied to the coder-readable private delivery root:
`baseline_opacity_36k` (ordinary vs opacity) and `incremental_near05_36k`
(opacity vs stronger depth). Files are physical copies, not links into the
private home. The exporter now supports explicit native sampling so camera
paths can retain the quality evaluator's integration settings. Denser export
remains the default and must be labeled separately.

### Matched40k weaker-depth gate

What was tested: from the exact opacity36k parent, continue4k steps with
opacity.1 only or add depth strength10/coefficient.00003 (effective.0003).
Native sampling, near bounds, restored optimizer state and first32 pixel/RGB
batches match. Optimization runs sequentially without other GPU work.

| Pinned native40k | PSNR | SSIM | LPIPS | Optimization seconds |
| --- | ---: | ---: | ---: | ---: |
| Opacity.1 only | 20.0879 | .817860 | .221339 | 277.51 |
| Opacity.1+weaker tight depth | 20.0737 | .817637 | .221319 | 321.16 |

Results: all six views and four region contacts were inspected. Neither
collapses; neither removes wrist noise or face softness. Held-out inner-face
PSNR changes+.044/+.051/−.035dB. Held-out face LPIPS changes+.00050/+.00026/
+.00010; train differences are tiny and mixed. No useful gain is established.
Depth adds15.73% optimization time for the same4k steps. Evaluation overlaps
later geometry diagnostics, so its duration is not a clean speed comparison.
These results do not establish faster time-to-quality with a prior.

Both histories select opacity-only36k under the declared PSNR/LPIPS criterion.
The opacity40k PSNR is.07448dB below the branch maximum32k, just outside the
.07dB tie window; lower40k LPIPS does not override that rule. Opacity-only36k
remains the accepted recipe. Ordinary baseline40k is now running under
supervision. The weaker-depth addition is not promoted.

- [Verified40k paired comparison](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity01_depth10_mass_weak_040000/comparison.json)
- [Opacity36k→40k visual gate](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity01_040000_gate/visual_review.json)

### Signed initialization appearance-transfer correction

The earlier surface bootstraps copied the frequency-grid tensors but omitted
the non-buffer feature reweighting strength. The trained teacher uses.3;
the new step0 field incorrectly used1.0. That confounds the earlier RGB
failure attribution, although the measured wrong first-hit depths remain
valid because independent geometry uses raw features.

The correction transfers this setting explicitly and verifies the same
1,024 teacher-point colors before geometry fitting. A corrected300-step,
free-ray-weight1 diagnostic records exact RGB parity (maximum difference0),
yet its preview remains severely fragmented. Increasing only the fit duration
to2,000 steps reduces sampled surface absolute error to.007126 scene units,
but its stride4 preview also fails (4.6641/.042957/.977823). These low-resolution
diagnostic metrics are not compared with native volume gates. No opaque40k
fit is accepted from these failed initializations.

- [Corrected300-step receipt](/home/pavlo/renders/luster_000470_opacity/surface_bootstrap_030000_free300_fr03/geometry_receipt.json)
- [Corrected2000-step receipt](/home/pavlo/renders/luster_000470_opacity/surface_bootstrap_030000_free2000_fr03/geometry_receipt.json)

### Final ordinary40k endpoint and intermediate delivery

What was tested: complete the ordinary no-prior40k recipe, compare it to
opacity.1 at the same endpoint, then export the accepted opacity branch's
selected checkpoint using native integration. The comparator verifies the
explicit common28k checkpoint ancestry, identical first32 pixel/RGB batches
at36k, calibration, GT and near/sampling controls.

| Pinned native40k | PSNR | SSIM | LPIPS | Last4k optimization seconds |
| --- | ---: | ---: | ---: | ---: |
| Ordinary, no priors | 19.7590 | .809926 | .226724 | 248.94 |
| Opacity.1 | 20.0879 | .817860 | .221339 | 277.51 |
| Opacity.1+weak tight depth | 20.0737 | .817637 | .221319 | 321.16 |

Results: all six face/hand/hair/clothing contacts were inspected. Opacity-only
improves overall PSNR.329dB and LPIPS.00538 at40k. Face LPIPS improves in five
of six views, but differences are small: ordinary40k partly catches up to the
facial artifact reduction seen with opacity at36k. Hand differences remain
mixed. These are single-seed endpoint results, not proof of faster convergence
or complete artifact removal. The measured4k intervals run without competing
GPU work; optimization excludes native evaluation. Ordinary history selects
30k, opacity history selects36k under the declared.07dB rule.

The accepted opacity36k export keeps coarse.00625/cap1024/legacy allocator and
near.05. All three native eval PNGs match selected renders byte-for-byte by
SHA-256; metrics also agree. It contains32 face frames at512×512 and32 body
orbit frames at480×640. Both H.264 videos decode to exactly32 frames. The
initial PATH ffmpeg failed because of missing x264; the working installed JPEG
environment binary assembled the saved face frames and completed the orbit.
No rendering settings or model weights changed during recovery.

Visual review of all64 contacts and the native face zoom finds a coherent
actor with remaining face softness, wrist noise, studio errors and pale skirt
ghosts at several orbit angles. The delivery manifest explicitly records that
this is an intermediate volume result, not an artifact-free hard surface.
All126 delivered files are physical copies with matching hashes under
`/var/tmp/lookcloser_000470_delivery/accepted_opacity36k_native`.

The corrected2000-step signed-head audit still has first-hit median absolute
error.036739 and p90.164029 against1,024 learned teacher rays; only26.465% hit
within.005. These are normalized scene units. The earlier audit key containing
`5mm` was misnamed; the numeric tolerance was.005 normalized units and is
renamed for future audits. Signed initialization remains rejected.

Next: a single2k opacity.3 screen from the exact accepted36k parent is prepared
but unlaunched, together with the saved opacity.1 step38k control selection for
native reevaluation. Inspect38k before extending it. Hard opaque40k, semantic
permissions/quality training, improved depth/geometry and repeat robustness
remain incomplete. All training and export processes are terminal at this
review gate; the full original goal remains active.

- [Verified ordinary/opacity40k comparison](/home/pavlo/renders/luster_000470_opacity/reviews/near05_baseline_opacity_040000/comparison.json)
- [Selected native export manifest](/home/pavlo/renders/luster_000470_opacity/exports/accepted_opacity36k_native/manifest.json)
- [Face path](/home/pavlo/renders/luster_000470_opacity/exports/accepted_opacity36k_native/face/contact.jpg)
- [Body path](/home/pavlo/renders/luster_000470_opacity/exports/accepted_opacity36k_native/orbit/contact.jpg)
- [Corrected2000-step first-hit audit](/home/pavlo/renders/luster_000470_opacity/reviews/surface_bootstrap_free2000_fr03_first_hit_audit.json)

```bash
scp coder@olive-lark:/var/tmp/lookcloser_000470_delivery/accepted_opacity36k_native/eval_img_0001.png ~/Downloads/
scp coder@olive-lark:/var/tmp/lookcloser_000470_delivery/accepted_opacity36k_native/face/learned_rgb.mp4 ~/Downloads/
```

### Additional38k evaluation and opacity.3 screen

What was tested: reevaluate the saved opacity.1 step38k using native integration,
then train opacity.3 for2k steps from the same36k parent. Only the coefficient
changes. Native GT, calibration, ray bounds and first32 pixel/RGB batches match.

| Native38k | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Opacity.1 control | 20.1719 | .816426 | .222436 |
| Opacity.3 | 20.1921 | .817917 | .220531 |

Results: all six face/hand/hair/clothing contacts were inspected. Global gains
are small. Inner-face PSNR changes+.008/+.256/+.157dB held-out, but−1.661/
−.408/+.013dB in train review. Face LPIPS differences are small and mixed.
The stronger coefficient is held rather than accepted from global scores alone.
Repeat robustness is unproven. No source masks are used.

The new opacity-only38k evaluation changes that branch's selection:38k is now
the maximum and36k lies.07624dB below it, outside the.07dB tie window. The older
36k export remains a historical intermediate artifact. A new selection/history
receipt preserves all evaluations without rewriting the original run history.

- [Verified opacity.3 screen](/home/pavlo/renders/luster_000470_opacity/reviews/near05_opacity03_038000/comparison.json)
- [Updated opacity-only selection](/home/pavlo/renders/luster_000470_opacity/selections/opacity01_with_38k_gate.json)

### Hard density boundaries: geometry/color diagnostics, not accepted models

What was tested: use the trained opacity38k field directly, bracket the first
density-level boundary with its native ARM intervals, bisect12 times, and render
one point with alpha1. Thresholds100/300/1000 are read-only counterfactuals,
with no MVS targets or masks. Empty rays stay black. The method can miss narrow
peaks and density/frequency-cell discontinuities; it is not certified geometry.

Results: all six face crops were visually inspected. Level1000 preserves a
recognizable face but has noise and holes; lower levels have severe color errors.
Face PSNR for level1000 is14.55/11.34/21.42 held-out and9.02/12.56/20.18 train,
versus volume24.88/28.50/28.18 and26.71/30.80/32.44. Hit fractions are81.6–92.5%;
95–100% of found roots meet the1% density residual criterion depending on level.
These are face-only diagnostics, not full-frame quality metrics or trained40k
opaque results.

Two1,000-step RGB adaptations keep the geometry teacher/frequency grid frozen:
fresh Adam.001 on the color MLP, or on a separate complete color field. The first
attempt failed at its first step because target RGB remained on CPU; the fixed
run moves targets to the renderer device. Both completed runs verify unchanged
geometry hashes. All RGB targets remain in the objective; no source mask excludes
empty rays. The full color field's density output is unused by the hard renderer.

| Central-half face,8bit PSNR | Before | RGB head only | Independent complete RGB field |
| --- | ---: | ---: | ---: |
| 012 | 24.16 | 25.72 | 26.35 |
| 095 | 26.20 | 27.64 | 27.77 |
| 150 | 24.88 | 27.52 | 27.62 |
| 011 | 24.36 | 25.77 | 25.76 |
| 097 | 26.12 | 27.71 | 27.91 |
| 151 | 26.29 | 27.31 | 27.85 |

Insights: copied volume color is not already the correct opaque surface color;
adaptation helps local interiors but remains worse than the ordinary volume.
Holes, grain and bad background coverage persist, so neither model is promoted.
Batch equivalence across these diagnostic fits is unproven; they do not establish
an isolated causal gain from the full encoding. Learned geometry also needs repair.

- [Density-level renders/receipts](/home/pavlo/renders/luster_000470_opacity/reviews/density_surface_opacity38k/manifest.json)
- [RGB head diagnostic](/home/pavlo/renders/luster_000470_opacity/density_rgb_adaptation_1000_38k_v2/manifest.json)
- [Independent color-field diagnostic](/home/pavlo/renders/luster_000470_opacity/density_rgb_fullfield_1000_38k/manifest.json)
- [Fixed interior evidence](/home/pavlo/renders/luster_000470_opacity/reviews/density_rgb_adaptation_inner_face.json)

### Independent density branch: parity passed, quality gate pending

What was tested: separate density hash/MLP parameters from RGB features while
keeping the exact initial field. The converter clones171,739,264 encoding values,
3,072 density-MLP values and their Adam moments, preserving the original four
parameter slots (including empty SH), scheduler, scaler and RNG.

Results:38 focused tests pass. On the real checkpoint,512 point queries have
exactly zero density/RGB difference, and256 native camera rays have zero RGB,
depth and accumulation difference. Density gradients reach only the two new
geometry modules; fixed-point RGB gradients reach only the original color branch.
The checkpoint loader confirms full optimizer/scheduler/scaler/RNG restoration.

A matched38k→40k repeat of the shared control is running first, followed by the
split branch with no new depth prior. Both use the same source state, opacity.1,
native sampler and near.05. This architecture is not accepted before native
quality review. Only after that gate will a stronger geometry prior be compared.

- [Actual native parity and gradient audit](/home/pavlo/renders/luster_000470_opacity/reviews/density_clone38k_parity.json)
- [Checkpoint conversion receipt](/home/pavlo/renders/luster_000470_opacity/split_density_initial38k/conversion.json)

### Independent density40k normal-quality gate

What was tested: two matched2k continuations from the same38k state. The shared
control retains four field parameters; the split branch adds two density-only
clones with copied moments. Native sampling, calibration, near.05, opacity.1,
optimizer/scheduler/scaler/RNG and first32 pixel/RGB batches match. No new depth
prior is enabled in either arm.

| Native40k | PSNR | SSIM | LPIPS | Optimization seconds |
| --- | ---: | ---: | ---: | ---: |
| Shared repeat | 20.0676 | .817846 | .221743 | 140.61 |
| Independent density | 20.0532 | .817861 | .221826 | 207.27 |

Results: all six face/hand/hair/clothing contacts were inspected. The split
branch preserves ordinary coherent foreground quality, without a large collapse.
Small residual noise, softness and studio errors remain. Differences are tiny;
the architecture alone does not improve the accepted recipe. Optimization is
47.4% slower. A repeat of the shared continuation scores.0202dB below the earlier
shared40k result, a warning against interpreting similarly small gains as robust.
Separate legacy field/HDR/JIT/default-recipe checks add31 passing tests.

Next: a same-parent split40k→42k pair adds only the previously tested interval-mass
depth strength10/coefficient.0001 to one arm. Native sampling stays unchanged,
and unknown depth stays unconstrained. The no-new-prior control is evaluating;
the depth arm launches after it finishes. This tests whether the prior is less
harmful when its direct gradients cannot alter color features.

- [Verified split quality gate](/home/pavlo/renders/luster_000470_opacity/reviews/density_split_control_040000/comparison.json)

### Independent density plus tight prior42k gate

What was tested: same-parent split40k→42k pair, opacity.1, either no depth or
interval-mass strength10/coefficient.0001. Native RGB sampling, calibration,
near bounds and first32 pixel/RGB batches match. Unknown depth is unconstrained.

| Native42k | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Split control | 20.0875 | .817927 | .221598 |
| Split+tight depth | 20.0985 | .818172 | .221013 |

Results: all six views and four region contacts were inspected. Inner-face
PSNR improves+.152/+.394/+.256dB held-out and+.097/+.198/+.291dB train. Face
LPIPS worsens slightly in all six views (+.00011–.00071). There is no major
foreground collapse, but noise/softness and room errors remain. Global gain
.011dB is too small to establish robust superiority. Optimization timing is
confounded by the overlapping38k export and is not used for a speed claim.

Actual pinned42k checkpoints, not their selected ancestor, are probed on the
same trusted native20×20 train-face grids. Reference front-mass tolerance stays
max(2uncertainty,1% depth); training uses strength10 and the uncertainty floor.

| Train face | Control90% width | Prior90% width | Control front mass | Prior front mass |
| --- | ---: | ---: | ---: | ---: |
| 011 | .002954 | .002747 | 2.3231% | 2.2373% |
| 097 | .002601 | .002505 | .01555% | .01482% |
| 151 | .002585 | .002405 | .00500% | .00591% |

Widths shrink4–7% on these conditional trusted samples; relative mode errors
also decrease. This does not certify all scene geometry or artifact removal.
A further matched2k continuation to44k is prepared but unlaunched to check
whether the local benefit persists, before changing the prior formulation.
All current jobs are terminal at this review gate.

The newly selected opacity-only38k export is visually reviewed on all32 face
and32 body frames, matches original eval PNG hashes, and both videos contain
exactly32 decoded frames. Physical delivery copies also match all hashes.
The default field compatibility checks pass31 additional tests. The user's
existing README edit remains untouched.

- [Verified42k comparison](/home/pavlo/renders/luster_000470_opacity/reviews/density_split_depth10_042000/comparison.json)
- [Pinned control geometry](/home/pavlo/renders/luster_000470_opacity/reviews/density_split_control42k_alignment.json)
- [Pinned depth geometry](/home/pavlo/renders/luster_000470_opacity/reviews/density_split_depth42k_alignment.json)
- [Accepted38k export](/home/pavlo/renders/luster_000470_opacity/exports/accepted_opacity38k_native/manifest.json)

```bash
scp coder@olive-lark:/var/tmp/lookcloser_000470_delivery/accepted_opacity38k_native/eval_img_0001.png ~/Downloads/
scp coder@olive-lark:/var/tmp/lookcloser_000470_delivery/accepted_opacity38k_native/face/learned_rgb.mp4 ~/Downloads/
scp coder@olive-lark:/var/tmp/lookcloser_000470_delivery/independent_density42k/face_comparison.jpg ~/Downloads/
```
