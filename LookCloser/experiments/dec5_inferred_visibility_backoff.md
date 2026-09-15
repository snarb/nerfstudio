# DEC5: original-surface texture backoff after inferred completion

## What was tested

The [surface-sampling repair](dec5_mhr_surface_sampling.md) produced three new
black RGB pixels in the old moving view of frame001193. The target depths at
all three are **bit-identical** to baseline, while source IDs change from
40/39/11 to255 (no selected source). Their portrait coordinates are
(560,1030), (538,1070), and (724,1285), at hair, neck margin and clothing.
They are not new missing target-geometry intersections. Matched inputs and
missing source IDs point to changed source visibility from the inferred
appendage, even for existing target surfaces; every individual visibility
predicate has not been separately replayed in this experiment.

`recover_original_surface_texture.py` tests a narrow opt-in visibility fallback.
It requires exact original mesh vertex/triangle prefixes, identical camera,
source-camera order, frame, color profiles/exposure and rendering recipe. Both
input renders must have verified receipts and train-only provenance.

For each black candidate pixel with source255 and a colored baseline source,
fresh raycasts must hit the **same original triangle**, with matching depth
(absolute1e-7) and barycentrics (1e-6). Only then may the baseline's already
verified train reprojection be reused. The same physical surface point has the
same source projection under the unchanged calibration. No GT, RGB averaging,
new colors, geometry modification or new-surface filling occurs.

This changes how uncertain inferred occlusion affects old texture support. It
is not proof that ignoring that occlusion is physically correct in every scene;
inherited baseline texture errors remain possible. No production renderer
default or delivered video is modified.

## Results

Root: `/mnt/data/dec5_mhr_original_surface_texture_backoff`.

Exactly **three** pixels recover their prior RGB and source ID. New black RGB
relative to baseline falls from3 to0 in this matched single view. Every other
RGB/source pixel remains bit-identical to the raw corrected render. Newly
visible but untexturable geometry is unchanged, including the ten uncolored
new hits recorded in the prior experiment. The remaining F/E geometry hole is
not addressed by this renderer fallback.

`review_original_surface_texture_backoff.py` independently rechecks input/output
hashes, all frozen renderer-helper hashes, exact changed-pixel inventory and
RGB/source equality. Parent actually viewed all three native baseline / raw /
backoff crops: [hair](/mnt/data/dec5_mhr_original_surface_texture_backoff/review/pixel_0.png),
[neck](/mnt/data/dec5_mhr_original_surface_texture_backoff/review/pixel_1.png),
[clothing](/mnt/data/dec5_mhr_original_surface_texture_backoff/review/pixel_2.png).
The isolated black changes revert without a visible surrounding color change;
ragged hair/neck contours remain. Verdict: **verified narrow regression recovery,
not full-video acceptance**.

Two tests reject different/new triangles even at equal depth, different depth
or barycentrics, valid candidate RGB/source, missing baseline color and missing
baseline geometry. These are 1080×1920 diagnostic frames; the delivered native
3456×6144 video remains untouched. PSNR/SSIM/LPIPS are N/A: no independent GT
quality comparison is claimed.

## Insights

A locally plausible inferred patch can change source visibility away from the
target hole. Depth equality alone is insufficient for safe fallback: original
triangle identity, barycentrics and exact source/calibration provenance are also
required. This bounded recovery prevents losing existing texture support, but
does not validate inferred shape or eliminate holes on newly exposed surfaces.
