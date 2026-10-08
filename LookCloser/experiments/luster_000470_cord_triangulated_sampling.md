# Frame 000470: triangulated cord and local ray sampling

## What was tested

The held-out cam095 shows red belt fabric where a teal hanging braid should
occlude it. The ordinary baseline's native ray at `(883,1340)` terminates
mostly on red fabric. PatchMatch does not give a continuous qualified depth
on this thin cord, so no PatchMatch cord depth was forced into training.

Two sparse physical landmarks were marked directly on original, undistorted
JPGs: the waist junction in eight **training** cameras and the first bead in
five. These are isolated keypoints, not image masks. The original COLMAP
PINHOLE cameras triangulate both points; the transform from their world frame
to the actual training frame is recovered from all camera centers with maximum
residual below `2e-7`. The keypoints and fitter are in
[`cord_000470_landmarks.json`](cord_000470_landmarks.json) and
[`triangulate_luster_cord_landmarks.py`](../scripts/triangulate_luster_cord_landmarks.py).
The [fit receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/landmark_audit_v2/fit.json)
records source/calibration hashes, each residual, camera omission checks and
annotation perturbation checks.

The largest original-JPG reprojection error is `8.73 px` at the junction and
`6.48 px` at the bead. The straight 3D segment projects onto the braid in
the [held-out cam095 original photo](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/landmark_audit_v2/cam095_segment_projection.png),
although cam095 supplied no fit observation. At the missing pixel, the
camera ray passes `0.00022` normalized scene units from the segment at ray
distance `1.04873`; the baseline's dominant red layer lies at `1.05357`.
With independent Gaussian annotation perturbations (5 original pixels for
the junction and 3 for the bead), the 5th–95th percentile inferred cord
distance is `1.04762–1.04964`. This validates an approximate visible cord
surface there, not its full width or every braid twist.

The first intervention changes **sampling only**. For rays within `.002`
normalized units of the 3D segment, the experimental opacity model replaces
the ordinary coarse traversal with a finer `0.001` traversal, maximum 4096
samples. All other rays retain their ordinary samples. The fit is pinned by
hash and checked against the train-only scene calibration. No parameters,
source RGB images, source masks, opacity loss or depth loss change in the
frozen-field screen. A separate RNG stream isolates selected-ray sampling.

## Results

On the frozen no-prior 30k baseline, the ordinary cam095 ray at `(883,1340)`
has zero termination mass within `±.002` of the inferred cord distance and
renders approximately RGB `(189,64,83)` versus teal GT `(74,118,127)`.
Sampling that same frozen ray at step `.001` gives the cord interval mass
`.756` and renders approximately `(92,114,119)`. The nearest field sample is
teal; thus the field already contains some cord geometry and color that the
coarse traversal misses. At `.0005`, the mass is `.788` on that ray.

| Frozen 30k renderer | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | cam097 cord PSNR |
|---|---:|---:|---:|---:|---:|
| Ordinary | 19.8388 | .807596 | .231483 | 20.7015 | 26.0630 |
| Local sampling, radius .002 | 19.8396 | .807628 | .231464 | 21.3678 | 26.3849 |

Full-scene quality is essentially unchanged. The [frozen cam095 comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/frozen_eval_cord_early.png)
shows a stronger teal braid, but its lower part remains too faint. A frozen
radius screen from `.002` to `.005` improves cam095 crop PSNR monotonically
from `21.37` to `21.67 dB`; the [wider-radius comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/radius_screen_v1/cam095_comparison.png)
still shows the break. The conservative `.002` radius was chosen for the
training comparison.

Both training runs resume the same ordinary no-prior 30k checkpoint, with
identical seed, RGB data, and 30.5k/31k gates. The control uses ordinary
sampling; the candidate adds only local cord sampling. First 32 actual
pixel/RGB batch hashes match exactly. Neither run uses source masks or a
depth/opacity objective.

| Training sampler | Step | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | cam097 cord PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary control | 30500 | 19.8614 | .806698 | .230085 | 19.4653 | 26.1256 |
| Local cord | 30500 | 19.8579 | .806497 | .230433 | 20.0684 | 26.4811 |
| Ordinary control | 31000 | 19.8459 | .806976 | .229897 | 19.1775 | 25.9653 |
| Local cord | 31000 | 19.8305 | .806558 | .229773 | 19.6143 | 26.2664 |

The standard best-checkpoint rule picks 31k for both. Local sampling raises
the held-out cord crop `0.44 dB` at 31k and improves its LPIPS `.108 → .091`,
while global PSNR differs by `−.015 dB`. Face and hand crop PSNRs on cam095
change by less than `.05 dB`. Yet the
[paired 31k cord render](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/paired_31000_cam095_cord.png)
still has a visibly thin/fading lower braid. This sampler is a useful
diagnosis and modest improvement, **not** an accepted fix. Optimization for
the first 1000 continuation steps took `66.42 s` versus `61.86 s` for the
control, about 7% slower; no speed advantage is claimed. Both runs ended
without OOM.

## Insights

The localized depth hypothesis is stronger than PatchMatch on this cord:
the directly projected segment has no measured PatchMatch depth on all 36
sampled positions in original sharp cams025–027. Where PatchMatch does
measure the upper cord in cams093–098, witness counts are sparse and most
lower positions remain unmeasured. A dense frozen-field probe of 88 valid
training rays projected along the segment gives median band opacity `.947`
within `±.003` of the line and **all 88** below `.99`. Thus a targeted
near-opaque depth-band objective has nonzero signal on the actual cord,
unlike a broad prior on unqualified PatchMatch pixels. The next controlled
test is to combine the local sampler with a small, train-only, sparse
cord-ray opacity prior. It must be screened against the same local-sampling
control and rejected if it thickens fabric or degrades the person's face.

## Sparse cord-ray and occupancy follow-ups

A controlled continuation added 32 auxiliary rays per step through the
triangulated segment, drawn only from cams025–027, 097–098, where **both**
landmarks were independently marked. A `±.003` depth band was encouraged to
opacity `.99`, with six probes in front. The coefficient was `.01`; all source
RGB images, pixel batches, model weights at 30k and local sampling controls
were identical to the preceding local-sampling run. All first 32 pixel/RGB
batch receipts match. This is a sparse train-only geometric prior, not a
source mask. A read-only 88-ray survey measured median baseline band opacity
`.946`, below the target `.99`.

| 31k continuation | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | cam097 cord PSNR |
|---|---:|---:|---:|---:|---:|
| Local sampling only | 19.8305 | .806558 | .229773 | 19.6143 | 26.2664 |
| Local sampling + cord-ray prior `.01` | 19.8300 | .80731 | .22936 | 19.4352 | 26.0751 |

The band prior does not repair the visibly fading lower braid and lowers both
cord crop PSNRs. Its [31k cam095 crop](/home/pavlo/renders/luster_000470_opacity/runs/cord_local_plus_ray001_031000/eval_031000/eval_001_cord.png)
is saved; this coefficient and objective are not accepted as a fix.

A separate forward-only experiment admitted occupancy cells within `.003`
normalized scene units of the fitted segment. At the no-prior 30k checkpoint,
the centerline passed four cells, one unoccupied; the tube passed 11 cells,
four unoccupied. The missing centerline cell occurs near segment fraction
`.78` and projects to cam095 pixel `(887,1368)`, in the lower fading braid.
On the frozen local-sampling 31k field, admitting those four
cells left the density/color fingerprint unchanged. Global eval metrics
changed from `19.83048/.806558/.229773` to
`19.83054/.806589/.229760` (PSNR/SSIM/LPIPS). Cam095 cord PSNR changed
`19.6143→19.5887`, SSIM `.7232→.7293`, LPIPS `.09094→.08695`;
cam097 cord PSNR `26.2664→26.3394`, SSIM `.8112→.8137`, LPIPS
`.07497→.07349`. Cam095 face PSNR was identical. The
[frozen crop comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/occupancy_frozen_local31k_r003/cam095_cord_comparison.png)
still shows the lower cord fading.

A second 30k→31k continuation changed only that `.003` occupancy admission,
retaining the same local sampler and no geometric or opacity loss. All first
32 pixel/RGB batches match the local-sampling control. The supervised run
finished without OOM; its first 1000 optimization steps took `66.90 s`
versus `66.42 s` for the local-sampling control.

| Training variant | Step | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | cam097 cord PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Local sampler only | 30500 | 19.8579 | .806497 | .230433 | 20.0684 | 26.4811 |
| Local sampler + admitted cord cells | 30500 | 19.8678 | .806405 | .230291 | 20.0806 | 26.5526 |
| Local sampler only | 31000 | 19.8305 | .806558 | .229773 | 19.6143 | 26.2664 |
| Local sampler + admitted cord cells | 31000 | 19.8139 | .806287 | .229863 | 19.4466 | 26.3778 |

The standard global selection rule picks 31k for both. On held-out cam095,
admission improves cord SSIM `.7232→.7268` but lowers cord PSNR
`19.6143→19.4466`; its LPIPS is nearly unchanged `.09094→.09019`.
Cam097 cord PSNR rises `.11 dB`, and cam095 face PSNR changes by only
`.011 dB`. The [four-way GT/control/frozen/trained crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/occupancy_31k_cam095_comparison.png)
shows a slightly denser middle braid, while the lower braid still fades
strongly. Thus missing occupancy contributed locally but does not explain
the whole artifact; keeping these cells occupied is safe and cheap, but
is not yet a complete recipe.
