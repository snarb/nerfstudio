# Frame 000470: excluding training cameras with an unclear belt cord

## What was tested

The original JPG for cam023 has a visibly blurred hanging cord, while cam025
resolves the braid. This is source-image evidence, not an inferred rendering
artifact. The COLMAP PINHOLE estimates have similar horizontal focal lengths
(4734.94 and 4759.24 pixels), so focal length alone does not explain their
sharpness difference. Focal length does vary substantially across the whole
rig. Cam099 has only the horizontal waist cord inside its 4096×3000 original
frame; the hanging part is below its bottom boundary. The earlier tight crop
was misleading. [Native-pixel cam023/cam025 comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_camera_quality_20261008/source_cam023_vs_cam025_native.png)
and [wide original cam099 crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_camera_quality_20261008/cam099_wide_original.png).

We manually reviewed all 165 original-source camera crops and selected 28 of
162 training cameras with a sufficiently visible and resolved cord:
`005–011, 021, 025–027, 036–044, 091, 093–094, 096–098, 156, 163`.
Eval cameras 012/095/150 stayed held out. The [selection manifest](/home/pavlo/renders/luster_000470_opacity/reviews/cord_camera_quality_20261008/camera_selection.json)
records each source hash and reason. This is a conservative **whole-camera**
filter, so other good content in excluded views is also removed. No source
masks, depth priors, or opacity regularizer were used.

An opt-in datamanager allowlist filters the cached training image batch before
frequency-aware pixel sampling, preserving original dataset indices for ray
generation and frequency buckets. Source files and eval data are unchanged.
Twelve focused tests cover shuffled/noncontiguous indices, target RGB identity,
frequency-aware strengths 0/.5/1, and invalid or held-out IDs. The first 32
actual training batches were audited to contain only the selected cameras.
The two from-scratch runs have identical recorded source-file hashes and
share the same seed, model settings, near plane, and eval views.

Two evaluations isolate initialization effects:

1. Continue the [known-good no-prior 30k baseline](luster_000470_opacity_depth.md) to 32k, with all 162 cameras or
   only the sharp 28 cameras for those 2k continuation steps. Seed, geometry,
   network, and eval protocol are matched. The field previously saw all
   cameras, so this tests a short camera-selection intervention.
2. Train from scratch with the same no-prior incremental-standard recipe,
   using all 162 cameras or only the sharp 28. Assess early at 2k and 4k;
   stop the poor variant before 8k.

## Results

### Continuation from the shared 30k field

| Sampling | Step | Eval PSNR | SSIM | LPIPS | Held-out cam095 cord PSNR | Train cam097 cord PSNR |
|---|---:|---:|---:|---:|---:|---:|
| All 162 | 30500 | 19.8614 | .80670 | .23009 | 19.4653 | 26.1256 |
| Sharp 28 | 30500 | 19.9794 | .81207 | .22507 | 20.7836 | 26.2522 |
| All 162 | 31000 | 19.8459 | .80698 | .22990 | 19.1775 | 25.9653 |
| Sharp 28 | 31000 | 19.8570 | .81331 | .22405 | 20.4180 | 26.3647 |
| All 162 | 32000 | 19.8918 | .80751 | .23080 | 18.4207 | 26.0106 |
| Sharp 28 | 32000 | 19.7965 | .81496 | .22276 | 21.5011 | 26.5670 |

The documented selection rule (maximum global eval PSNR, LPIPS tie-break
within .07 dB) picks step 31000 for all-162 and 30500 for sharp-28. The
32k selected-camera render has a substantially better cord-crop score, but
the lower hanging braid still fades/disappears against the red belt and teal
skirt. Thus improved crop PSNR does **not** establish that transparency is
fixed. Face crops show no similarly clear person-quality improvement.
[Held-out cord comparison across gates](/home/pavlo/renders/luster_000470_opacity/reviews/cord_camera_quality_20261008/compare_eval_001_cord.png),
[training-view cord comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_camera_quality_20261008/compare_train_094_cord.png),
[held-out face comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_camera_quality_20261008/compare_eval_001_face.png).

### From-scratch early gates

| Sampling | Step | Eval PSNR | SSIM | LPIPS | Held-out cam095 cord PSNR |
|---|---:|---:|---:|---:|---:|
| All 162 | 2000 | 19.5252 | .7158 | .4105 | 17.2475 |
| Sharp 28 | 2000 | 8.9873 | .4917 | .8479 | 11.4934 |
| All 162 | 4000 | 16.3244 | .6915 | .4431 | 18.1828 |
| Sharp 28 | 4000 | 8.8425 | .4749 | .8494 | 11.5647 |

At 2k, the all-camera model reconstructs approximate belt colors/shape;
the sharp-28 model is nearly uniform gray on the same held-out cam095 patch.
[Native crop comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_camera_quality_20261008/fromscratch_002000_cam095_cord.png).
[Matched 4k crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_camera_quality_20261008/fromscratch_004000_cam095_cord.png).
The filtered run was intentionally stopped immediately after the 4k visual
gate because its 2k-to-4k results failed to improve. Its 8k target was not
reached; no later convergence is claimed. Global PSNR of the matched all-camera
control also falls between 2k and 4k, mainly on cam095's complete image;
the **cord** crop improves from 17.25 to 18.18 dB. These early gates do not
replace the known-good no-prior 30k baseline.

## Insights

Removing blurred and nonvisible whole-camera views removes too much angular
coverage and breaks early generalization. Short continuation of a pretrained
all-camera field can improve cord-crop metrics, but does not restore a solid,
continuous cord. The existing ray audit separately shows that an opaque red
belt layer is visible at the missing-cord pixel where the teal braid should
occlude it. Camera-image quality is therefore relevant, but this hard filter
is not a viable stand-alone cord repair. Preserve broad-view baseline
coverage while testing a more local geometry/visibility constraint.
