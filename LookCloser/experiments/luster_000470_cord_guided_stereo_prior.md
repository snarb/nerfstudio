# Frame 000470: guided stereo depth on the lower hanging cord

## What was tested

The two-landmark straight cord segment is uncertain near the lower braid. A
read-only, train-only stereo audit projected 12 fractions of that segment into
the **original 4096px source JPGs** and scanned depth along cam097 rays. It
compared grayscale patches in cams025/026/027/098 at patch sizes 5, 9, and
13, recording per-camera correlations, separated peaks, and leave-one-camera-
out estimates. All five cameras and the landmark fit are hash-pinned training
sources; no dataset masks or held-out RGB entered the audit or training.

The 9×9 median selected an offset of `−.00325` normalized scene units at
fraction `.691` (NCC `.793`, three target-camera witnesses) and `−.00075` at
fraction `.844` (NCC `.776`, three witnesses). These are photometric
hypotheses, not certified surface depths. In particular, leaving out cam026
changes the `.844` estimate to `−.00325`. An independent check of the
existing PatchMatch confidence maps finds **0 qualified pixels among 30**
direct line/candidate projections at fractions `.691`, `.767`, and `.844` in
the five source views; the largest witness count is two, below the existing
six-witness threshold. [Stereo audit and NCC curves](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/result.json),
[source-patch comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/lower_line_vs_stereo_train_patches.png),
and [PatchMatch cross-check](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/patchmatch_crosscheck.json)
preserve the evidence.

From the same native-4K, no-prior 32.5k checkpoint, two matched pairs
continued for 500 steps with the sharp-28 train-camera allowlist and local
fine sampler unchanged. Each pair used identical selected source cameras and
fraction indices, coefficient `.003`, and `±.003` termination band. Only the
auxiliary 3D target differed: fitted straight line versus stereo point. The
first pair selected both lower fractions (eight rays); the second isolated
only the more stable `.691` fraction (four rays). The first 32 ordinary
pixel/RGB batches match exactly within each pair. Evaluation used the same
three held-out cameras at native-image stride two and the fixed cord crops.
Both pairs finished without OOM; optimization took approximately 39 seconds
per arm, versus 33 seconds for the 500-step no-prior control.

## Results

| 33k continuation | Eval PSNR | SSIM | LPIPS | Held-out cam095 cord PSNR | SSIM | LPIPS | Train cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-prior 4K control | **19.70369** | .81483 | .23199 | **23.95539** | **.76829** | .10119 | **26.61852** | **.81212** | **.08408** |
| Two fractions, line targets | 19.69632 | .81509 | .23183 | 23.92386 | .76572 | .09938 | 26.32053 | .80523 | .08477 |
| Two fractions, stereo targets | 19.70796 | .81511 | .23088 | 23.66123 | .75226 | .10485 | 25.71826 | .79154 | .09305 |
| One fraction, line targets | 19.70749 | .81471 | .23165 | 23.92795 | .76678 | **.09794** | 26.42556 | .80793 | .08524 |
| One fraction, stereo target | 19.70400 | .81489 | .23170 | 23.58181 | .75074 | .10701 | 25.64549 | .78967 | .09764 |

The single-point stereo arm loses `.346 dB` PSNR and `.0160` SSIM on the
held-out cord versus its matched line arm; LPIPS rises `.0091`. The two-point
stereo arm loses `.263 dB`, `.0135` SSIM, and `.0055` LPIPS versus its line
arm. The [held-out cam095 one-point comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/fullres33000_lower_u8only_cam095_comparison.png)
and [train cam097 comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/fullres33000_lower_u8only_cam097_comparison.png)
were inspected: neither prior restores distinct fibres in the lower braid;
stereo visibly smears it more. The [two-point cam095
comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/fullres33000_lower_u8u10_cam095_comparison.png)
agrees. The [one-point line selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_lower_u8only_line_003_033000/selection.json),
[one-point stereo selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_lower_u8only_stereo_003_033000/selection.json),
[two-point line selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_lower_u8u10_line_003_033000/selection.json),
and [two-point stereo selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_lower_u8u10_stereo_003_033000/selection.json)
pin all checkpoints and metrics. Each run has a `supervision.jsonl` recording
worker progress, GPU memory, completion and no OOM.

## Insights

Patch correlation around this braided, view-dependent thin object is not
precise enough to impose a hard surface target. Even the point with a stable
leave-one-camera-out peak degrades both held-out and train cord views. The
small LPIPS gain of the four-ray **line** prior does not recover the visible
braid and comes with slightly worse cord PSNR/SSIM; it is not an accepted
recipe. Keep the no-prior 4K checkpoint as the cord baseline. Do not tighten
depth or opacity on this segment until independent cross-view surface
correspondence, or a better calibrated local model, can be established.
