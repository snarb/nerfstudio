# Existing hybrid scores: faces versus clothing

## What was tested

This is arithmetic over the already saved1670-model sweep:16 shader variants
plus neural reference, with eight donor-heldout contexts in TRAIN cameras HC
and IC. No fit, render, metric inference or profile implementation was added.
Means below give each context equal weight, matching the original sweep;
PSNR is an arithmetic mean of per-context dB scores. All original eight-context
aggregates were reproduced to1e-12.

Six contexts contain faces/hair and two contain the same shirt crop. The
scale2 and4.2667 versions reuse the same underlying target crops with different
transfer/filter scales: they are not independent samples. The native-scale
subset has four distinct face crops (two actors, two cameras) and one shirt crop.

## Results

| Contexts / existing variant | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|
| All8: baseline detail σ8/gain1 |28.5542|.606737|.416033|
| All8: selected detail σ4/gain.75 |28.7757|.616199|.410062|
| Faces6: baseline σ8/1 |28.8840|.628642|.402888|
| Faces6: selected σ4/.75 |29.1219|.638836|.404622|
| Faces6: best existing LPIPS, detail σ4/1 |28.8957|.628751|.402169|
| Shirt2: baseline σ8/1 |27.5650|.541024|.455468|
| Shirt2: selected σ4/.75 |27.7369|.548289|.426384|
| Shirt2: best existing LPIPS, direct blend.5 |27.8481|.553707|.418119|

The selected profile improves shirt LPIPS by.0290844 (6.39%) while worsening
face-context LPIPS by.0017343 (.43%). Thus the measured overall improvement
in this eight-context sweep comes from clothing, with a small face tradeoff.

The correctly scaled native subset makes the face tradeoff more explicit:

| Native face4 / existing variant | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|
| Baseline detail σ8/1 |28.7411|.621706|.394705|
| Selected detail σ4/.75 |29.0483|.634753|.397915|
| Best existing: detail σ4/1 |28.7535|.621756|.393858|
| Next existing: multiband σ8/1 |28.7977|.622259|.394485|

Selected native-face LPIPS is.0032099 (.81%) worse than baseline in all four
contexts. Detail σ4/1 improves native-face LPIPS versus the selected profile
in all four, but has lower PSNR/SSIM. The larger LPIPS tradeoff is on the gray
actor: native-face mean baseline.305024 → selected.310666; σ4/1 gives.304344.
For the black actor the corresponding values are.484387 →.485164 and.483372.
These are context scores, not a video-wide or skin-only face assessment.

Full17-row rankings, per-context deltas and all eight source-score hashes are
saved at
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/video_final_control_binding_audit_20261108/existing_face_cloth_scores.json`
(SHA `bb7ed6da4f0eee4659973cc2f4b4c17615dd90ffbd2f8410554f43edd4af1581`).
The original source scores remain unchanged.

## Insights

The shared σ4/.75 profile is not the face-only LPIPS optimum in these samples.
Its face penalty is measurable but modest, and PSNR/SSIM favor it. One model,
two TRAIN views and four physical face crops do not establish a better
video-wide face profile or justify an unqualified spatial blend. The eight
contexts also do not establish a global optimum over the entire actors.

ROOT retained the approved shared profile for its overall measured tradeoff
and stopped further profile work. No shader, model, mask, grid or active hook
changed. The final48 frames are actual paused2020 photographs; neural hybrid
gain does not apply to that face-focused tail. Full72-frame temporal inspection
remains required.
