# Best eval delivery, 2026-11-01

## What was selected

Same checkpoint75512, screen_v11/world_sampling_fine875512, selected by lowest comparable mean person ROI LPIPS. Replays differ below1e-6 and are numerically equivalent. Native saved full images are copied without cropping or resampling.

## Results

|Camera|Person ROI LPIPS|Full-image PSNR|Full-image SSIM|Full-image LPIPS|
|---|---:|---:|---:|---:|
|DD|0.182839|32.37088|0.903829|0.104997|
|HC|0.217571|29.94358|0.877804|0.136645|
|MD|0.263227|27.09036|0.892024|0.125391|

Mean person LPIPS0.221212. Outputs and SHA256 provenance: /fsx/lookcloser_misha_maskfix_20261031_pavlo/best_eval_person_lpips_20261101/ . Full resolution1920x1013(DD,HC),1920x1012(MD). Contact sheet visually inspected. This remains the old numerical leader; known AB floaters and novel-view forearm notch mean it is not an accepted artifact-free result.

## Insights

Eval-camera metrics do not certify clean novel views. No newly accepted clean leader is claimed.

## Why step75512 was delivered

Verified original request: end_step75512 was a preset finite512-update pilot from migrated75000, not validation-loss early stopping. Person LPIPS improved75000:0.223101,75256:0.222103,75512:0.221212. The matched unchanged-training control screen_v12/fine_support_control76024 resumed75512 through76024 (request differences only inference probes/review cameras/reference bookkeeping):75768=0.223364,76024=0.221448. Thus75512 is best among those measured checkpoints, not established convergence of this recipe. Later runs to85752 changed recipes and cannot substitute for a long unchanged continuation. Original automated selection.json chose75000 by PSNR; delivery explicitly reselected75512 by user-priority person LPIPS. No validation-loss plateau claim is justified.

## Open DD review, 2026-11-01

The user reports most floaters have gone. Remaining on best_DD_HD.png: white haze beside the right arm closest to the observer; tiny white spots at the outer sleeve edge/in a fold toward the back of that shoulder. Cause not verified. Preserve this image as the baseline and do not launch a new improvement series until the user's next instruction. Exact recorded settings and checkpoint identity: [best Misha recipe](misha_002304_best_recipe.md).
