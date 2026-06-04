# LookCloser Visual Baseline Audit

## What was tested

Visual audit requested after the higher-metric LookCloser renders appeared worse than the bounded Instant-NGP baseline.

Compared artifacts:

- Ground truth eval images: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns/images/frame_eval_*.jpg`
- Instant-NGP baseline renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_focus_scene15/instant-ngp-bounded/20260527_122100/renders_last_step_60751`
- LookCloser fixed 384, all completed seed renders
- LookCloser fixed 512, all completed seed renders
- LookCloser fixed 640, completed seed 42 and 43 renders only; seed 44 was still running during this audit

Important setup mismatch found during the audit:

- Instant-NGP baseline used `scene_scale=1.5`, `scale_factor=1.0`.
- LookCloser fixed-sample runs used `scene_scale=2.0`, `scale_factor=1.15`.
- Therefore the fixed-sample LookCloser runs are not an apples-to-apples baseline comparison. The visual conclusion below is still useful for diagnosing the current artifacts, but the metric comparison must be treated as invalid for final acceptance.

The `eval_img_*.png` render files are `3840x1080` side-by-side images. The left half matches the ground truth exactly, and the right half is the model prediction. Crop sheets below use the right half for model renders.

## Results

Global metrics alone are misleading here.

| Run | PSNR | SSIM | LPIPS | Eval loss | Visual verdict |
|---|---:|---:|---:|---:|---|
| Instant-NGP baseline final eval | 24.417955 | 0.639772 | 0.460250 | 0.00374830 train-time best | Best visual detail on audited crops |
| LookCloser fixed 384 mean | 26.616046 | 0.585817 | 0.402468 | 0.03442633 | Worse fine detail |
| LookCloser fixed 512 mean | 27.009452 | 0.595257 | 0.390554 | 0.03358400 | Worse fine detail |
| LookCloser fixed 640 seed42 | 27.214973 | 0.604006 | 0.378298 | 0.03406570 | Worse fine detail; non-comparable setup |
| LookCloser fixed 640 seed43 | 27.334343 | 0.604640 | 0.377598 | 0.03286000 | Worse fine detail; non-comparable setup |

Notes:

- LookCloser improves PSNR and LPIPS relative to Instant-NGP, but SSIM remains lower.
- Eval loss is much worse for LookCloser. The loss was not changed for this fixed-sample sweep; this reinforces that the earlier PSNR/LPIPS gains should not be treated as an overall quality win.
- Visual crops show LookCloser is smoother and loses tiny structures, so the global metric gains are not aligned with the target objective of tiny-detail reconstruction.

### Crop Sheets

All completed higher-PSNR/lower-LPIPS LookCloser fixed-sample renders are included in these sheets.

| Crop | Sheet |
|---|---|
| Stand label / small writing | [stand_label_eval2.png](visual_crops/all_high_metric/stand_label_eval2.png) |
| Tangled cable / thin wires | [tangled_cable_eval2.png](visual_crops/all_high_metric/tangled_cable_eval2.png) |
| Center hand/fingers | [fingers_center_eval2.png](visual_crops/all_high_metric/fingers_center_eval2.png) |
| Right hand/fingers | [fingers_right_eval1.png](visual_crops/all_high_metric/fingers_right_eval1.png) |
| Floor crack / thin bright line | [floor_crack_eval0.png](visual_crops/all_high_metric/floor_crack_eval0.png) |

Focused sheets with fewer columns:

- [stand_label_eval2.png](visual_crops/stand_label_eval2.png)
- [tangled_cable_eval2.png](visual_crops/tangled_cable_eval2.png)
- [fingers_center_eval2.png](visual_crops/fingers_center_eval2.png)
- [fingers_right_eval1.png](visual_crops/fingers_right_eval1.png)
- [floor_tape_eval0.png](visual_crops/floor_tape_eval0.png)

### Local Crop Metrics

These local metrics support the visual read. The floor-crack crop is a useful failure case: LookCloser gets much higher local PSNR by smoothing the mostly flat floor, but local SSIM and visual inspection show that the thin crack/detail is lost.

| Crop | Run | PSNR | SSIM |
|---|---|---:|---:|
| stand_label_eval2 | instant_ngp | 27.915 | 0.8176 |
| stand_label_eval2 | lc384_s42 | 25.449 | 0.7183 |
| stand_label_eval2 | lc512_s44 | 26.064 | 0.7444 |
| stand_label_eval2 | lc640_s42 | 26.094 | 0.7498 |
| stand_label_eval2 | lc640_s43 | 26.359 | 0.7583 |
| tangled_cable_eval2 | instant_ngp | 27.943 | 0.7859 |
| tangled_cable_eval2 | lc384_s42 | 26.129 | 0.7078 |
| tangled_cable_eval2 | lc512_s44 | 26.745 | 0.7300 |
| tangled_cable_eval2 | lc640_s42 | 26.964 | 0.7339 |
| tangled_cable_eval2 | lc640_s43 | 27.224 | 0.7393 |
| fingers_center_eval2 | instant_ngp | 24.335 | 0.7971 |
| fingers_center_eval2 | lc384_s42 | 26.276 | 0.7501 |
| fingers_center_eval2 | lc512_s44 | 26.658 | 0.7644 |
| fingers_center_eval2 | lc640_s42 | 27.135 | 0.7804 |
| fingers_center_eval2 | lc640_s43 | 27.071 | 0.7783 |
| fingers_right_eval1 | instant_ngp | 25.377 | 0.7809 |
| fingers_right_eval1 | lc384_s42 | 25.141 | 0.7118 |
| fingers_right_eval1 | lc512_s44 | 25.462 | 0.7346 |
| fingers_right_eval1 | lc640_s42 | 25.768 | 0.7481 |
| fingers_right_eval1 | lc640_s43 | 26.245 | 0.7475 |
| floor_crack_eval0 | instant_ngp | 21.289 | 0.5912 |
| floor_crack_eval0 | lc384_s42 | 27.433 | 0.5386 |
| floor_crack_eval0 | lc512_s44 | 27.098 | 0.5412 |
| floor_crack_eval0 | lc640_s42 | 27.319 | 0.5522 |
| floor_crack_eval0 | lc640_s43 | 26.826 | 0.5492 |

## Insights

The current LookCloser/frequency-grid changes have not proven a visual quality improvement over the bounded Instant-NGP baseline. The accepted fixed-sample decisions were valid only as improvements over earlier LookCloser internal references, not as improvements over the actual baseline target.

Observed failures:

- Small writing on the stand remains more legible in Instant-NGP.
- Tangled cables and thin wires are sharper in Instant-NGP.
- Fingers and hand boundaries are smoother/less separated in LookCloser.
- Thin floor crack/detail is mostly removed by LookCloser despite higher PSNR.

Decision update:

- Do not treat fixed samples 512 or current 640 as an overall quality win.
- Stop using these fixed-sample results as final baseline evidence because their dataparser scale differs from the Instant-NGP baseline.
- Keep the metric tables for analysis, but require a visual crop gate against Instant-NGP before accepting any future change.
- SSIM and eval loss should be reported alongside PSNR/LPIPS and should be weighted more heavily for this tiny-detail objective.
- Next debugging should first rerun a LookCloser control with `scene_scale=1.5` and `scale_factor=1.0`, matching the Instant-NGP baseline, then repeat the visual crop gate. Candidate follow-up directions after that: loss weighting, exposure/color alignment, sampling/ray batch mismatch versus Instant-NGP, frequency-grid update quality, and whether final eval image output is being compared at the correct split/pose/scale.
