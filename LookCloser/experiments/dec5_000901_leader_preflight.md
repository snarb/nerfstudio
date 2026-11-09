# DEC5 000901: full leader HD preflight

## What was tested

Clever-shadow, genuinely fitted1920×960 frequency maps,64 train cameras and
held-out D004_A005_12103C. Full106317-update body80/FAS50/Jacobian recipe,
independent foreground NeRF plus frozen background color model. This is an HD
preflight, not native6K production completion. Production preparation continues
independently on silver-sable. Predictions are sealed before reference semantics
are computed for metrics; references never enter fitting or source blending.

## Results

At the first two gates, fixed GT-composited semantic hair/face crops:

| Step | Hair PSNR | Hair SSIM | Hair LPIPS | Face LPIPS | Boundary LPIPS |
|---|---:|---:|---:|---:|---:|
| 15188 | 24.2076 | .77921 | .24876 | .12855 | .10587 |
| 30376 | 23.7813 | .77228 | .22057 | .11324 | .10642 |

Hair LPIPS improves11.3% during this interval, face LPIPS11.9%; contour quality
does not yet improve. These are training-progress comparisons, not a win over
the final baseline. All-eval PSNR/SSIM/LPIPS at30376 are16.6513/.74396/.49663.
Full-frame metrics are dominated by the poor independent background RGB model.
Trusted background NeRF opacity averages.0226, versus.9315 inside reference hair
and.9946 inside the face. Visually, the person is substantially sharper than the
blotchy background; the hair remains soft. Actual accumulated pixel allocation
at the pilot audit:11.6% hair,61.8% other person,25.7% background,1.0% uncertain.
This is why support80% must not be described as a strict80/20 pixel split.

Matched5000-update continuations from the exact same full-state15188 checkpoint:

| Projection during frequency updates | Hair PSNR | Hair SSIM | Hair LPIPS | Face LPIPS | Boundary LPIPS |
|---|---:|---:|---:|---:|---:|
| Expected depth, control | 23.9461 | .77461 | .238265 | .127544 | .109563 |
| Front-q10 | 23.8482 | .77498 | .239077 | .123903 | .110024 |

Both arms restore Adam/scheduler/scaler/RNG and perform the same5000 updates.
The front-q10 adapter reuses packed weights only during grid updates, without
extra density queries. Normal render depth/RGB and the opaque-support gate remain
unchanged. Neither hair criterion improves: **closed, not promoted**.
[Control](assets/dec5_silver_sable/frequency_projection_control_hd.json),
[front-q10](assets/dec5_silver_sable/frequency_projection_front10_hd.json).

## Insights and next steps

- Production retains expected-depth frequency projection. A matched continuation
  with distortion.01 instead of1 worsens hair LPIPS.238265→.242478 (1.77%).
  Boundary LPIPS improves only.109563→.109058 (0.46%); mean trusted-background
  opacity rises.02407→.03329. Below the improvement gate: closed, retain1.
  [Metrics](assets/dec5_silver_sable/distortion01_continuation_hd.json).
- A frozen-foreground background-capacity test also fails: width128, origin2 /
  direction9 Fourier levels and20000 steps, fitted strictly on64 training views,
  versus width64/origin1/direction5/5000. With identical foreground/depth/opacity,
  trusted-background PSNR worsens14.7926→14.2318 and boundary LPIPS
  .10587→.17834. More capacity does not guarantee cross-camera generalization.
  Closed; production background recipe is unchanged.
  [Metrics](assets/dec5_silver_sable/background_capacity_hd.json).
- A matched5000-update ablation turns off the observed-RGB correction only on
  trusted eroded training background. Relative to the existing correction-on
  control, hair LPIPS worsens.238265→.240337, face LPIPS.127544→.135779 and
  boundary LPIPS.109563→.115850. Keep the correction in the new scene's recipe.
  This does not reverse the old1055 continuation's negative diagnostic or claim
  a large hair-depth gain; it checks the objective choice on the new901 pilot.
  [Metrics](assets/dec5_silver_sable/observedbg_off_continuation_hd.json).
- The final FR.3 stage remains required before judging the full recipe. Native
  checkpoint selection follows the mandated whole-view PSNR/.07dB-LPIPS rule;
  supplementary hair metrics do not silently change that selection.
- HD pilot metrics, masks and numeric values use a different protocol from the
  old1055 UHD benchmark; do not compare them across frames as an improvement.
- Artifacts: `/home/ubuntu/lookcloser_transfer/recipe_pilot_20261028` on
  clever-shadow. New short trials never mark a native production frame complete.
