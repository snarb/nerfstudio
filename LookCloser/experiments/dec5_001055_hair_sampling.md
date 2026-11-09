# DEC5 001055: hair and contour sampling ablation

## What was tested

Four matched native continuations on clever-shadow, same full-state step4999
parent, through9999. Fixed60% hair /30% other-person quota, frozen background,
same empty-space objective, existing59-train/3-validation split. Changed only
allocation inside the hair quota: control30% boundary share,65% boundary share,
per-pixel frequency weighting1→3, or both. No temporal objective was added.

The per-pixel frequency arm is **different from** the recovered Luster000540
leader's sampling of nonempty frequency levels followed by uniform within-level
draws. This series does not invalidate that leader recipe.

## Results

Mean of three held-out views at step9999; native evaluation stride3, fixed
GT-composited semantic hair/face tight crops for LPIPS and SSIM:

| Arm | Hair PSNR | Hair SSIM | Hair LPIPS | Face LPIPS |
|---|---:|---:|---:|---:|
| Control | 24.9421 | .76273 | .239941 | .190497 |
| Boundary65 | 24.8431 | .76120 | .241732 | .190092 |
| Frequency | 24.9420 | .76226 | .243793 | .191645 |
| Frequency + boundary65 | 24.8756 | .76135 | .241213 | .190011 |

[Sealed per-view metrics](assets/dec5_silver_sable/sampling_hair_face_metrics.json).
These crop metrics use a different protocol from the UHD opacity benchmark and
must not be compared numerically across the two reports.

## Insights and next steps

No candidate improves hair LPIPS. None is promoted or receives further local
quota tuning. Stronger contour sampling alone does not repair inaccurate or
multilayer geometry. The new50-frame queue retains the recovered, measured
body80/FAS50 leader recipe. A separate HD pilot tests that recipe with all64
allowed training cameras. Its separate projection/distortion probes and metrics
are documented in the [000901 preflight report](dec5_000901_leader_preflight.md).
