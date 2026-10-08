# Frame 000470: cord termination mass and frozen depth-band colour

## What was tested

The empty-cell hypothesis failed in the preceding [occupancy audit](luster_000470_cord_voxel_cover_and_layers.md). This experiment checks whether the remaining fading is caused by ray weight landing in front of or behind the cord. A read-only renderer diagnostic replaces RGB only for pixels close to the fitted train-only 3D line: it renormalizes field colour from samples inside a `±.003` depth band if their native weight exceeds `.5`. The line fit uses train views; no source masks or held-out pixels were used to train or fit it. This is a diagnostic colour composition, not physical volume rendering or a trained result.

The trainable alternative adds `cord_ray_objective=termination_mass`. It queries 60 density samples within `±5×.003` of the fitted line on 32 train-only rays per step and penalizes ray termination outside the central `±.003` band, missing central opacity, and six more distant foreground probes. It shares the same `.003` coefficient, source sharp-28 32k checkpoint, optimizer, seed, 28 selected cameras, local cord sampler (`radius=.005`, step `.001`), and evaluation as the earlier density-band arm. The first 32 ordinary pixel/RGB batch hashes matched exactly. Both arms finished their 32k→33k continuations without OOM. The termination objective adds about 1 second over the earlier density-band arm's 73-second optimization segment.

## Results

| Training | Step | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Density band `.003` | 32500 | **19.8513** | **.81728** | **.22065** | 24.0604 | .78214 | .07578 | 26.4249 | .82301 | .07633 |
| Termination mass `.003` | 32500 | 19.8350 | .81702 | .22109 | **24.1803** | **.78729** | **.07182** | **26.6763** | **.82814** | **.07401** |
| Density band `.003` | 33000 | **19.8082** | **.81742** | **.22068** | **24.2345** | .78785 | .07368 | 26.7808 | .82827 | .07325 |
| Termination mass `.003` | 33000 | 19.8052 | .81718 | .22097 | 24.2186 | **.78862** | **.07212** | **26.8668** | **.82988** | **.07251** |

The global checkpoint rule selects 33k for the termination-mass arm: the PSNR gap between its 32.5k and 33k steps is below `.07 dB`, and 33k has lower LPIPS. At 33k, held-out cord SSIM and LPIPS improve over the density-band version, but its global metrics worsen slightly. The [GT/no-prior/density-band/termination-mass comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/termination003_33000_fourway.png) shows that the lower braid still fades into the skirt. The [selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_termination003_033000/selection.json) and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_termination003_033000/supervision.jsonl) document the completed run.

The fixed-ray [audit](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/termination003_33000_ray_audit/result.json) uses the same held-out cam095 pixels in the prior-free and termination-mass 33k fields. On 91 rays within `.002` scene units of the line, mean native weight in the central depth band rises `.651→.812`, while front weight falls `.096→.053` and back weight `.252→.135`. For 36 lower rays, band weight rises `.566→.804`, front falls `.100→.054`, and back `.334→.142`. The prior therefore changes the actual rendering weights in the intended direction. On the 69 near-line rays with band mass above `.5` in both fields, native RGB PSNR improves `21.85→22.40 dB`, but the field colour normalized within that band worsens `23.62→23.05 dB`. The latter indicates a colour or cross-view alignment problem remains after the mixing problem is reduced.

The frozen diagnostic on the prior-free 33k checkpoint used a radius `.003` and a smooth radial falloff. Relative to native local sampling with occupancy admission, global PSNR/SSIM/LPIPS change `19.82482/.81686/.22052 → 19.82484/.81686/.22052`; held-out cam095 cord changes `22.6863/.78680/.06694 → 22.9923/.78883/.06569`. Training cam097 cord PSNR changes `27.4103→27.3707 dB`, while LPIPS improves `.06749→.06596`. A wider hard replacement improves cam095 cord PSNR to `23.5023 dB` but degrades SSIM to `.77504` and LPIPS to `.07595`, visibly changing adjacent fabric. The [four-way crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_control33k_conditional_band_fourway.png) shows the smooth version is localized, but neither version reconstructs the GT lower braid.

The same frozen smooth diagnostic was independently screened on the globally
best 30.5k baseline without retraining. Global PSNR/SSIM/LPIPS change
`19.98045/.812109/.225064 → 19.98049/.812107/.225060`; held-out cam095
cord changes `21.7108/.75707/.08283 → 22.0051/.75854/.08161`. Training
cam097 cord changes `26.5995/.81836/.07767 → 26.5168/.81771/.07689`.
Only 8,158 of 16,197,120 rendered pixels change by more than one 8-bit
level, and the field fingerprint is unchanged. The [GT/baseline/diagnostic
crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_30500_conditional_band003_tube003_soft_comparison.png)
shows a small local improvement but persistent lower-braid blur.

The train-only [two-landmark fit](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/landmark_audit_v2/fit.json)
helps explain why a hard prior may be risky: the junction has median/max
training reprojection residuals of `2.65/4.36 px` at the half-resolution
training scale; the first-bead anchor has `1.61/3.24 px`. These are landmark
residuals, not a measured dense calibration error, but they are large relative
to the visible braid details in the crop. A straight line between two anchors
cannot model every twist.

A [source-resolution check](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/cam095_gt_original_vs_1920.png)
compares cam095's original 4096×3000 GT crop with the actual 1920×1406
training/evaluation image enlarged back to the original pixel grid. The lower
braid loses some fine edges through downsampling, but its separate twists are
still clearly visible at 1920. Resolution loss alone therefore cannot explain
the much stronger blur in the rendered lower section; a full-resolution
continuation would add cost without first fixing the observed colour/depth
misalignment.

## Insights

Filling the occupancy grid and pushing native termination into the depth band are insufficient to recover sharp texture. Stronger geometry improves the depth distribution but can damage the predicted colour at that depth. A small, smooth suppression of front/back colour helps both prior-free checkpoints on cam095, suggesting layer mixing is one real contributor, yet the narrow benefit and the remaining lower-braid blur prevent adoption as a general renderer. The full-person best remains the sharp-28 30.5k baseline with `19.9804 dB` global PSNR. A subsequent [separate-density matched gate](luster_000470_cord_repeat_and_density_split.md) also failed on both cord views; local multiview colour alignment remains the next diagnosis.
