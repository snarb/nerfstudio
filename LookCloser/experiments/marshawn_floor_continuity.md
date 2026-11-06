# Floor continuity under temporal AABB clipping

## What was tested
Read-only re-render of immutable R1 step 20000, camera HC/frame 1660. Same floor-aligned frame and checkpoint as temporal review commit 40e6a054. Three expanded boxes; no optimizer updates, no camera/field/occupancy changes, no compositing or image inpainting. Original actor box remains selected.

## Results
| Bounds changed | Volume / original | Mat low-opacity pixels | Behind feet low-opacity pixels |
|---|---:|---:|---:|
| Original zmin -0.07, ymax 0.42 | 1.000 | 8087 | 4545 |
| zmin -0.10 | 1.043 | 5052 | 4545 |
| zmin -0.16, ymax 0.48 | 1.236 | 1096 | 4542 |
| zmin -0.20, ymax 0.82 | 1.939 | 71 | 0 |

Diagnostic pixels have clipped alpha < 0.5 and full-field alpha > 0.95. Mat ROI [550,600,810,675] has 19458 valid pixels; behind-feet ROI [960,588,1220,628] has 10400. These are clipping diagnostics, not ground-truth geometry or image-quality metrics. All images visually reviewed. Wide variant still has small dark residuals on the mat and includes lighting equipment; rejected as a slight actor-only box correction.

Ray traces confirm removed density: mat sample peak approximately (-0.503,0.362,-0.111), mat edge (-0.444,0.453,-0.156), behind left leg (-0.286,0.748,-0.149), in the floor frame. Some learned floor/mat appearance is represented below the fitted floor or beyond the actor box. Expanding only the bottom cannot restore the back-leg patch. This historical radiance field cannot be treated as reliable surface geometry just because the full render looks continuous.

## Insights
Small expansion improves mat clipping but does not satisfy continuous-floor request. Large expansion substantially restores the floor at the cost of 94% added volume and background equipment. Keep accepted temporal actor box unchanged; retain variants for review. No claim of a fully uniform floor or optimal revised box. A later geometry correction would require separately authorized model work, not silently enlarging the actor box to accommodate distorted density.

Artifacts: `/fsx/tmp/marshawn_001660_hd/two_actor_isolation/floor_continuity_v2`.
Download: `/fsx/tmp/marshawn_001660_hd/downloads/aabb_floor_continuity_comparison_20261106.zip`.
Open `comparison_floor.jpg`, then `z020_y082_HC/HC_inside_box_visible.png`. PNGs are actual field renders. Original and all variants are included, with ray traces and receipts. Box JSON metadata margins were refreshed after runs; AABB and rigid transforms were unchanged by that metadata correction. Receipts retain their original file hashes.
