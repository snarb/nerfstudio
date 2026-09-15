# DEC5: guarded local patches from a silhouette-constrained MHR prior

## What was tested

One frame (`001193`), one final prior from the [100-step silhouette continuation](dec5_mhr_silhouette_convergence.md), and the two frozen admission branches: direct multiview measured-depth support versus direct support plus certified local interpolation. This is inferred local completion, not a replacement for COLMAP or a claim that MHR predicts metric truth. The continuation reached its iteration cap, not convergence; its unsafe facets remain invalid.

Candidate root: `/mnt/data/dec5_mhr_silhouette_patch_candidates`. Admission/render root: `/mnt/data/dec5_mhr_silhouette_patch_admission`. All older controls remain unchanged. Original raw mesh SHA starts `316dc2b6`, with exactly 66,373 vertices and 128,203 triangles preserved as the output prefix. This is not yet integration with the separately carved production mesh or the active 6K video.

The generic domain uses neutral MHR y=135..153 cm, excludes every current crossing/reversed parent facet (including inherited unsafe facets), subdivides to edge ≤.00075 scene units, and requires all vertices within .002 of the original surface, .003 of an original boundary vertex, and normal dot ≥.25. The centroid gap ≥.00002 remains unchanged. There is no nearest-open-edge requirement and no target ROI in construction, fitting, or admission. Explicit exact-file aliases bind the NEW final prior/topology seal to the frozen admission helper; they do not impersonate the older smooth100 prior.

Admission keeps the previous masks, depth receipts, footprint/free-space vetoes and thresholds. Direct support is based on actual calibrated multiview depth correspondences. The interpolation branch additionally requires local measured seeds (≥8, each with ≥3-view observed support), radius .003, a containing projected hull, and bounded quadratic leave-one-out residual/offset ≤.0005 at every proposed vertex. Camera/frustum counts alone are insufficient. Final native checks cover 62 train cameras at both integer and half-pixel lattices; only added facets can be removed, for at most eight rounds.

The independently constructed D/D measured foreground override is rebound only as a depth/mask artifact, not its different source-mesh assertion. All 62 train views participate in this admission. The prior's eight reserved fitting cameras are therefore **not** independent admission validation; moreover original COLMAP and the inherited override already have all-62 provenance. The three true held-out RGB views never supply fitting, admission, texture or selection inputs.

## Results

| Stage | Strict | With certified interpolation |
|---|---:|---:|
| Raw generic proposals | 577,120 | same |
| Semantic support and zero disagreement | 281,031 | same |
| Initially admitted | 179,109 | 185,115 |
| Removed by native measured-depth veto | 137 | 181 |
| Final added triangles | 178,972 | 184,934 |
| Fixed original puncture: missing pixels | 44 → 1 | **44 → 0** |
| Filled puncture pixels with nonzero train-textured RGB | 43 | 44 |
| Source-255 fallback among those fills | 0 | 0 |

Extraction excluded 77 unsafe anatomical parent facets, subdivided 2,960,384 facets, and rejected 20,192 otherwise local facets by the unchanged centroid gate. All 44 posthoc requested points survive the geometric/semantic gates; none is rejected by the centroid heuristic or trusted free-space veto. Direct depth admits 43; interpolation supplies the last one. There are 2,740 validated observed seeds and 83,392 certified vertices among 148,631 queries. Admission took 135.86 seconds on CPU. Both branches terminated after two native passes, with zero final violations. The full unused candidate vertex suffix remains retained; neither mesh is a compact export.

Actual native RGB review:

- [Moving view: baseline / strict / interpolated](/mnt/data/dec5_mhr_silhouette_patch_admission/rgb_review/old_moving_native.png): the black puncture below the cheek disappears; the true cast shadow persists. No obvious new seam immediately at the fill. Parent independently inspected this panel and agreed.
- [F/E train camera](/mnt/data/dec5_mhr_silhouette_patch_admission/rgb_review/F004_E_native.png), [M/B train camera](/mnt/data/dec5_mhr_silhouette_patch_admission/rgb_review/M004_B_native.png), [C/E train camera](/mnt/data/dec5_mhr_silhouette_patch_admission/rgb_review/C004_E_native.png): no broad new neck fold or obvious new source-color seam in these reviewed views. Existing jagged under-jaw/neck silhouettes, missing underside fragments and source-color mosaics remain. These views do not demonstrate whole-neck repair.
- [Native clay and count evidence](/mnt/data/dec5_mhr_silhouette_patch_admission/native_clay_review/result.json), [strict/interpolated difference](/mnt/data/dec5_mhr_silhouette_patch_admission/branch_difference/result.json), [posthoc facet attribution](/mnt/data/dec5_mhr_silhouette_patch_admission/facet_attribution/result.json).

Append-only topology does **not** mean unchanged rendering of old geometry. New facets can occlude an old deeper layer. Exact signed-depth localization found:

| View | Pixels >.003 nearer, strict / interpolation | Maximum nearer depth shift |
|---|---:|---:|
| Moving | 4 / 4 | .015071 |
| F/E | 105 / 115 | .016917 |
| M/B | 6 / 6 | .012538 |
| C/E | 8 / 8 | .003286 |

These are not rounding errors. Moving changes occur at native x=540..541, y=1047..1050, adjoining the repaired puncture/rim. F/E's largest component is 82 pixels at x=713..722, y=1168..1182, along the thin under-jaw/neck fringe; M/B changes are also at that rim, whereas C/E changes touch the collar boundary. [Moving localization](/mnt/data/dec5_mhr_silhouette_patch_admission/occlusion_review/old_moving_interpolated_2.png), [F/E largest component](/mnt/data/dec5_mhr_silhouette_patch_admission/occlusion_review/F004_E_interpolated_1.png), [M/B](/mnt/data/dec5_mhr_silhouette_patch_admission/occlusion_review/M004_B_interpolated_1.png), [C/E collar](/mnt/data/dec5_mhr_silhouette_patch_admission/occlusion_review/C004_E_interpolated_5.png). No common ray moved farther. The confidence guard found no corroborated measured free-space contradiction, but it does not prove that every new nearer occlusion is anatomically correct.

At unchanged moving-view geometry, source labels changed at 307 strict / 321 interpolated pixels, because the source-label graph is recomputed on the changed mesh. Across the full moving frame, RGB changed at 1,410 / 1,480 pixels. Thus this is not a texture-neutral 44-pixel edit. F/E strict also has one newly hit pixel with zero RGB (1 of 2 new hits); interpolated has 7/7 colored new hits. No blanket color-completeness claim is made.

The 12 serial full-native RGB renders use fixed original train EXRs, calibration, color profiles/exposure, source-mask and temporal-view-prior wrappers for every mesh. The D/D geometry override is **not** used for texture. Device execution alone is shimmed to CPU with two Torch/raycast threads and one source-loading worker; CPU/CUDA equivalence was not measured. Carried parent repair metadata does not select geometry: every request explicitly binds the raw baseline or the corresponding admitted mesh. These are matched train-textured diagnostics, not independent reference images. PSNR/SSIM/LPIPS: N/A; no held-out fidelity improvement is claimed.

Reproduction (Python `/home/brans/repos/nerfstudio/.venv/bin/python`; `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`, plus `OPENCV_IO_ENABLE_OPENEXR=1` for RGB): run `build_mhr_silhouette_patch_candidates.py`, `admit_mhr_silhouette_patch.py`, `review_mhr_silhouette_patch.py`, `render_mhr_silhouette_patch_cpu.py`, `review_mhr_silhouette_patch_rgb.py --views old_moving F004_E M004_B C004_E`, and `localize_mhr_silhouette_patch_occlusion.py`. Producers require new output directories; do not overwrite these frozen artifacts. `audit_mhr_silhouette_patch_candidates.py` completely replays candidate arrays and raw PLY; `admit_mhr_silhouette_patch.py --audit` replays measured admission and final native vetoes. The final seal binds render inputs, 62 actual train EXRs, helpers, receipts, views and this report.

Verification passed: complete candidate extraction reproduced every stored array and the raw PLY byte-for-byte; measured admission replay reproduced 2,810,310 depth/footprint samples, 148,631 vertex certificates and 248 final native ray checks. All original vertices/triangles remain an exact prefix. The bounded tests cover unchanged guard function identities, the retained generic locality gate, unsafe-parent exclusion and signed occlusion accounting; the existing measured-admission tests are also rerun.

## Insights

This is the first positive local requested-hole result in this MHR sequence: silhouette-aware depth relocation followed by unchanged measured-confidence admission succeeds where the earlier front-layer priors did not. Certified interpolation contributes one final puncture pixel, not the bulk of the improvement. It is a single-frame local result, with retained nonzero old-surface occlusion and source-label side effects.

Do not promote the full prior, all 185k added facets, or this frame into the production/6K baseline from this check alone. A large overlapping inferred shell remains possible even when sampled confidence checks pass. Production-mesh integration, independent held-out fidelity, temporal transfer and wider-view safety are untested. Hair was not modified or evaluated by this face/neck-only experiment. No defaults, source geometry, source masks, camera path or video were changed.
