# DEC5: bounded MHR local-completion transfer to 001195

## What was tested

Transfer of the [001193 silhouette-constrained MHR patch method](dec5_mhr_silhouette_patch_admission.md) to `001195`, using fresh observations and anatomical alignment—not copied `001193` fitted pose or geometry. One frame, unchanged hyperparameters, CPU-only. Artifact root: `/mnt/data/dec5_mhr_transfer_001195`.

The pipeline reuses only the neutral/template topology and frozen functions: fresh canonical similarity from this frame's measured face landmarks; MHR similarity and regularized head20 controls; named neck/head rotational fit (indices 24..29); measured smooth100 conformance; the same ten-step control and exact 100-step continuation. Magnitude/Laplacian penalties remain anchored to this frame's original smooth100 base throughout. There is no weight search, silhouette-mask relaxation, target-ROI input, predicted-landmark-z depth, or whole-prior replacement.

Per-time geometric depths, calibrated train RGB, person masks and independent D/D measured foreground override are hash-bound. Eight train cameras remain excluded from landmark/anchor/silhouette fitting and direct depth voting; original COLMAP and inherited override still have all-62 provenance. Final patch admission explicitly uses all 62 train cameras. The three true held-out views never supply inference, texture or selection input. Independent held-out image fidelity and temporal interpolation remain untested.

The generic candidate and confidence gates are identical to `001193`: neutral y=135..153 cm; exclude all final crossing/reversed parent facets; subdivided edge ≤.00075 scene units; vertex distances ≤.002 to original surface and ≤.003 to original boundary vertices; normal dot ≥.25; centroid gap ≥.00002. There is no nearest-open-edge gate. Direct measured-depth admission and the certified local-interpolation branch retain the exact observed-seed, hull, leave-one-out, free-space, semantic and 62×two-lattice native veto rules. All additions are inferred geometry; old vertices/triangles are untouched.

## Results

Fresh initialization used 6,927 fitting and 1,084 reserved measured canonical anchors. The MHR stage obtained 12,400 validated per-camera depth anchors, with 1,600 reserved. The coarse “neck candidate” group includes chin/neck/clavicle observations, not 6,200 guaranteed underside constraints. Actual final head20+neck6 associations within .006 retain 5,400 fitting face and 5,356 fitting neck-group points. Native G/B anchor inspection shows skin with a few ear samples; no obvious hand/hair/background samples in that panel.

| Prior stage | Reserved neck surface-distance P90 |
|---|---:|
| Head20 | .006231 |
| Head20 + named neck6 | .004479 |
| Measured smooth100 | .000151 |

The whole MHR face still does not reproduce the actor reliably. [Fitted controls, G/B](/mnt/data/dec5_mhr_transfer_001195/head/review_similarity_head20_head20_neck6/G004_B005_1210FG.png) and [final C/E clay](/mnt/data/dec5_mhr_transfer_001195/silhouette100/review_v2/C004_E005_1210X7_clay.png) were actually inspected; eye/nose folds rule out whole-prior use. The silhouette stage improves the lower-neck fit, not those frozen upper-face defects.

On the fixed 11,471-sample reserved silhouette cohort, mean excess beyond the unchanged two-pixel tolerance falls 14.779→.00530 px; outside samples fall 2,343→61. Final all-available training samples still include 982 outside points. The continuation hits the 100-step cap **without convergence** (unconstrained max step .001031), with max displacement .033275. It retains 149 crossing pairs, including 24 new pairs, and 86 >90° normal changes; all affected candidate parents are excluded. Normal change and actual self-crossing are distinct diagnostics.

All 73 original missing moving-view rays now hit prior points that pass all 62 masks at **zero** tolerance, versus 0/73 before silhouette fitting. The posthoc 37-point original-hit rim has final distance median/P90 .000174/.000342; signed-normal median/P90 −.000060/+.000176. The target points' original-surface distance P90 is .000451. These are feasibility diagnostics only; they never select fit data or candidate facets. [Requested prior-only clay](/mnt/data/dec5_mhr_transfer_001195/silhouette100/review_v2/requested_hole_prior_only.png), [native projected silhouette evidence](/mnt/data/dec5_mhr_transfer_001195/silhouette100/review_v2/G004_B005_1210FG_projection.png).

| Local completion stage | Strict | With certified interpolation |
|---|---:|---:|
| Raw candidates | 567,318 | same |
| Semantic admission | 282,846 | same |
| Initial measured admission | 179,464 | 185,243 |
| Native-veto removals | 239 | 257 |
| Final added triangles | 179,225 | 184,986 |
| Fixed moving puncture, missing pixels | **73 → 1** | **73 → 1** |

The candidate domain excludes 88 unsafe anatomical parent facets and rejects 21,944 otherwise-local facets through the frozen centroid gap. Both native-veto branches need four passes (strict removals 232/6/1/0; interpolation 250/6/1/0), within the original eight-pass cap. The original 66,279 vertices and 127,932 triangles remain an exact prefix. There are 2,701 validated observed seeds and 88,306 certified vertices among 149,142 queries. The unused candidate vertex suffix is retained, not compacted for export.

The final missing pixel is explicitly rejected by the unchanged centroid-gap rule: its associated facet centroid is .000016253 from original geometry, below the .000020 minimum. All 73 points otherwise match the safe/local prior domain, and all 72 extracted target facets pass masks and direct depth. Interpolation provides **no additional requested-hole closure** on this frame. No threshold was relaxed to remove the last pixel. Admission took 170.30 s. [Admitted native clay](/mnt/data/dec5_mhr_transfer_001195/admission/native_clay_review/requested_hole.png), [facet attribution](/mnt/data/dec5_mhr_transfer_001195/admission/facet_attribution/result.json).

The [moving native RGB comparison](/mnt/data/dec5_mhr_transfer_001195/admission/rgb_review/old_moving_native.png) was actually inspected. Both branches fill 72 puncture pixels with nonzero train-textured RGB and zero source-255 fallback; the natural shadow persists, with no obvious new seam at the fill. The one remaining missing pixel remains explicit. Across the whole moving image, strict/interpolation introduce 74/77 new colored hits, change common-hit depth at 1,510/1,533 pixels (maximum .010039 scene units), and change source labels at 112/113 geometrically unchanged pixels. RGB changes at 1,527/1,553 pixels total. This is not a texture-neutral edit or proof that every appended facet is harmless.

The [F/E](/mnt/data/dec5_mhr_transfer_001195/admission/rgb_review/F004_E_native.png), [M/B](/mnt/data/dec5_mhr_transfer_001195/admission/rgb_review/M004_B_native.png) and [C/E](/mnt/data/dec5_mhr_transfer_001195/admission/rgb_review/C004_E_native.png) native RGB panels were also actually inspected. No obvious new large neck fold or broad source seam appears in these views. The severe pre-existing thin under-chin/neck fringe, other small jaw gaps and neck texture mosaic remain. New hits are colored in all reviewed frames: F/E 102/117, M/B 58/59, C/E 15/21 (strict/interpolation). This does not imply that the full inferred shell is correct.

| View | Common-hit pixels >.003 nearer, strict / interpolation | Maximum nearer depth change |
|---|---:|---:|
| Moving | 1 / 1 | .010039 |
| F/E | 101 / 104 | .014872 |
| M/B | 2 / 3 | .011710 |
| C/E | 4 / 4 | .003153 |

These are real nearer occlusions, not roundoff, and no common-hit ray moves farther. The moving pixel is at native (539,1054), adjoining the puncture. F/E's largest component is 85 pixels in x=713..724,y=1167..1183 at the thin neck fringe; M/B changes touch the jaw/neck rim, and C/E changes touch the collar boundary. The [moving](/mnt/data/dec5_mhr_transfer_001195/admission/occlusion_review/old_moving_interpolated_1.png), [F/E](/mnt/data/dec5_mhr_transfer_001195/admission/occlusion_review/F004_E_interpolated_1.png), [M/B](/mnt/data/dec5_mhr_transfer_001195/admission/occlusion_review/M004_B_interpolated_1.png) and [C/E](/mnt/data/dec5_mhr_transfer_001195/admission/occlusion_review/C004_E_interpolated_2.png) localized crops were inspected. The measured veto finds no corroborated free-space contradiction; that is not a guarantee that every new nearer occlusion is correct.

All 12 full-native RGB renders use the same per-camera color profiles/exposure, source-mask and temporal-view-prior wrappers, and original train EXRs for all three meshes. The geometry-only D/D override is not used for texture. Only device execution is adapted to CPU; CPU/CUDA equivalence was not measured. These train-textured views are not independent references: PSNR/SSIM/LPIPS are N/A, and no held-out fidelity improvement is claimed. Timings are diagnostic, not an uncontended hardware benchmark.

Seven bounded tests passed. Fresh MHR model forward/residual replay (three fits), all 12,400 integer-depth anchor positions and reserved-excluding support counts, exact measured-conformance arrays, all 100 continuation iterates, and complete candidate extraction/byte-identical raw PLY replay passed. The final admission audit reproduced 2,828,460 depth/footprint samples, 149,142 certificates and all 248 final native ray checks. Its first shell session returned 143 after the complete passed receipt/success line; that abnormal supervision result remains unexplained and retained. A separate complete rerun reproduced a **byte-identical audit receipt and exited normally (0)**. No fit, candidate, gate or geometry changed between audits.

Execution/provenance: `/home/brans/repos/nerfstudio/.venv/bin/python`, two CPU threads; isolated `/home/brans/lookcloser_temp/mediapipe_hand_env/bin/python` for semantics. An initial semantic orchestration attempt failed because that isolated environment lacks Torch; a minimal-import wrapper runs the exact same semantic function successfully in 9.62 s. The failed log is retained; no dependency installation, new model or numerical change was needed. `transfer_mhr_001195.py` owns explicit frame/config rebinding; `patch_mhr_transfer_001195.py` records exact source substitutions for literal frame/path references, retaining original and generated function text/hashes. No frozen producer was edited.

## Insights

The anatomical/silhouette method transfers its local improvement to a second time with fresh alignment and identical confidence thresholds. Strict measured support already gives the observed 72-pixel puncture repair; certified interpolation adds no target benefit here and changes additional pixels elsewhere. This is a positive bounded transfer with known residuals, **not production acceptance** or proof of full-neck/continuous temporal reconstruction. The current untouched production/carved-base integration is a separate study. `001191` was not reconstructed or tested; hair was not changed or evaluated. Original data, defaults, active 6K rendering, production camera path and source-color settings are unchanged.
