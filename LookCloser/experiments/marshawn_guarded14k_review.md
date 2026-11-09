# Marshawn14k known-point guard review: cleanliness gate failed

What was tested: exact14k field `e92e525fe7451ef9cc2959ecca57a09e68925b20143714bf463c54a6ed998ae9`, native core versus authoritative core UNION9022 observed train-witness voxels followed by exactly one cube-radius1 halo. No NN, EMA, checkpoint, AABB, camera or colour changes. Three new prediction-only1920×1012 renders plus raw buffers were produced with the guarded mask; immutable existing default renders were reused.

Results:

| Policy/model | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
|14k default |29.235580|.813125|.269608|
|14k known-point +one halo |29.252612|.814590|.267418|
|Published safe13k guarded |29.089779|.803826|.300436|

Native14k misses68/9022 known points including11/495 BODY points. New core291543cells plus38 extra observed seed cells, then one halo, admits1007524cells with0/9022 and0/495 known misses. Ordinary one-halo(core) admits1007443cells and still missespoint2813. Added81 cells over ordinary halo are inference safety; no cell is certified physically occupied or empty. Core/EMA bytes and the original model state are unchanged.

Same BODY128 source selection and geometry remain exact. Native, temporary all-cells and uniform16384 profiles all retain0/128 free-prefix mass>10% and126/128 surface-band mass≥50%. Guarded native also retains0/128 and126/128. These sparse fitted visible-ray checks do not certify hidden volume, floor depth or all object edges.

All3 full HD images, native KA floor/rig crop and actual RGB/opacity/depth difference maps were manually inspected. Recognizable opaque actors remain, but broad gray/brown floor mottling and dark KA left-floor patches remain. Root's actual visual review records `quality_goal_pass=false`. No promotion follows from the improved aggregate metric.

KA specifically regresses29.910138→29.577829PSNR under the guard. Saved-buffer differences locate the regression mainly in broad interior floor/mat regions:

-64pixel border improves28.482→28.556PSNR and offsets5.5% of the net error increase.
-Interior worsens30.311→29.847PSNR.99.987% of these pixels have opacity≥.99 in both controls; median alpha delta is−2.4e−7. Median encoded-luma delta−.00232 and expected-depth delta−.01239 prepared-world units indicate colour/depth redistribution at nearly fixed opacity.
-Predeclared suspect-floor ROI `[150,700,290,820]` worsens26.041→25.546PSNR, is opaque in both, and has median encoded-luma delta−.00339 /expected-depth delta−.02006.
-Another standard foreground floor ROI improves, so the effect is spatially nonuniform. Raw differences alone cannot separate all newly admitted density from changed traversal/quadrature positions or assign physical false-occupancy labels.

Evidence:

- [Prediction-only DC](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/DC.png), [HC](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/HC.png), [KA](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/KA.png).
- [GT /guard14k /guard13k contact](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/review/all3_GT_guard14k_guard13k.jpg).
- [Four-way native KA floor/rig crop](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/review/KA_native_floor_defect_rig.png).
- [Default/guard RGB, alpha, depth and error maps](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/review/KA_default_guard_rawdiff/overview.jpg), [numeric region audit](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/review/KA_default_guard_rawdiff/audit.json).
- [Source/state validation](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/checkpoint_source_check.json), [manual review](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/manual_review_occupancy.json), [root visual failure receipt](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/current_best_guarded_HD/root_visual_review.json).

Insight: known-point coverage and clean rendered geometry are separate gates. Preserve both14k controls, keep the published13k fallback unchanged, and do not claim an ideal grid or clean floor. No final14k atlas or release package was created. Further14k rendering/tracing stopped after these receipts.
