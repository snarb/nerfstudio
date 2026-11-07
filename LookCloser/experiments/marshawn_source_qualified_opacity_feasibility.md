# Marshawn source-qualified opacity feasibility

## What was tested

CPU-only inspection of existing TRAIN sources for a narrowly qualified opacity objective. No masks, source images, geometry, sampling or production recipes change. Artifacts: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/opacity_source_feasibility_v1` (`C` below means the parent campaign).

Historical accepted Luster evidence is `experiments/luster_000470_opacity_depth.md`: opacity_profile coefficient .1 combined with qualified TRAIN depth admission. It is not the later rejected spatial density_elements .01. Existing pure profile implementation integrates T(1−T) over normalized ray intervals; the accepted model adds −log(accumulation.clamp_min(1e−6)), warmup4096/ramp4096. The positive deficit is appropriate only for independently qualified opaque interiors, never arbitrary actor silhouettes/hair. Profile alone also permits empty rays. The joint depth-admission result does not isolate opacity causality.

Current selected semanticR0 has no deep-eroded black interior alpha<.5; alpha<.9 is1.6–4.4%, while visible exterior haze is large. Negative clear-ray feasibility therefore has priority; positive clothing remains a separate possible factor.

## Results

Original body80 samples819/4096 whole-valid rays,546 band and2731 core/FAS rays (after FAS onset the last quota splits). Outside actor confidence, RGB receives zero weight; existing distortion remains global. An unchanged sampler can already encounter clear rays; no resampling intervention is needed initially.

Across63 TRAIN cameras, existing source `valid & train_valid & ~supportW64 & ~UNKNOWN` labels81.78% of uniform-valid pixels clear. Exact calibrated normalized ray/AABB intersection leaves44.56%; under the819 whole-valid quota this predicts364.92 such rays per batch. This is a source-cache eligibility count, **not certified negative alpha GT**, and does not establish actual nonzero-opacity/gradient frequency. CE/JE/NE are entirely UNKNOWN and contribute zero. Existing policy covers reviewed mats/bag fragments/contacts; it explicitly does not certify complete visibility of other foreground clutter.

All26 proposed diagnostic rays have a clear9×9 patch, remain outside nativeRAW64 hair support and UNKNOWN, and have nonempty actual AABB intervals with min floor-coordinate z>.10. `negative_clear_candidates.png` shows coordinates and intervals. Manual inspection rejects MC/NC/OA equipment/people and several cable/stand patches. Root considered HA head(1067,113), AA shoulder(961,239), BA shoulder(954,175) unobstructed at the marker, conditional on floor being behind the box. Their intersections with z=0 occur respectively .7834/.4763/.4116 scene units **after** AABBfar. This uses the already accepted approximate floor frame; it is not a proof that arbitrary background rays are empty. IE lighting board and JB cardboard depth remain unknown. No loss mask is created.

A separate necessary positive-clothing screen accepts22 of36 existing nonreserved black-clothing observations (valid, black-only, core/W64=1, at least16HDpx from silhouette). Contacts show mostly solid fabric, but mask distance does not prove material or complete visibility. Reserved points—including5568, now interpreted as hair/chest overlap—are excluded from possible fitting. All22 remain unqualified for an opacity target until material review; no dense body expansion is inferred.

## Next evidence and limits

Root authorized a bounded read-only diagnostic on immutable selectedR0@111316:26 TRAIN candidates +12DC renderer-parity rays, FR.3, idle-exclusive GPU,8GiB allocator/10GiB process/900s, no optimizer. Actual native accumulation is required; sum of independently integrated weights alone is insufficient. Exact signed-floor-v2 helpers are imported by absolute path with observed path/SHA receipts; frozen checkpoint/config/schema2/source hashes are verified before/after. Diagnostic directory `C/opacity_train_alpha_v1`. First attempt failed closed before GPU/model because another healthy audit held GPU4's global lock despite0MiB visible usage; nothing was interrupted. A retry of the same immutable policy can occur after the lock releases.

If actual alpha is nonzero on the three conditionally qualified rays, a later separately reviewed loss-only experiment could minimize accumulation there. Its derivative is proportional to end transmittance and can vanish in opaque clumps; a clamped negative log(1−alpha) also has numerical saturation concerns. Actual gradients would need their own read-only check before choosing a formula. No global zero-alpha assumption, alpha-GT-from-mask, new sampler, hard occupancy deletion, geometry target or GPU training is authorized by this feasibility work.
