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

## Actual native alpha and exact patch coverage

Read-only selectedR0@111316 replay passed in54.26s, peak1,758MiB.38 rays:26TRAIN candidates and12DC saved-render checks. Actual native RGB and accumulation agree with independent integration within1.91e−6/2.93e−6; all12DC RGB8/alpha16 codes differ by at most1. Frozen model/checkpoint/source/schema2 and actual imported helper paths/hashes are bound in `C/opacity_train_alpha_v1/summary.json`. No optimization occurred.

| Conditional clear TRAIN ray | Native alpha | Native samples | Floor z=0 beyond AABBfar |
|---|---:|---:|---:|
| AA(961,239) |.991806|331|.4763|
| BA(954,175) |.845896|386|.4116|
| HA(1067,113) |.399099|71|.7834|

Root accepted the three9×9 patches conditionally on the reviewed floor geometry. **Every one of243 pixels** passes valid/RAW64/UNKNOWN, positive interval, min interval floor-z>.10 and floor-after-far checks. Minimum floor margins across each patch are .4621/.3961/.7644. All have W64=0 and core=False, but unexpectedly **band=True**: therefore exact unchanged-sampler expectation includes both the546 protective-band rays and819 valid rays. Expected target hits are .050612/step (.048986 band +.001626 valid), or202.45 in4000 attempts across243 pixels. The whole-valid-only estimate would substantially understate coverage; neither estimate is dense training coverage.

Proposed paired continuation keeps main4096 body80 rays unchanged and adds a separately labelled64-ray auxiliary microbatch, own checkpointed RNG, only these243 targets. Source is completed selected semanticR0@111316, FR.3, exact full-state resume; model/semantic core/EMA/admission, RGB weights, LR and distortion.01 stay unchanged. Control performs the same auxiliary sampling/render computation with coefficient0. Diagnostic gates are500/1000/2000/4000 additional attempts; no quality-completion claim. New targets outside these patches need separate source review.

## Optical-thickness gradient diagnostic

Instead of saturated alpha or clamped −log(1−alpha), root proposed mean optical thickness τ=∫sigma dt through each qualified AABB segment. Uniform quadrature retains the fixed semantic selector but does not inherit learned occupancy skips or early stopping. This defines its own regularizer; it is not asserted equal to the native renderer's −log(1−alpha), and it is not a global density-elements penalty.

`C/negative_tau_gradient_v2` passed actual original4096 body80 versus64 predeclared clear rays at256/512/1024 samples,38.65s total/7,062MiB. Initial/final seven serialized-state hashes, sampler, cache, RNG, native core/EMA, optimizer and GradScaler match;0Adam updates. Actual import bindings and physical GPU UUID are recorded before checkpoint/model load. Main reference gradient L2 .0124921; raw τ gradient norms:

| Samples/ray | Mean τ | Unscaled gradient L2 | .001 norm/main | .01 norm/main |
|---|---:|---:|---:|---:|
|256|2.57222|17.80288|1.425|14.251|
|512|2.49976|17.11717|1.370|13.702|
|1024|2.51470|17.28274|1.383|13.835|

512→1024 changes mean τ by .60%; median per-ray relative difference1.22%, worst11.08%;93.75% are within5% or absolute.05. This remains approximate quadrature. At512 gradient cosine with main objective is .10769. Actual FP16 autocast is preserved. Main backward used source GradScaler scale16384; raw unweighted τ needed read-only diagnostic backoff to64. Every reported gradient was divided by its backward scale in FP32; actual GradScaler/optimizer state never changed. Coefficient .01 is therefore aggressive relative to this one measured main batch; .0001/.001 are proposed for review, not silently substituted or launched.

The first v1 harness omitted CUDA_VISIBLE_DEVICES, briefly loaded on defaultGPU0 and failed its strict CUDA RNG-state gate **before gradients/updates**. It is retained, not erased. GPU0 sourcepoint producer1158631/start3509741 continued110016→112000 with fresh progress, no OOM/CUDA errors and no diagnostic process left behind. Corrected v2 sets the leased physical UUID in both supervisor and child, verifies one logical CUDA0 maps to that exact UUID before model load, and guards process placement. Shared `marshawn_gpu_isolation.py` plus negative tests are required in the future paired worker; scientific frozen source was unchanged.

No diagnostic continuation or new objective training has been launched. Runnable pair preparation remains subject to root scientific review of exact source/config/full-state/auxiliary-RNG gates.
