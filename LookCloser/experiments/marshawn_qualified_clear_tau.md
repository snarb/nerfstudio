# Marshawn: source-qualified clear-ray optical thickness

## What was tested

Three fresh full-state continuations from completed semantic R0 selected step111316: auxiliary coefficient0, .0001, .001. The original actor RGB,4096-ray body80 sampler, W64 confidence, radius0, learned core/EMA, FR.3, distortion.01 and optimizer schedule remain unchanged. A separate private checkpointed CPU generator selects64 rays and512 stratified jitter samples per ray on every attempt. The objective is mean optical thickness through fixed semantic admission, independent of learned occupancy skipping and early termination.

Only243 TRAIN pixels are qualified: three reviewed9×9 patches centered at AA(961,239), BA(954,175), HA(1067,113). Every pixel is valid, outside RAW64 actor support and UNKNOWN veto, has zero W64 RGB weight, and intersects the AABB entirely above floor z=.10, with the observed floor behind AABBfar. This conditional source qualification is not a global free-space certificate. Unchanged body80 sampling supplies only .0506 such rays/step, motivating the separate auxiliary batch. No new alpha GT is inferred from actor-mask absence alone.

Generation: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/negative_tau_candidate_v1`. Its1503-file source inventory preserves all1499 previous semantic-fork files, including the original1495 semantic files. Parent checkpoint SHA `ec682be7c8205e9c4a27caf9cef7b0fcfab8fe09389e7bb2c59e1aee04057b8f`. Inventory SHA `c99503bd74188f50f39d0b924ca90a9a392a8880aad6e8256dd75efa218986e2`.

## Results

Fixed V1 FHD actor evaluation after4000 attempts (step115316); these are diagnostic gates, not plateau completion.

| Auxiliary coefficient | PSNR | SSIM | LPIPS | AA mean alpha | BA mean alpha | HA mean alpha |
|---|---:|---:|---:|---:|---:|---:|
| 0 |31.50350|.837740|.02227096|.97650|.83164|.57298|
| .0001 |31.51661|.837915|.02224934|.09027|.03071|.07690|
| .001 |31.51026|.837881|.02228515|.00228|.00043|.00611|

All per-actor3% regression guards pass. Three GT/parent/control/.0001/.001 visual contacts preserve gross hair, clothing and shoes at inspected scale, while broad exterior haze and interleg/floor clumps persist. Final LPIPS differences are negligible; .001 is not the final metric winner. [DC comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/negative_tau_candidate_v1/comparisons/step-000115316/DC_GT_parent_control_tau0001_tau001.png), [HC comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/negative_tau_candidate_v1/comparisons/step-000115316/HC_GT_parent_control_tau0001_tau001.png), [KA comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/negative_tau_candidate_v1/comparisons/step-000115316/KA_GT_parent_control_tau0001_tau001.png).

Zero-update gates verify all seven serialized state trees exactly, with explicitly bounded numeric renderer parity (RGB8/alpha16≤1 code). Technical32+32 attempt gates prove actual32+31 Adam updates, one common AMP skip, strict state/private-RNG resume, finite successful combined gradients and identical main-ray/private-auxiliary streams. Final4000 attempts retain identical main indices, auxiliary jitter chain, final global/private RNG and sampler state across all three arms. Actual successful Adam updates are3998/3998/3997; no skipped update is disguised as training. Historical private cache ordering was not serialized in the original producer; current reconstructed cache association is verified, not claimed historically observed. [Technical proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/negative_tau_candidate_v1/technical_verification.json), [final stream proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/negative_tau_candidate_v1/diagnostic_verification.json).

Independent final103 TRAIN clothing +12 DC geometry audit finds effectively stable clothing bands/widths and no obvious new hair/foot loss in inspected crops; this does not certify the whole grid. [Geometry report](marshawn_negative_tau_geometry.md).

Outside-target read-only audit replays26 old TRAIN diagnostic coordinates plus12 DC evaluation rays at500 and4000 for each arm. Excluding the three trained centers,23 untouched **unqualified** TRAIN rays have mean native alpha .38682 control versus .37453 for.001 at4000; maximum individual change .12218. These are behavior diagnostics, not additional free-space labels. Native alpha/RGB agree with independent trace integration within4.37e−6/3.30e−6; all72 DC replay samples match saved RGB exactly and alpha16 within one code. [Audit summary](/fsx/tmp/marshawn_001660_hd/tuning_20261106/negative_tau_outside_audit_v1/summary.json).

The audit supervisor's final all-file check failed on one generated helper `.pyc`. All12 GPU phases had completed; the failed receipt is retained. Independent CPU postflight verification confirms every actual source/input/checkpoint binding and all1503 scientific files unchanged, while explicitly recording changed bytecode hashes. This is an operational cache-binding failure, not an original whole-file PASS; no GPU rerun was performed. Future helper scopes exclude volatile bytecode and disable bytecode writes.

## Insights and next steps

Qualified negative optical thickness strongly removes local unwanted opacity without a measurable actor-quality penalty here. It does not establish global halo removal. The next reviewed hypothesis expands only the accepted AA/BA/HA source polygons to7369 filtered pixels under the same per-pixel predicates, comparing coefficient.001 on243 versus7369 targets with a matched coefficient0 control. New source validation and short resume proofs are required before that diagnostic is released. Original masks and frozen runs remain unchanged; no production recipe is promoted.
