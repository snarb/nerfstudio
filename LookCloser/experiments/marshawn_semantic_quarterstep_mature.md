# Mature quarter-step source qualification

## What was tested

Immutable evaluated-intermediate quarter checkpoint91128, while its producer continues. Exact original semanticR0/W64/body80, coarse.0015625, legacycap1024, FR.3, original global2048 frequency update and distortion.01. No producer or occupancy mutation.

## Results

PSNR31.452810, SSIM.828438, LPIPS.01934326. Checkpoint SHA `81a226cdc96a437b221d9891ab8eac5bb9aa4ca5e5179ee03f92ceb16002c9a9`. Seven full-state hashes, four nonempty Adam entries (step91087), schema2 core/EMA/active, source/config/spec/release and renders are protected under `SC/independent_occupancy/quarter_91128_anchor`.

All103 TRAIN clothing and12 DC diagnostic rays converge at8192/16384 and match native rendered pixels. Native witness ±.02 bands for old18/new66/old8reserve/new9reserve are84.914/84.931/90.079/95.237%; preceding mass3.768/3.433/5.462/.776%. These are local diagnostics, not a whole-scene physical occupancy certificate.

Actual native cap hits0/115; cloth max981/1024 is close to the cap, DC max512. The all-cells diagnostic alone saturates1/103 cloth ray, so its truncation is not called a native issue. Dense tail mass beyond the native final interval is at most.001685 cloth/.001109 DC. Final selector/rawmin audits are queued after natural completion, with automatic native cap/tail reporting; an actual native hit would require a separate cap2048 counterfactual, not an automatic producer change.

## Visual inspection and limits

Personally inspected DC black, HC grey and KA black GT/source body contacts. Clothing is sharper and gross hair/shoes remain visible, while broad colored halos, floor remnants and the KA bag fragment remain. This qualifies an immutable numerical source for controlled later experiments, not a clean asset or automatic τ release.

Evidence: `C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`, `C/independent_occupancy/quarter_91128_geometry/{clothing103,DC12,quarter91128/sampling}`. All3 source contacts are stored alongside. The final selector/rawmin audit supervisor is `SC/independent_occupancy/semantic_cold_quarterstep_final_geometry` and does not stop or cap training.

## Natural completion and final choices

The cold run completed naturally at126316. Historical selection and plateau logic were unchanged; all three choices are protected separately.

| Choice | Step | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Established selector |45564|31.610453|.835400|.020019663|
| Raw LPIPS minimum |111316|31.427557|.827178|.019231721|
| Terminal |126316|31.432397|.827229|.019268997|

Raw-min checkpoint SHA `542e97f59ec1fd41452b0a04fde6bd832e0b430773d1989eef2ec062ef3f1a34`; actual schema2 NPZ SHA `18a767edf9ae2d25bcd1d73cca762b2e08f85110a7e452684a35840740c5fbea`. Raw-min black/grey LPIPS .020379021/.018084421, PSNR30.135038/32.720077, SSIM.790154/.864203.

Read-only final audit covers both selector and raw-min, all206 clothing+24DC rays, exact native traversal, single evaluation halo and dense8192/16384 convergence. Native pixel parity and convergence pass. Native cap hits are zero for both choices: selector cloth/DC maxima746/339; raw-min941/570. Dense tail maxima are .002500/.002226 and .000842/.001008 respectively. Three raw-min all-cells diagnostic rays hit1024; this does not indicate native truncation. No cap change was made.

| Group | Selector band mass % | Raw-min band mass % | Selector median10–90 width | Raw-min width |
|---|---:|---:|---:|---:|
| Old18 |84.6160|85.1648|.031038|.028278|
| New66 |85.0403|85.0180|.019825|.016358|
| Old8 reserve |90.4495|90.0423|.018305|.017591|
| New9 patch reserve |94.4265|95.9901|.017306|.013960|

Bands are ±.02 around measured source depths. Narrower termination does not prove uniformly more accurate surfaces: median absolute residual old/new/reserve-old/reserve-new changes .000796/.001711/.000583/.001111 to .000798/.001876/.000581/.001330. Material5568 remains an annotated hair/chest overlap, not unqualified cloth truth.

Personally inspected all three final GT/R0/selector/raw-min/terminal body contacts. Detail improves over R0, gross feet/hair are present, while broad colored halos, floor remnants and KA bag remain. At16HD pixels outside the fixed actor mask, alpha>.1 counts DC/HC/KA are R0 185261/107305/74830, selector192394/112431/77076, raw-min190316/111420/76436 and terminal189578/111208/76215. Total exterior alpha is lower for quarter, but its low-opacity footprint is wider. This diagnostic is not new empty-space truth.

Evidence: `C/independent_occupancy/semantic_cold_quarterstep_final_choices`, `semantic_cold_quarterstep_final_geometry_v2` and `semantic_cold_quarterstep_final_exterior`. The first final-audit watcher failed before any GPU query because it expected generic sidecar names; the retainer uses step-named sidecars. Failure evidence remains in the v1 folder; v2 fixes only lookup, with3CPU tests, and uses unchanged checkpoint/grid/source hashes. Final geometry is numerically qualified as a structural comparison base, not certified clean occupancy.
