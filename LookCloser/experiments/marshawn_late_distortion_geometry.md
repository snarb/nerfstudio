# Late distortion: independent matched 64752 geometry audit

## What was tested

The completed full-RGB native-exp pair forks the same evaluated checkpoint60752 and runs4000 additional attempts: distortion coefficient.01 versus1. Both retain corrected SH, exp(h+1), body80, native R0, legacy frequency projection and FR1. Neither uses semantic admission or auxiliary geometry labels. Fleet verification confirms identical initial state and sampled pixel-ray streams; successful Adam updates are3999 and3998 because GradScaler skipped different overflow attempts.

This independent auditor performs no optimizer update or training callback. It protects both final checkpoints by immutable hardlink, exports their **serialized actual GPU core capture**, and verifies saved owner/sampler grid aliases, active masks, policy and capture freshness. It never reconstructs core from CPU EMA thresholds. Actual core/active/EMA are hashed with the checkpoint. CPU regressions reject mismatched aliases, stale capture and wrong active state; a deliberately contradictory EMA fixture confirms the saved core remains authoritative.

The same sealed103 TRAIN clothing pixel rays and12 post-hoc DC failure/control rays are replayed with native traversal, one-cell evaluation dilation, all cells, uniform8192 and16384 samples. Camera rays and actual field activation/SH/FR are checked against the saved source. The diagnostics preserve the old/new source grouping, including whole-patch reserves, but **none of those groups is auxiliary supervision for either arm**. Point5568 remains hair/clothing overlap; HA5556 is qualified opaque grey clothing. DC pixels have RGB evidence, not triangulated depth truth.

## Results

| Matched64752 arm | Masked PSNR ↑ | Masked SSIM ↑ | Masked LPIPS ↓ |
|---|---:|---:|---:|
| Control.01 |26.86503|.792562|.05485945|
| Treatment1 |30.66083|.833477|.03705419|

Both KA actors regress: black PSNR30.017→29.736 and LPIPS.02828→.03428; grey PSNR33.023→31.263 and LPIPS.02110→.03481. DC/HC improve strongly. Visual inspection confirms cleaner clothing and less front fog, but large background sheets and clumps remain. No full-scene quality promotion follows from the aggregate metric.

[Inspected GT/control/treatment actor crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/late_distortion_matched64752/GT_control_treatment_crops.jpg).

| Predeclared TRAIN group | Control native mass within witness±.02 | Treatment native mass within witness±.02 | Control→treatment median10–90 depth width |
|---|---:|---:|---:|
| Old18 |87.437%|95.962%|.011681→.006108|
| New66 |89.075%|96.834%|.018540→.000358|
| Old8 reserves |97.181%|99.987%|.007161→.004510|
| New9 reserves |80.854%|100.000%|.006654→.000161|

For HA5556, native median depth minus the witness projection improves from−.318972 to−.000898; its10–90 width changes from.350387 to.000166. This resolves the previously measured front sheet at that qualified clothing observation. Quantiles use unconditional ray mass and exact exponential inversion inside stored constant-density intervals. Extremely small widths do **not** prove correspondingly fine physical surface resolution; sampling, triangulation and pixel-ray uncertainty remain.

All206 clothing rays and24 DC rays pass uniform8192/16384 RGB/opacity convergence checks. The largest clothing band-mass component difference is.003632. Native DC replay matches saved PNGs. For diagnostic pixel939,263, max RGB error falls.1677→.0287; for1024,379 it falls.1419→.0367. Pixel901,380 changes median depth.61925→1.07009 and RGB error.1677→.0131. These are relative field/RGB comparisons, not DC depth truth.

Treatment new66 **dense** integration still has19.635% front mass versus3.166% under native traversal; evaluation halo remains3.166%. The strongest differences occur on black trouser-lettering observations8173/8679/9020 and grey shoulder8931NC. Learned density in skipped space is not certified physical occupancy. Conversely, the old8 and new9 reserve gains survive dense integration:99.987% and100% in the±.02 band.

[Complete native/halo/dense report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/late_distortion_matched64752/comparison.json), [signed quantiles and widths](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/late_distortion_matched64752/depth_residuals.json), [DC diagnostics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/late_distortion_DC12_matched64752/comparison.json), [mass comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/late_distortion_matched64752/native_mass.png).

## Insights and provenance

Late distortion improves measured surface placement as well as RGB on these clothing witnesses, unlike the early cold distortion1 arm. Remaining background geometry and KA regression prevent promotion on mean LPIPS. A separately controlled semantic-admission fork can test whether late distortion retains this benefit while limiting incompatible background supervision; this audit does not launch that experiment or alter any producer.

Protected remote roots are `SC/independent_occupancy/late_distortion_{control001,treatment1}_064752`, where `SC=/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106`. Checkpoint SHAs:

- Control: `0e6952376a408744318413f67e9d9ee1fd255bc5777b8a79fece5b23b5387b27`.
- Treatment: `3e422b4b699cab13c42a92811aee3144c06634502f66640c65e7caca7d85ca11`.

The sequential read-only auditor used silverGPU5 under the authorized10GiB ceiling with identity, progress, memory and OOM checks every5seconds. Existing prefixP producer562989 remained healthy; its optimizer, RNG, source and global GPU lock were untouched. The separate read-only lock was released after all four traces. Timings are not performance benchmarks. Full raw trace arrays remain beside the protected remote checkpoints; source hashes and health receipts accompany the local reports.
