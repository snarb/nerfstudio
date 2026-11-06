# Native exp(h+1) with reviewed clothing geometry

## What was tested

This separate candidate extends the frozen corrected-SH `structural_exp_hplus1` full/body80/R0 recipe. The two selected arms are P: expanded84 visible-prefix coefficient .003; PB: the identical prefix plus opaque-band coefficient .001. Both reuse the sealed31-point/84-ray source and the same whole-patch reserves reviewed for the [softplus geometry arms](marshawn_sh_geometry_arms.md). No images, masks, calibration, AABB, occupancy policy, feature projection, sampling recipe, distortion schedule, optimizer or validation selection changes.

The field is the parent's actual `trunc_exp(h+1)` implementation. Geometry queries call its `density_fn`, preserve the native clipped exponential backward, and integrate positive density in FP32. No exponential wrapper is applied to softplus output. Prefix and positive-band sampling use independent private streams; adding the band does not change prefix samples or RGB RNG. Band half-width remains max(.005,.01×depth), clipped strictly inside the accepted AABB. These prepared-world distances are not certified meters.

Candidate root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/exp_geometry_candidate_20261106`; silver equivalent: `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/exp_geometry_candidate_20261106`. Its `code` directory copies the frozen structural parent and adds nine candidate modules. All1485 existing files are byte-identical; the1494-file source inventory is hash-bound. Full dataclass configuration comparison differs only in the module identity of the identical scene-box parser.

## Results

Preparation checks:14 focused CPU tests pass. They cover the true native exponential forward and clipped backward, prefix/band gradient signs and finite differences, independent sampling, unchanged global RNG, activation/AABB rejection, and the production gate binding a completed exp parent. The22 newly fitted points' available leave-one-source-view shifts fit their actual clipped positive bands; old9-point checks retain the earlier source receipt. These are consistency checks, not independent confidence intervals.

The actual4096-ray zero-update probe passed at AMP scale65536. Model hashes before/after matched; no optimizer or parameter-gradient state changed. Geometry-gradient ratios below use the hash encoding plus geometry MLP; the RGB color head is excluded.

| Term | Coefficient | Norm / RGB geometry norm |
|---|---:|---:|
| Prefix | .001 | 2.01% |
| Prefix | .003 | 6.02% |
| Prefix | .01 | 20.07% |
| Positive band | .0003 | 2.79% |
| Positive band | .001 | 9.29% |
| Positive band | .003 | 27.88% |

All12 combined gradients were finite. The predeclared .003/.001 pair yields a combined norm1.0069×RGB, so it is retained. Peak allocated probe memory was8.11GiB within the12GiB cap. Both objective paths completed32 attempted iterations with a durable pause after15 and resume through31. Adam contained successful step14 before pause and step30 after resume; moments, parameters and actual full-resolution RGB/opacity outputs were finite. AMP scale settled to16384 after two initial skipped updates, demonstrating why attempts alone cannot validate optimizer continuation. Private CUDA prefix and band samples reproduced exactly from saved policies without altering global RNG.

The [GT/P/PB technical overview](/fsx/tmp/marshawn_001660_hd/tuning_20261106/exp_geometry_candidate_20261106/technical32_overview.jpg) was visually inspected: all six outputs contain the expected initial opaque fog inside the bounded volume, with no blank/failed raster. At32 attempts this is solely an integration check, not a quality result. `verification_P.json`, `verification_PB.json`, `probe_P/probe.json` and `probe_PB/probe.json` retain complete evidence. Both suites finished and released silverGPU0.

**Production is conditionally authorized, not yet launched.** The release gate requires the completed structural-exp production parent, its root-bound completion hash, a root-approved exact spec, frozen source inventory, actual gradient probe and successful Adam-resume evidence. Coldseed42 follows the unchanged mandatory106316-step trajectory and subsequent validation plateau; there is no fixed iteration ceiling. The conditional queue uses any free GPU0–7 after the existing five-job queue receives dispatch precedence, under shared locks.

## Insights

The motivation is measured rather than architectural speculation. At matched15188, actual native traversal in the exp parent concentrates more shirt-ray termination near the verified garment depth than the softplus parent. Separately, adding the positive band to expanded softplus geometry raises new whole-patch reserve ±.02-band termination from75.58% to87.60%. Combining them is a controlled next question, not a quality claim.

The [103-ray matched audit](marshawn_clothing_geometry_matched15k.md) also exposes residual fog and surface bias: C's fitted median termination is about .002 in front of the recovered witness and its fitted10–90% width remains about .009–.011. New reserve mean width is .098 because some rays retain large tails. A large ±.02-band mass is therefore insufficient evidence of a thin, correct cloth surface. The new arms preserve this audit, including signed depth residuals and width, rather than narrowing supervision from unmeasured assumptions.

## Conditional launch and independent audit

Root authorized the separate conditional policy `conditional_launch_policy.json` (SHA256 `852e54910048c50583fd890e71bbb749a3d6c47f4b172efa553a1776070c4c08`); local watcher PID2063616 checks it every60 seconds. Actual launch still requires the exact exp parent to finish its mandatory106316 ancestry and plateau, selected v1 LPIPS≤.06, and each actor≤1.03×the fixed SH15188 reference. Six original actor/view metric records and their exact evaluation domain are required. The existing five follow-up jobs receive dispatch precedence; P/PB then use any free GPU0–7 with the common locks. No worker is interrupted.

The policy binds24 artifacts, including source/spec/probe/resume receipts, the two-job plan and a separate reporter registry. The reporter includes the new `exp_geometry` source but has no geometry-approval receipts. Technical launch explicitly grants neither geometry approval nor model promotion. Five additional release-policy tests plus four saved-ray distribution tests passed.

The authorized audit process preserves both actual15188 checkpoints, replays all103 predeclared rays through native/halo/dense8192+16384 paths and reports signed median-depth residuals and unconditional10–90% width. Its read-only overlap is capped at10GiB with producer health checks; an idle fallback requires the global GPU lock. A separate selected-parent audit waits for the real completed selection, preserves its own checkpoint/completion hashes, and sets FR from that checkpoint's step. Neither audit changes training or launch eligibility.

## Matched15188 production result

The conditional guards passed on the completed native-exp parent, and both cold arms dispatched under shared GPU locks. Their15188 checkpoints were protected before pruning. The automatic103-ray audit finished with unchanged checkpoint hashes, native/all-cell/single-halo profiles,8192/16384 convergence, signed depth medians and10–90% widths. Producer P progressed21776→23333 and PB remained healthy through18000; no optimizer or training callback ran in the auditor.

| Arm | PSNR | SSIM | LPIPS | Old18 fitted band mass (%) | New66 fitted band mass (%) | Old8 reserved band (%) | New9 whole-patch reserved band (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native-exp parent |27.1029|.798674|.056944|86.56|90.37|97.59|82.41|
| P expanded prefix.003 |26.6139|.794221|.061572|88.75|94.73|96.39|83.84|
| PB prefix.003 +band.001 |26.8551|.795347|.059599|97.75|99.55|96.10|82.88|

PB concentrates termination at fitted surfaces: new66 median signed depth residual+.000054, median10–90% width.00710. Whole-patch reserves do not show a corresponding improvement. Qualified HA5556 still has front bias−.2874 and width.3637 (parent−.3202/.3578; P−.3014/.3610). The5568 material annotation remains separate and all sealed queries stay unchanged.

[Complete trace comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/exp_geometry_matched15k/comparison.json), [signed depth/width results](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/exp_geometry_matched15k/depth_residuals.json), and [GT/parent/P/PB montage](/fsx/tmp/marshawn_001660_hd/tuning_20261106/exp_geometry_review_15188/GT_parent_P_PB.jpg) were reviewed. Full-objective fog and background sheets remain visible in all three evaluation cameras. Both15188 RGB metrics and reserved geometry prevent promotion; training continues under the unchanged mandatory106316→plateau rule.
