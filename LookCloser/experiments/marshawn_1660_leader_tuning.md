# Marshawn1660: leader recipe, actor quality and occupancy

## What was tested
Campaign under implementation,2026-11-06. Primary objective: maximize the two fighters' detail and masked LPIPS; remove observed artifacts. No finite iteration ceiling. Every production arm completes historical gates through106316, FR1→.3 at75941, then validation every5000 until both signals plateau for four consecutive gates (Charbonnier relative improvement0.1%; masked LPIPS absolute improvement.001). Small improvements accumulate against the last significant best. No early quality stopping before both leader stages.

Accepted reference is `static_leader_reproduction_recipe.md`, not the obsolete from-scratch recipe. Full-frame control retains canonical LookCloser defaults including historical ARM, softplus, uncorrected SH directions, legacy frequency projection, B4096, hash23/max8192, stable occupancy,4096 fixed256 warmup, Charbonnier sqrt(error²+1e-4),distortion.01, FAS1/FR1 and Adam.01→.0001 over200000. After200000 LR remains.0001. Sparse-depth coefficient.001 remains configured but no depth target is supplied: it is inactive, not claimed supervision. Cameras are already calibrated/prepared and rigidly transformed into the accepted floor frame, so focus/up/auto-scale are replaced by identity parser normalization. New scene63train/3validation split replaces69camera historical data.

Box remains `configs/aabb_review/marshawn_1660_1900_floor.json`: min[-.84,-.21,-.07],max[.15,.42,.63] in floor frame. No enlargement to recover distorted historical floor density. RGB is the frozen corrected radial-k4 joint-scalar dataset,1920×1012 preserving aspect. Native hair-reviewed source masks are remapped with the same RAW prefilter/undistortion geometry as RGB. Both actors are separately scored; workers and background are excluded from scoring.

Matrix: full RGB/soft actor objective × historical sampling/body80 × occupancy radius0/1 (8runs). Then actor width32 vs64nativepx; coarse step.003125/.0125 vs.00625; distortion ramp.01→1 only on geometry diagnostic trigger; finally SSIM frequency threshold.93/.97 vs.95, optional neighbours±.01 after≥3%LPIPS gain. Final control/candidate repeat seed43. At most18 training runs, each to plateau; one additional zero-update structural selector.

## Results
Production campaign launched at approximately16:47UTC,2026-11-06. All eight matrix arms are training on local GPU0 and silver-sable GPUs0–6. Silver passed source/data hashing, matched Torch2.7.1/TCNN qualification and a two-update numerical comparison.42tests pass. Actual two-update full and actor/body80/halo1 technical smokes rendered all3validation views. Actual SIGTERM/checkpoint recovery resumed fromstep3 through99 and completed; these technical renders are excluded from quality selection. Remote CPU-only SSH/DAG smoke passed. Logs and receipts are under the campaign root.

Provisional gates inspected around17:40UTC (unfinished runs; these are not selected final checkpoints):

| Variant | Step | Actor PSNR ↑ | Actor SSIM ↑ | Masked LPIPS ↓ |
|---|---:|---:|---:|---:|
| Full / body80 / radius0 |75940|24.6313|.750811|.078250|
| Full / body80 / radius1 |60752|23.8364|.716098|.088802|
| Actor / baseline / radius0 |91128|24.8913|.716686|.091607|
| Extra distortion1 / full / body80 / radius0 |45564|21.4357|.608098|.129126|

The extra distortion1 arm runs on silver GPU7 through a separate durable controller, leaving the original campaign manifest unchanged. Its early launch was triggered by the independently measured false foreground density, not by a validation-mask fit. Stronger distortion worsens HC/DC appearance, concentrating density into floating sheets; it is not promoted. All arms retain the required full leader trajectory and plateau rule.

[Late HC/KA crop comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/online_review_late/late_comparison.jpg) was visually inspected: the best current LookCloser arm retains fog, while actor-only variants retain blurred/ghosted boundaries. Better actor scores do not certify clean geometry. A separate [historical native reference](marshawn_native_reference_actor_metrics.md) is sharper, but includes additional SfM/MVS supervision and is not an architecture-only control. Its semantic-grid cleanup is an auxiliary experiment.

Independent source review also found a stationary floor bag incorrectly included in several grey-actor masks. TRAIN-only corrected labels are being prepared separately for the auxiliary semantic grid; main campaign inputs remain immutable. Held-out annotation review is separately versioned and cannot supply training/gate constraints. Current v1 scores retain this known annotation limitation.

At approximately17:44UTC, silver's shared home filesystem exhausted the checkpoint reserve. Five workers stopped safely before replacing their last committed checkpoint; three were paused for relocation and then saved their live states. All4,853 campaign files were verified after copying to `/cache`, and the original `/home/.../marshawn_tuning_20261106` path now resolves there through a symlink. Original files/logs remain in a separate backup. All eight silver jobs resumed their full optimizer/scheduler/scaler/RNG state and advanced; the local control never stopped. The blocked dependency ledger was restored without recipe changes. [Recovery receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/recovery_20261106/silver_recovery_complete.json). Reporter path checks now resolve storage aliases;9 tests pass.

Prepared source masks were visually inspected in DC/HC/KA and AA/GE/NE. Intended actors remain visible; unrelated workers excluded, including foreground occlusion in GE. Data: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/data`.

## Insights and constraints
- Main score is neutral-composited fixed-mask/fixed-crop AlexNet LPIPS in Full HD, equal weight per actor and camera. It is not historical full-frame or strict native interior LPIPS. Report PSNR/SSIM/LPIPS; validation objective remains an internal stopping signal.
- Checkpoint selection: minimum masked LPIPS, within.001 choose highest actorPSNR then earliest step. A>3%LPIPS regression on either actor blocks automatic promotion, not evidence reporting. Final selection needs actual visual review.
- Homogeneous historical FAS interprets dataset camera bucket indices as shuffled cache positions. Baseline preserves this exact behavior, including RNG, and computes the confidence normalizer for that actual proposal. Body80 uses explicit physical-camera mapping. This is a recorded baseline limitation, not a silent global sampler fix.
- No density-derived or visibility-unknown cell is automatically certified free. Audits export actual GPU core/active/EMA, enforce exactly one halo, distinguish warmup and uncertain cells, and retain checkpoint/source hashes.
- Workers retain latest, FR-boundary and the current LPIPS-tie Pareto checkpoint frontier; all metrics/renders and checkpoint hash inventory remain. Atomic checkpoint state contains optimizer/scheduler/scaler/RNG, sampling and plateau state.
- silver-sable required a user-confirmed SSH host-key update. Its campaign environment now matches local Torch2.7.1+cu128 rather than using the pre-existing2.9runtime. dev3's2.4runtime is used only for the explicitly separate historical native render/audit path.
