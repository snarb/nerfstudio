# Training-time semantic admission candidate

## What was tested

One isolated change to the frozen native-exp actor/body80/R0 parent: fixed, conservative TRAIN semantic admission during density evaluation and traversal. SH correction, exp(h+1), frequency projection, LR schedule, sampling, AABB, seed42 and mandatory106316→plateau remain unchanged. No geometry prefix or positive-band objective is added.

The source is the independently reviewed occlusion_v2/W64 pad1 grid from63TRAIN cameras. Whole-cell mapping retains any source overlap, boundary contact and semantic UNKNOWN. It preserves103 witness XYZ/pixel-depth points and both qualified shoe anchors; source whole-TRAIN-ray coverage and source/frame hashes are sealed. DC/HC/KA never contribute admission labels. The original learned occupancy EMA and core are saved separately: active is native noncumulative dilation (or native all-active binary warmup) intersected with fixed admission.

A separate inference-reloadable model subclass gates the native common position selector. Therefore fixed256-sample warmup, adaptive ray marching, `get_density`, `query_points`, inherited `density_fn` and evaluation use the same support. Native encoding, activation and admitted-point derivatives remain intact. The primary model defaults and running snapshots are unchanged. Snapshot schema2 stores actual learned core, EMA, admission and active separately; skipped cells are not certified physically empty.

## Technical results

Generation1 was preserved after a zero-update diagnostic failed on an empty native parameter-gradient tensor. Generation2 changes only that diagnostic and its shape/dtype/empty regression checks; production model/worker files are byte-identical. Ten CPU tests pass, including brute-force whole-cell overlap, XYZ/contact/outside UNKNOWN, density and gradient gating, all-ones identity, source/bounds tampering, core/halo separation and empty-gradient mismatches.

| Actual native GPU check | Result |
|---|---|
| Full4096-ray all-ones warmup RGB/alpha difference |0 /0|
| Full4096-ray all-ones ARM RGB/alpha maximum difference |2.38e-7 /4.17e-7|
| Largest parameter-gradient difference warmup/ARM |1.19e-6 /2.74e-6|
| Forbidden-point density_fn/query_points |Exactly0 in both paths|
| Warmup successful Adam updates before→after resume |16→32|
| ARM successful Adam updates before→after resume |15→30|
| RNG restore |Exact recorded state digest in both phases|
| Technical rendered DC/HC/KA |Finite complete rasters in both phases|
| Peak monitored candidate process memory |7,948MiB|

ARM smoke starts cold at step4096 only to exercise the postwarmup code path; it is not a learned4096-step checkpoint. GradScaler legitimately skipped one update in each16-attempt ARM segment (32768→16384 across resume); successful Adam states and finite moments are verified on both sides. There is no quality claim from these32-attempt runs.

The root-approved technical overlap lasted189.7s. The independent production worker remained alive and progressed82639→88301 without OOM. Its global GPU lock and files were untouched. The candidate used a separate overlap lock,10GiB allocator cap and12GiB process-memory ceiling; that lock was confirmed free afterward. These concurrent timings are not performance benchmarks.

[Complete technical receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/technical_complete.json), [GPU identity/path proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/native_probe/probe.json), [warmup resume](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/verification_warmup.json), [ARM resume](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/verification_ARM.json), and [visually inspected three-view technical montage](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/technical32_overview.jpg).

## Insights and launch state

Technical gates pass; initial renders contain broad actor silhouettes and retained foreground/mat regions. This is consistent with conservative semantic UNKNOWN and does not establish good geometry. Production remains separately guarded by an exact root release, binding19 artifacts and1495 frozen source files. The concrete single-job plan queues a cold actor+admission arm on the first idle GPU under the common global lock; production overlap is prohibited. Compare matched geometry rays, DC failures, both actors, hair/feet and per-view PSNR/SSIM/LPIPS before any promotion.

Frozen root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106`; remote counterpart under silver-sable `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106`. Source manifest SHA `f428fdc2527d189cec319001aced76f9de7b7a6caae5b5a1e8702106b5ab434c`; support manifest SHA `fce7fc24735cd62d5242ce719964c5d6d61291feef1cc1c536c12f2ce194b15e`.

Root production review approved the exact19-artifact release (`c97566fb2ab559aa1c22839e81373637f4c1ba9015a2c9ed07611f84d75d42ab`). Controller676019 launched worker718446 on silverGPU7; independent watcher691770 retains the first15188 checkpoint/grid/renders before replaying the sealed103TRAIN and12DC diagnostics. Its semantic-aware adapter uses schema2, keeps fixed density admission in every profile and computes native halo from the saved learned core before intersection. Six per-view/actor metric records and DC replay/PNG parity are preserved; DC diagnostic pixels never become TRAIN labels.

A separate R1 control was prepared at `semantic_actor_r1_prepared_20261106`. Actual resolved config differs only in native occupancy radius0→1 and output/support path relocation; source pad1 bytes are identical. Its only source-code change is exact-spec validation of radius1. Preparation and config-diff receipts required R0's first15188 visual/geometry gate plus separate release before launch; both are now recorded below. Source SHA `804b6858155537565b38080c17304037e19bbb64e3094b2d35ff2281c4981f5a`; spec SHA `08de9623c26c78d49fec04b33b21b399be574aa7c7cb5b01f6348b6fae59f298`.


## First production visual and geometry gates

At15188, actor+TRAIN admission fixes the catastrophic DC clothing holes seen in the actor-only parent. Both actors and their shoes are coherent in all three views. Hair/background halos and floor islands remain visibly fuzzy; these outputs are not a clean geometry or occupancy certificate.

| Semantic R0 step | Masked LPIPS ↓ | Masked PSNR ↑ | Masked SSIM ↑ |
|---|---:|---:|---:|
|15188|.03117262|31.41439|.841056|
|30376|.02741222|31.46039|.843210|

DC black PSNR at15188 improves from actor-only17.5784 to30.9556dB. The full six view/actor records are retained. [GT/15188/30376 visual comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_actor_progress/step-000030376/GT_15188_30376_crops.jpg).

The independent103-ray replay uses actual saved core/active/admission and keeps the fixed density support in native, halo, all-cell and uniform8192/16384 profiles. Native before/±.02/after/transmission mass for old18 is8.21/80.23/10.07/1.49%; new66 is5.34/83.95/9.15/1.57%; old8 reserves9.76/84.48/5.10/.66%; new9 reserves1.09/95.95/2.79/.17%. HA5556 grey-clothing median depth bias improves to+.001155 and10–90 width.010305, versus the full parent's−.32018/.35776. Five of66 new-source rays do not reach90% opacity, so their t90 remains unknown. Point5568 is still annotated hair/clothing overlap. Reserved groups were never geometry supervision for this arm, but remain TRAIN RGB evidence.

The12 diagnostic DC rays match saved PNGs. At pixel939,263 the actor-only front sheet had depth.54490/maxRGBerror.74427; admission gives1.00444/.05886. This is relative field/RGB evidence, not DC depth ground truth. All103+12 convergence checks pass; no optimizer or training callbacks run in the auditor. [Geometry and signed-depth report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_actor_matched15k/report.md), [DC diagnostics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_DC12_matched15k/report.md).

The authoritative15188 schema2 grid has51894 actual core/active cells and131911 mapped admission cells. Source pad1 has105049 cells; conservative whole-cell mapping deliberately admits additional overlap/contact cells across slightly different float32/float64 bounds. Do not compare these grids as if their cell boundaries were identical. The evaluation watcher preserves each newly evaluated checkpoint by remote immutable hardlink and fetches grid, metrics and all3 RGB/alpha rasters every60seconds until completion. Training continues through106316 and the existing plateau rule.

## R1 native proof and separately released cold control

R1 zero-update proof uses protected R0@15188 weights explicitly; it is not an R1-trained result. Actual native dilation of51894 saved core cells followed by the unchanged admission produces92486 active cells, exactly matching independent26-neighbor dilation. Repeating from the same fresh core is noncumulative. Core, EMA, parameters and gradients remain untouched. All208 source XYZ/pixel-depth/sole-anchor checks are admitted and remain in native positive cells.4096 actual TRAIN-camera ARM rays produce finite RGB/alpha.

The root-approved proof took20.7seconds with1798MiB peak process memory, while the existing R0 producer progressed43864→44649 without OOM. No speed claim. [Specific R1 proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_r1_prepared_20261106/R1_native_proof.json), [resource receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_r1_prepared_20261106/technical_complete.json). R0 all-ones and successful Adam-resume receipts are explicitly inherited because the model/gating code is unchanged; they are not relabeled as R1 training.

Root release SHA `d05d8eebff8bda73a92270fe6af92a72159123b6de49a74f3d933df7a4ffad5d` authorizes the separate cold R1 controller846189. After a clean shared-lock retry, worker870564 started on silverGPU1; the first verified progress was811 steps with no OOM. Evaluation-retention watcher2080625 uses a distinct R1 archive, and auditor863243 waits for the matched15188 geometry gate. Native radius1 is the only recipe change; fixed source W64pad1 remains identical.

A W32 loss-margin arm is also prepared, not queued, at `semantic_actor_w32_prepared_20261106`. Only the existing actor confidence map and its analytic normalization changeW64→W32. Body80 pools/FAS, nativeR0, fixed W64-derived admission and all model parameters remain unchanged. Actual dataclass config differs only in output/support path relocation; runtime width difference is recorded separately. SourceSHA `bc7fb84aa228c603d46c594c307eaa8deb291d54ae038934eeb28b04de8afabc`; specSHA `0e7ccecb31af0012ce6860583f2bcd0a2fafe089f87627f0dd293051bfd3fba6`.

Root independently exercised the actual frozen SamplingAdapter on all63 cameras for W32. Pools/FAS match, W32≤W64, and no actor-union pixel becomes newly zero-weight. Mature analytic proposal mean weights are.67779185 versus.68931994. No GPU/model/optimizer was created. This does not protect unsegmented hair automatically; visual hair/feet checks remain required. [Runtime parity receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_w32_prepared_20261106/runtime_width_parity.json).

W32 exact production release SHA `48e09e73e46589304eb84c8a79aeed41ac0beed2005a1c541513c1e9f4b4d4b3` was verified against1495 source files and15 artifacts. Controller887206 queues the cold arm under shared exclusive GPU locks. Evaluation archive/fetch watcher2082758 and matched15188 auditor890739 are active; the comparison retains W64/R0, W64/R1 and W32/R0 as separate trained models. Hair, shoes and exterior halo must be reviewed visually before any promotion.


## Matched15188 R0 versus R1

The actual cold R1 gate has56362 learned core cells and96622 active cells, versus R0's51894 active cells, with identical fixed admission and bounds. CheckpointSHA `8ca64dcbc486488f234731b3349b04949d0f8b3b1c72dfc6fe3de3b1fea6f714`. R1 PSNR31.42941/SSIM.836184/LPIPS.03053236 is a marginal LPIPS improvement over R0, with worse SSIM. Visually, shoes and hair remain present but exterior fog persists and appears broader in places. [GT/R0/R1 crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_actor_r1_progress/step-000015188/GT_R0_R1_15188_crops.jpg).

The independent matched replay shows worse early geometry: native witness±.02 mass changes R0→R1 from80.23→66.78% for old18,83.95→79.53% for new66,84.48→70.66% for old8 reserves and95.95→89.20% for new9 reserves. Old8 front mass rises9.76→25.16%; its median10–90 depth width grows.02668→.06216. HA5556 remains near its witness, but width grows.01031→.02110. All206 clothing rays and24 DC rays pass convergence and saved-PNG parity checks. R1 is not promoted from the small mean-LPIPS gain; its approved trajectory continues. [R0/R1 geometry report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_r0_r1_matched15k/comparison.json).

W32 subsequently acquired silverGPU3 naturally, without dispatcher interruption. Worker914736 and supervisor914729 were alive at the first813-step check, with7868MiB process memory and no OOM. The fixed W64-derived admission remains unchanged.

A separate read-only source check confirms6522 manually qualified stationary-bag false grey pixels remain in the original TRAIN semantic files and body80 core pools; W32 and W64 give them nearly unit confidence. The auxiliary admission source uses corrected supports and treats qualified bag/shell regions as UNKNOWN, not certified free. Narrowing the loss margin therefore cannot remove these particular positive labels. MC/NA temporal TRAIN images show the bag remaining fixed after the shoe leaves; source hashes and six-camera weight counts are in [the bag-weight audit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/bag_loss_weight_audit/report.json). No production data was edited; a potential correction needs a separately controlled source and shoe/shell review.

## Matched15188 W32 margin

W32/R0 yields PSNR31.33553/SSIM.833060/LPIPS.03108392, essentially unchanged LPIPS and worse SSIM than W64/R0. Its actual active/core count rises51894→81913. Viewed GT and all three variants for both actors in DC/HC/KA: hair and shoes remain visible, while broad exterior haze persists. [Six-actor comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_actor_w32_progress/step-000015188/GT_W64R0_W64R1_W32R0_15188_crops.jpg).

The103-ray geometry gate also regresses. W64→W32 native witness-band mass is80.23→65.75% (old18),83.95→76.34% (new66),84.48→68.94% (old8 reserves),95.95→84.64% (new9 reserves). Old8 early mass rises9.76→28.99%; median10–90 width grows.02668→.06194. HA5556 median bias remains small(+.0000713), but width grows.01031→.02376. Two old18, six new66 and one new9 rays do not reach90% opacity, so their t90 is unknown. All309 clothing and36 DC comparisons pass convergence and replay/PNG parity. These are measured early results; the released W32 trajectory continues, without promotion. [Comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_w32_matched15k/comparison.json).

## R0 completed trajectory

R0 stopped naturally at131316 under the unchanged plateau rule. Its recorded selector chooses111316: PSNR31.51845,SSIM.838073,LPIPS.02233986. The terminal LPIPS is.02196161; terminal and selected checkpoints are distinct and both are preserved. Viewed all six GT/30376/selected111316 actor crops: modest detail improvement, but broad exterior haze and floor/bag islands persist. [Final selected comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_actor_progress/selected111316_GT_30376_crops.jpg), [exact completion/selection receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_actor_progress/run_complete.json). This does not certify a clean asset or ideal occupancy.

The artifact watcher now retains surviving metrics/renders/actual-grid evidence even if a dominated checkpoint was pruned before its poll. It explicitly records checkpoint_retained=false rather than inventing weight retention. R0@121316 has this limitation; first15188 weights remain protected in their separate first-gate archive, and selected111316 weights are retained. Three CPU tests cover retained versus pruned weights, idempotence and hash mismatch rejection. Production models, checkpoints and plateau decisions were not modified.

Selected111316 download: `downloads/semantic_actor_000111316_review.zip` under
the campaign root,2617493bytes, SHA256
`15b4a3e3a4f29ad1335953494a2c49f76fed23fb65c3330d313bf53e95bde9a0`.
It contains unchanged three-view HD RGB/16-bit alpha, metrics, accepted box and
review receipts. ZIP entries were checksum-verified; no weights are included.
The README explicitly discloses remaining halos and floor/bag fragments.


## Completed selected R0 versus R1

Both naturally completed trajectories select111316. R1 stops126316 and gives PSNR31.58028/SSIM.837760/LPIPS.02247381 versus R0's31.51845/.838073/.02233986. The new final comparison replays actual native radius0/1 and FR.3, preserving schema2 core/active/EMA/admission and exact checkpoint/source hashes. Both idle-exclusive audits perform zero updates; all206 clothing and24 DC convergence/PNG checks pass.

| Source group | Native band mass R0→R1 | Median10–90 width R0→R1 |
|---|---:|---:|
|Old18|80.026→64.942%|.02962→.07208|
|New66|83.124→79.897%|.02646→.04160|
|Old8 reserves|84.218→71.118%|.03480→.05928|
|New9 patch reserves|98.757→93.007%|.003224→.010905|

Old8 early mass grows3.214→24.202%. Dense16384 replay also regresses: old8 band81.219→72.933%,new9 96.163→92.760%; therefore this is not solely an inference-grid skipping effect. HA5556 remains close to its witness but its width grows.002227→.007296. Three new66 rays in each model have unknown t90. Manual DC-black,HC-black,KA-grey comparison confirms both still have broad exterior haze, hair-adjacent floating material and floor fragments. R1 is not promoted from its slight PSNR advantage. W32 continues naturally and its actual completed selection will be replayed separately by the durable supervisor; the early15k audit is not substituted for final evidence.

[Final selected source geometry](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/R0_R1_clothing103/report.md), [signed depth/widths](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/R0_R1_clothing103/depth_residuals.json), [DC parity](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/R0_R1_DC12/report.md), [HC selected images](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/HC_black_selected_comparison.jpg).


## Completed W32 selection and final three-arm verdict

W32 stops naturally146316 and selects126316: PSNR31.55804/SSIM.836564/LPIPS.02288901. This is a comparison of each trajectory's recorded selection, not a claim of identical step counts: R0/W64 and R1/W64 both select111316. All three use FR.3 at replay. Exact selected checkpoint hashes, schema2 captures, all frozen source files and native radius/width are bound in the protected archives. Each idle-exclusive two-part replay takes about86.5seconds and1758MiB, performs zero updates and releases its lease. All309 clothing and36 DC rays pass convergence and native-PNG parity.

Relative to W64/R0, W32 native witness-band mass changes80.026→76.803% old18,83.124→85.822% new66,84.218→79.960% old8 reserves and98.757→95.465% new9 reserves. Old8 front mass increases3.214→10.488%; dense reserve-band mass also worsens81.219→74.065% old8 and96.163→94.244% new9. Median10–90 widths change.02962→.03402/.02646→.01654/.03480→.03677/.003224→.004711. W32 has one old18 and six new66 unknown t90 values. Actual selected active-cell counts are48509 R0/W64,96515 R1/W64 and47487 R0/W32, with the exact same131911 fixed admission and cell bounds; counts alone are not geometry accuracy.

Manual inspection of both actors in all three views finds some narrower W32 shoulder/torso haze, with visible hair and shoes retained. Rough floor islands and hair-adjacent floating material remain in all three models. W32's mixed source geometry and slightly worse perceptual score do not support global promotion. The final evidence favors retaining nativeR0/W64 as the comparison baseline. This remains an imperfect bounded asset, not an ideal occupancy certificate.

[All selected geometry](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/clothing103/report.md), [signed residuals and widths](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/clothing103/depth_residuals.json), [DC rays](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/DC12/report.md), [GT and all three selected DC actors](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/DC_black_three_selected_comparison.jpg), [HC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/HC_black_three_selected_comparison.jpg), [KA](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_selected_geometry/KA_black_three_selected_comparison.jpg).
