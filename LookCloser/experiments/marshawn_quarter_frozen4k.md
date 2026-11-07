# Quarter-step/.1 with existing4K sources

## What was tested

Prepared one cold trial using the current HD structural recipe on the already frozen4K RGB/native-confidence/maps from `semantic_4k_production_v1`. No frequency map is refitted. The new configuration is quarter coarse step.0015625, distortion.1, R0, original W64/body80, seed42, legacy projection and the original LR/FR/plateau schedules. The old4K experiment used coarse.00625 **and distortion.01**.

| Comparison | Held fixed | Factors that differ |
|---|---|---|
| New4K versus current HD quarter.1 | Structural recipe, fixed admission, seed/schedules, FHD evaluation | Actual4K RGB, native-confidence raster and existing4000-update maps versus HD/1000-update maps |
| New4K versus old4K | Exact frozen4K inputs/maps, fixed admission | Coarse.00625→.0015625 and distortion.01→.1 |

Neither comparison is a pure resolution effect or a complete resolution×coarse factorial. The purpose is to test the available4K input bundle with the stronger structural recipe. The original4K dataset, original HD data, all masks/maps and existing runs remain unchanged.

## CPU proof and launched trial

Package: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1`. Policy SHA256 `5272460ee94a12d7d3c605ed9a3bc63fd04cdc251f3bb9c469dad1583065abfe`; CPU completion SHA256 `42099e01eef1176f72d8765251273526708ce49b77453b215d25434782b0e4fb`. All2371 bindings pass. The1495 quarter.1 scientific files are unchanged. Two existing4K source-validation helpers are copied byte-for-byte outside that source tree; a small external factory binds the datamanager to4K and keeps the model's semantic geometry source at the original HD dataset.

The actual frozen4K validator verified654 source files and189 map/metadata/censor files, the existing READY receipt, all63 maps, native RAW confidence widths, exact FHD photometry replay, and unchanged three-view EVAL RGB/mask/semantic bytes and crops. Actual CPU parser checks report63 TRAIN images at3840×2024 and3 EVAL at1920×1012, exact2× TRAIN intrinsics, unchanged camera poses/order and accepted AABB. The real admission loader succeeds against the explicitly separate HD geometry source: original semantic-grid SHA `e804d736dd766e8adf11df07e346a6317af87e0b106a94dfc524ab32f1690d71`, mapped boolean SHA `d60baa9186a63f3769204b93cb46dcfd2ea16e3d41a72ca788934fe19bd2659f`.

The actual proposed model/optimizer/schedule configuration equals the current HD quarter.1 configuration after normalizing only datamanager data, dataparser data and output directory. The model semantic-data path remains the original HD source in both. Normalized config SHA256 is `9c20499ea14a440e5a71aab84f53d4ecf563a74a602eb65627d7033fd6d3aeb1`. The CPU preparation created no model or optimizer; the subsequent root-released trial starts cold with no loaded checkpoint.

The pre-existing4K technical proof records actual cache/sampling, frequency update1024 and32 resumed Adam updates. Its CPU cache peak RSS was17.51GiB; the current preparation observed approximately102GiB available local RAM. These are inherited mechanism/resource observations, not a new GPU test of the finer-step interaction. The candidate requires20GiB available RAM before caching, a physical UUID-bound global lease on local olive GPU0, and a10GiB Torch allocator cap. Other workers remain on silver. The original loop runs through minimum106316 and then automatic plateau without a fixed22k cap; FR switches at75941 to.3.

Root independently checked all2371 bindings, actual parser/admission/config proofs and factory, then released and launched the trial. `root_training_review.json` SHA256 is `c59ce285243099c6793221ac7129f57fadeb1d61d039fc2f98f3abf837544868`. Controller2150533, supervisor2150540 and worker2150545 run on local olive GPU0, physical UUID `GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46`. The worker loaded all63 frozen maps and began cold updates with no OOM; initial Torch peak is3.48GiB. The first saved checkpoint at attempt2000 contains1999 Adam updates across all four field parameter states (SHA256 `f4e61827f212b0b85369d6b70e9fe60ea9ca5ee41a82cf8b8d90756fa62f3dd0`). Peak process RSS observed19.29GiB. A read-only observer initially asserted the HD count1998; its failure and corrected actual4K count are recorded in first_Adam_check.json, without changing the producer. The6000 checkpoint confirms5997 Adam updates and a nonzero128³ frequency grid spanning0..15 after the4096 warmup; `warmup_check.json` binds its bytes. The first15188 visual gate is complete below. The external CPU archive watcher2151107 preserves evaluated checkpoints and evidence under `C/independent_occupancy/quarter_frozen4k_progress`; no checkpoint pruning is added. `manual_supervision.jsonl` records process, GPU, OOM and RAM observations.

## First matched15188 gate

| Input/recipe | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| HD quarter/.1 | 32.09120 | .854067 | .02219093 |
| Existing4K coarse/.01 | 31.29313 | .836767 | .03006208 |
| New4K quarter/.1 | 31.67569 | .852289 | .02321341 |

All3 actor and9 hair/hand/feet panels were reviewed. The new4K recipe is visibly less diffuse around body/hair than old4K, but has no compelling FHD detail advantage over matched HDquarter/.1. Gross actor shapes remain, while substantial halo and floor/mat/bag fragments persist. The improvement over old4K combines coarse-step and distortion changes; the comparison to HD changes the declared4K input bundle. Neither establishes a pure-resolution effect. [Actors](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000015188/all_actors.jpg), [details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000015188/all_details.jpg). The savedalpha16 exterior diagnostic (outside a16HD-pixel Euclidean margin of fixed V1 actor masks) shows new4K alpha>.1 area is6.23%/3.87%/3.12% lower than HD in DC/HC/KA. Old4K is47.74%/25.17%/30.77% larger than HD. This indicates modest exterior reduction versus HD alongside worse actor LPIPS; it is not a physical geometry or complete sole-preservation certificate. The worker remains healthy and continues; no promotion follows from this early gate.

## Insights and limits

This removes the structural-recipe mismatch from the HD-versus4K comparison while preserving the declared input bundle. Existing4K maps remain highly censored and legacy projection remains unchanged; higher pixel resolution does not guarantee finer learned detail. Required later gates are actual first Adam/warmup health,15188 three-view source/actor/detail comparisons, primary FHD PSNR/SSIM/LPIPS, and the unchanged stopping/frontier policy. No quality or geometry promotion follows from CPU feasibility.

Reusable [preparer](../scripts/prepare_marshawn_quarter_4k.py) and [external factory](../scripts/train_marshawn_quarter_4k.py).


## Root matched30376/45564 visual review (2026-11-07)

| Step | Recipe | PSNR | SSIM | LPIPS |
|---:|---|---:|---:|---:|
|30376|HD quarter / distortion .1|32.01860|.847094|.02053743|
|30376|Old4K coarse / distortion .01|31.46806|.843863|.02720744|
|30376|New4K quarter / distortion .1|31.63265|.845332|.02180955|
|45564|HD quarter / distortion .1|31.87417|.842044|.02000830|
|45564|Old4K coarse / distortion .01|31.51175|.845015|.02515922|
|45564|New4K quarter / distortion .1|31.60984|.839385|.02101009|

Root manually inspected all three actor views and nine detail crops at each gate. The new structural recipe substantially improves over old4K, but remains below matched-step HD on actor LPIPS. Both4K variants contain considerable unwanted geometry; no clean asset or resolution advantage is established. RGB, frozen maps and confidence footprints change together against HD, so this is not resolution-only causality. Training continues to minimum106316 and subsequent original plateau protocol.

[30376 actors](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000030376/all_actors.jpg), [30376 details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000030376/all_details.jpg), [45564 actors](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000045564/all_actors.jpg), [45564 details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000045564/all_details.jpg). Each directory contains a hash-bound root_manual_review.json.


## Further root matched-step gates

| Step | Recipe | PSNR | SSIM | LPIPS |
|---:|---|---:|---:|---:|
|60752|HD_quarter01|31.81075|0.837931|0.01947274|
|60752|old4K_coarse001|31.49613|0.844755|0.02367900|
|60752|new4K_quarter01|31.46883|0.835004|0.02054359|
|75940|HD_quarter01|31.71675|0.835111|0.01927048|
|75940|old4K_coarse001|31.57082|0.844172|0.02287375|
|75940|new4K_quarter01|31.38063|0.831393|0.02077134|

Root viewed all3 actor comparisons and9 hair/contact/feet crops at both gates. Large floor, bag, between-leg and silhouette artifacts persist; no clean geometry or promotion. New4K remains worse than matched HD and its75940 LPIPS is worse than60752. Neither observation changes the original stopping criterion. Each gate directory contains provenance and a hash-bound root_manual_review.json.


## Root91128 review

| Recipe | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
|HD quarter / distortion .1|31.69281|.834074|.01902171|
|Old4K coarse / distortion .01|31.54728|.843935|.02246367|
|New4K quarter / distortion .1|31.35041|.829843|.02065490|

All3actor and9detail crops were manually inspected. No gross new limb/shoe loss is visible at contact scale, but substantial unwanted geometry remains; new4K still trails matched HD on actor LPIPS. Training continues to the original minimum and plateau protocol. [Actors](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000091128/all_actors.jpg), [details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000091128/all_details.jpg).


## Root minimum and postminimum4K gates

| Step | Recipe | PSNR | SSIM | LPIPS |
|---:|---|---:|---:|---:|
|106316|HD_quarter01|31.66471|0.832548|0.01890305|
|106316|old4K_coarse001|31.53144|0.843568|0.02205500|
|106316|new4K_quarter01|31.31533|0.826771|0.02036358|
|111316|HD_quarter01|31.63683|0.832166|0.01898528|
|111316|old4K_coarse001|31.55453|0.842771|0.02178174|
|111316|new4K_quarter01|31.30169|0.826354|0.02051743|
|116316|HD_quarter01|31.63524|0.831819|0.01892447|
|116316|old4K_coarse001|31.56483|0.843412|0.02176120|
|116316|new4K_quarter01|31.29074|0.826119|0.02049429|

All3actor and9detail crops were manually reviewed at each gate. New4K retains substantial unwanted floor/bag/silhouette geometry and remains below matched HD on actor LPIPS. No new resolution advantage or promotion. The original post106316 plateau protocol remains active; all per-gate root receipts and provenance are retained.


## Terminal4K decision

The new4K trial reached the original plateau at126316, without an artificial22k cap. Root inspected all3 actor views and9 details through terminal126316; the raw-best106316 and selector45564 renders had also been reviewed.

| Ranking | Step | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
|New4K raw best|106316|31.31533|.826771|.02036358|
|New4K historical selector|45564|31.60984|.839385|.02101009|
|New4K terminal|126316|31.27167|.824878|.02052030|
|HD raw best / matched terminal|126316|31.63973|.831264|.01877312|

Keep the HD structural recipe for now: this frozen4K RGB/map/confidence bundle does not improve the primary FHD actor metric and still contains substantial unwanted floor/bag/halo geometry. It does not isolate resolution alone or evaluate native4K heldout quality. The original plateau/provenance/checkpoint inventory and separate raw/selector rankings are preserved in `C/quarter_frozen4k_candidate_v1/root_final_summary.json`. [Terminal actors](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000126316/all_actors.jpg), [details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/quarter_frozen4k_candidate_v1/reviews/step-000126316/all_details.jpg). The GPU is released for further actor-isolation experiments; the overall optimization remains active.
