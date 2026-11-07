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

Root independently checked all2371 bindings, actual parser/admission/config proofs and factory, then released and launched the trial. `root_training_review.json` SHA256 is `c59ce285243099c6793221ac7129f57fadeb1d61d039fc2f98f3abf837544868`. Controller2150533, supervisor2150540 and worker2150545 run on local olive GPU0, physical UUID `GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46`. The worker loaded all63 frozen maps and began cold updates with no OOM; initial Torch peak is3.48GiB. The first saved checkpoint at attempt2000 contains1999 Adam updates across all four field parameter states (SHA256 `f4e61827f212b0b85369d6b70e9fe60ea9ca5ee41a82cf8b8d90756fa62f3dd0`). Peak process RSS observed19.29GiB. A read-only observer initially asserted the HD count1998; its failure and corrected actual4K count are recorded in first_Adam_check.json, without changing the producer. Warmup and first visual gate remain pending. The external CPU archive watcher2151107 preserves evaluated checkpoints and evidence under `C/independent_occupancy/quarter_frozen4k_progress`; no checkpoint pruning is added. `manual_supervision.jsonl` records process, GPU, OOM and RAM observations.

## Insights and limits

This removes the structural-recipe mismatch from the HD-versus4K comparison while preserving the declared input bundle. Existing4K maps remain highly censored and legacy projection remains unchanged; higher pixel resolution does not guarantee finer learned detail. Required later gates are actual first Adam/warmup health,15188 three-view source/actor/detail comparisons, primary FHD PSNR/SSIM/LPIPS, and the unchanged stopping/frontier policy. No quality or geometry promotion follows from CPU feasibility.

Reusable [preparer](../scripts/prepare_marshawn_quarter_4k.py) and [external factory](../scripts/train_marshawn_quarter_4k.py).
