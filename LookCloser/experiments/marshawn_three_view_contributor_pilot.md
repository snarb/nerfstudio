# R1 selected20k: frustum guard and three-view contributor pilot

## What was tested

Read-only R1 step20000, immutable checkpoint SHA `fd56e541d91abde26b776ffe03b91caf997b3c0ada0044cebf6407728c988fa2`. All38 pipeline tensors are bitwise identical to the earlier gate20 HD checkpoint. Field, EMA and original stored radius1 mask were restored exactly after each reader; model-state SHA `9d5791d6f4fdde0e1fff4ef3f93629589d56b5161939ff56ca274c0a24617366` is identical in all three completed reader receipts.

First, intersect the stored mask with the conservative calibrated-camera support plus exactlyONE voxel halo. Second, collect actual packed termination weights on DC/HC/KA FULLHD pixel-center rays from the original stored mask. For each ray, retain a sample if `w >= 1e-4/N`, with N its actual original sample count. Discarded original mass is accumulated in FP64 and asserted below1e−4 for every ray. No GT RGB is used to choose samples. Construct the pilot as `stored_active & ONEhalo(contributorMIDcells | known9022cells)`. Stored R1 is never redilated.

## Results

| Policy | Active cells | DC mean samples | HC mean samples | KA mean samples |
|---|---:|---:|---:|---:|
| Original stored R1 | 1,887,128 | 770.18 | 854.09 | 490.50 |
| Frustum-support+1 intersection | 1,409,772 | unchanged | unchanged | unchanged |
| Three-view contributor pilot | 118,273 | 339.42 | 410.45 | 220.45 |

The frustum intersection has **exactly identical full packed interval/index hashes and per-pixel sample counts** to the fresh original-mask pass for all3 views. Float RGB differs by at most1.67e−6, consistent with GPU accumulation variation. No physical-empty claim is made for excluded space: it is outside the input cones and remains unknown.

The contributor union has53,712 midpoint cells and1,881 distinct known witness cells; their ONEhalo covers128,582 cells before intersection. The118,273-cell candidate removes93.73% of stored admission. It preserves all9022 known points and495 BODY point cells. Ninety-four contributor midpoint cells lie outside the original binary mask; they are reported rather than treated as physical occupancy truth. The neighborhood union retains original admitted neighbors, but this is not a separate proof of complete geometry.

Actual maximum discarded original mass is1.18116e−5 across5,829,120 rays, below the predeclared1e−4 bound. Per-ray N and discarded mass arrays are preserved. Candidate full packets differ because samples are omitted. Actual all3 candidate RGB maximum errors versus the fresh original are DC1.192e−6, HC1.967e−6, KA.954e−6: **zero pixels exceed1e−4**.

Own-domain candidate metrics are PSNR30.808124, SSIM.817456, LPIPS.260500; common OLDscalar metrics31.170195/.826837/.254102. Differences from the original are numerical noise. The whole3-view contact and native KA floor crop were actually viewed: no observed image degradation, and the existing gray/brown floor mottling remains. This pilot does not improve actor or floor appearance.

## Insights and pending work

The frustum intersection passes the measured3-view phase/RGB check. The contributor candidate passes a **three-view numerical RGB pilot**, not a full-scene safe-grid release. A stronger retained-original-interval/BODY128 probe failed before model initialization because the older frozen BODY helper contains Python3.11-only `grid[0,*ij.T]` syntax under DEV3 Python3.8. Thus retained significant-interval equality and actual BODY128 candidate ray preservation are **pending**, not claimed from point membership or output metrics.

The task was then superseded by the user's actor-only AABB/masking priority. All reader PIDs142862/143390/143683/144018 are terminal; DEV3 has no active CUDA process. No compatibility retry, full66 bake, review-path bake or new frequency worker was launched. No full66 proposal is approved. The11-map frequency chunk remains separately complete at16/63, with47 unqueued; preparation/launch of a next chunk is deferred.

- [Four-way actual3-view contact](/fsx/tmp/marshawn_001660_hd/frustum_R1_20k_HD_and_3view_contributors_PILOT/all3_GT_original_frustum_contributor_PILOT.jpg)
- [Native KA floor comparison](/fsx/tmp/marshawn_001660_hd/frustum_R1_20k_HD_and_3view_contributors_PILOT/KA_floor_native_original_contributor.jpg)
- [Per-view RGB/sample comparison](/fsx/tmp/marshawn_001660_hd/frustum_R1_20k_HD_and_3view_contributors_PILOT/HD_pixel_and_sample_comparison.json)
- [Candidate and exact source bindings](/fsx/tmp/marshawn_001660_hd/frustum_R1_20k_HD_and_3view_contributors_PILOT/contributor_candidate_receipt.json)
- [Actual weight-mass evidence](/fsx/tmp/marshawn_001660_hd/frustum_R1_20k_HD_and_3view_contributors_PILOT/baseline_contributors_packed/receipt.json)
