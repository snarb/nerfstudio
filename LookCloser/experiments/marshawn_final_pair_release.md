# Marshawn 1660 final pair: practical initial quality accepted

## What was tested

A matched cold native NGP pair uses scalar-corrected SDR RGB, unchanged camera
geometry, approved scene.loose AABB (volume 154.6652), native field/model,
max resolution 8192, hash table exponent 21, world step .003, no appearance embedding, and
all-view evaluation every 2000 steps. Both add observed-train-SfM LOS and
surface-band opacity terms, each weight .01; no RGB compositing or new neural
parameters. Radius 0 uses the native core; radius 1 applies exactly one
noncumulative halo after each native 16-step occupancy update. Every trial is
capped at 22,000 attempts. The actual helper source/probe is preserved under
`scripts/marshawn_final_pair_frozen/` with exact SHA bindings.

## Results

Fresh native HD evaluation of selected checkpoint 8000, over all three held-out
views (selection: all-eval PSNR, LPIPS tie-break within .07 dB):

| Policy | PSNR | SSIM | LPIPS | Conservative attempts |
|---|---:|---:|---:|---:|
| Native R0 | 27.532591 | .780458 | .359162 | ≤8170 |
| Training noncumulative R1 | 26.243208 | .762026 | .399108 | ≤8140 |

Both ended normally. R0 wins the defined quality selector. Witness-cell misses
are 36/9022 (13/495 body) versus 1/9022 (0/495 body). Ordinary inference halo on
R0 leaves two non-body high-density misses. The candidate `(actual R0 core
UNION observed train-witness voxels) -> ONE halo` adds 24 seed cells and only
79 cells over ordinary halo, admitting all 9022 witnesses/495 body points.
Its actual HD metrics are 27.543148 / .780200 / .358109, marginally above the
original R0 PSNR with comparable SSIM/LPIPS. BODY128 native free-prefix mass:
median 4.77e−9, p90 .001714, 0/128 above 10%; surface-band mass median .999132.
Original field/EMA/checkpoint restoration passed. Sparse admission is not
complete geometry certification.

All 24 pages for each stored R0/R1 policy and all 24 final candidate pages were
manually reviewed, covering every XYZ slice.
Peripheral admission remains geometrically unverified. Initially rejected
white novel-view regions now have actual peak-XYZ/source evidence: the same
real softboxes appear at those projections in 12/8/11 train views for path
03/06/21. Root and geometry manually inspected the source sheets. Preserve
that hardware rather than treating it as unexplained free-air fog; the geometry-selected .15 forward-standoff path passed manual review of all
24 views with recognizable opaque actors, mat and floor. Fine blur remains;
the original near-hardware interpolation path remains a preserved stress test.

- [Reproducibility README with exact load/render CLI](/fsx/tmp/marshawn_001660_hd/final_pair/README_REPRODUCIBILITY.md)
- [Current acceptance status and colour chain](/fsx/tmp/marshawn_001660_hd/final_pair/release_status.md)
- [Run summary / original metrics](/fsx/tmp/marshawn_001660_hd/final_pair/run_summary.json)
- [Actual CUDA source/callback/export/reload probe](/fsx/tmp/marshawn_001660_hd/final_pair/actual_cuda_probe.json)
- [Actual cloud-ray traces](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/novel_cloud_traces/audit.json)
- [Independent train-source softbox projections](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/cloud_GT_visibility/audit.json)
- [Inference candidate provenance](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/radius0/witness_union_halo1_candidate/receipt.json)

## Insights and next steps

The chosen loose box is the smallest sufficient candidate among tested bounds,
not a proven continuous optimum. Training-time halo improves sparse admission
but degrades this matched pair's held-out image metrics. An inference admission
correction can preserve the better R0 learned field; its actual
render/metric/state-restoration checks and 24 inner-view gate passed. The
selected R0 + protected halo is accepted as a practical initial result; fine
blur and unknown hidden geometry remain explicit limitations. Never dilate a
stored radius-1 active mask again or rewrite the original checkpoint.

Final native RGB is uint8 PNG from REDWideGamutRGB/Log3G10 TIFF decoded into
positive linear Rec.709, exposure/scalar corrections baked, then Reinhard and
sRGB. The float32 EXR master is linear; archival Rec.2020 PQ PNG16 and PQ-code
EXR surrogate have separate contracts. Final scalar-SDR frequency maps are
prepared and manually reviewed under `/fsx/tmp/marshawn_001660_hd/frequency_gain4_scalar`;
they are not HDR maps and are not consumed by this native pair. No ideal-grid
or complete hidden-geometry claim is made. Final package integrity is being
recorded; this is not an ideal physical-grid certificate.

Final evidence: [candidate HD/BODY128 comparison](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/grid_candidate_hd_comparison/comparison.json), [accepted inner 24-view manual gate](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/inner015_witness_candidate/root_manual_review.json), [all 24 candidate atlas coverage](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/radius0/witness_union_halo1_candidate/atlas_manual_review_coverage.json).
