# DEC5 wrist: increasing TSDF voxel scale does not repair the void

## What was tested

Actual time 001037, cached 62 fixed-pose geometric depth maps. Rerun the existing
CUDA TSDF fuser with voxel sizes 0.0005, 0.00075, 0.001 and 0.0015. Keep truncation
0.004, extraction weight 2, original activation mode, depth truncation 4,
crop and component filtering unchanged. No new PatchMatch, RGB processing,
pose optimization, person masks or model-default changes enter these controls.

The new fine control matches the earlier control's image statistics, parameters,
normalization and vertex/triangle counts. Its binary mesh hash differs; this is
not advertised as byte-identical reuse. All scale comparisons use fresh outputs.

## Results

| Voxel | Vertices / triangles | Finding |
|---|---|---|
| 0.0005 | See hash-bound metadata / 147,416 | Large wrist/forearm void remains. |
| 0.00075 | See metadata / 62,124 | Moving-view gap persists; details become coarser. |
| 0.001 | See metadata / 33,462 | Gap remains in train H/A and moving views. |
| 0.0015 | See metadata / 13,744 | Coarse fingers and contour, no continuous forearm. |

All four complete with 62 identical train image statistics and matching
normalization. Each supervised stage spans about 10 seconds including the
controller's polling granularity; do not interpret that as precise kernel time.
No raw Open3D volume is serialized—only extracted meshes and metadata.

The main agent inspected the three moving-view scale comparisons and the H/A
0.001 comparison. Additional E/C panels are retained but not claimed reviewed.
No textured RGB run or image-quality metrics were warranted by this failed
geometry gate. The optional RGB control helper is available but was not run.

- [0.00075 moving view](/mnt/data/dec5_forearm_tsdf_scale/001037/geometry_review/moving_medium.png).
- [0.001 train view](/mnt/data/dec5_forearm_tsdf_scale/001037/geometry_review/H004_A005_1210M6_coarse.png).
- [0.0015 moving view](/mnt/data/dec5_forearm_tsdf_scale/001037/geometry_review/moving_coarsest.png).
- [Frozen requests/results](/mnt/data/dec5_forearm_tsdf_scale/001037/complete.json).

## Insights

The large missing wrist region is not remedied by this voxel-size range at
unchanged measured inputs and truncation. This does not prove all TSDF settings
fail; it rejects coarse voxel spacing as the proposed local cure. Reliable
new depth/shape evidence is needed before replacing fine geometry. Continue
with the separate [calibrated neural stereo canary](dec5_foundation_hand_stereo.md).
Production meshes, camera path and the 150-frame movie are unchanged.
