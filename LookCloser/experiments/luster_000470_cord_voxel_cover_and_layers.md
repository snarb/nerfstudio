# Frame 000470: complete cord occupancy and ray-layer audit

## What was tested

The best cord-specific frozen renderer used the sharp-28 ordinary-trained 33k
checkpoint, local ray sampling within `.005` of the train-fitted 3D cord line,
and permanent occupancy admission at radius `.003`. The original admission
marked cells containing 257 points on the segment and eight radial samples.
To check whether gaps between those samples caused the fading lower braid, a
forward-only screen conservatively admitted **every** grid cell whose bounding
sphere intersects the radius-`.003` line tube. This over-includes cells rather
than missing a cell touched by the estimated tube. The field and source images
were unchanged; no masks were used.

A separate read-only audit decomposed 138 held-out cam095 rays in a horizontal
sweep across the upper, middle and lower cord. It recorded termination mass
before, within `±.003` of, and behind the approximate 3D cord line, along with
the predicted RGB and RGB of the depth band alone. The 3D line comes only from
training-camera landmarks; it is an approximate centreline, not pixel-perfect
cord geometry or ground-truth depth.

## Results

The [voxel-cover receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_control33k_local_r005_full_voxel_cover/result.json)
reports 11 cells from sampled admission and 12 in the complete conservative
cover. Eight cover cells were occupied in the checkpoint, and all four missing
cells were already included in the sampled admission. The one extra cover
cell was therefore already occupied. The grid cells are approximately `.0234`
normalized scene units wide, much larger than this cord tube. Widening the
complete-cover tube to `.006` also covers the same 12 cells, as recorded in its
[separate receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_control33k_local_r005_full_voxel_cover_r006/result.json).
The complete cover changed **zero** RGB
pixels in the cam095 cord crop and left all eval metrics exactly unchanged:

| Renderer | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | cam095 cord SSIM | cam095 cord LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Sampled 11-cell admission | 19.824818 | .816858 | .220518 | 22.686262 | .786803 | .066941 |
| Complete 12-cell voxel cover, `.003` | 19.824818 | .816858 | .220518 | 22.686262 | .786803 | .066941 |
| Complete 12-cell voxel cover, `.006` | 19.824818 | .816858 | .220518 | 22.686262 | .786803 | .066941 |

The [GT/sampled/complete-cover comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_control33k_full_voxel_cover_comparison.png)
confirms the lower braid remains too faint. The `.003` and `.006` full-cam095
renders are pixel-identical. No extra occupancy initialization could improve
this frozen field at the same cells.

The [ray-layer receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_control33k_ray_layers/result.json)
shows that near the fitted line the ray is already fully terminated, but the
cord-depth interval does not carry all of its weight. For the closest sampled
pixel at row 1330, its band carries `.867` of the ray mass, `.092` lies in front
and `.041` behind. At row 1360, the corresponding split is `.804/.098/.097`
for band/front/back. The lower row has 1520 fine samples and a high band
density; it is not being skipped by occupancy. In 69 sampled pixels within
`.002` of the line and with band mass above `.5`, the native RGB gives
`21.85 dB` PSNR against GT, while the **conditional colour of the existing
depth band alone** gives `23.62 dB`. This is only a diagnostic counterfactual:
it is not a valid complete renderer or a trained result.

### Globally best 30.5k checkpoint

The same conservative cover was checked on the globally best sharp-28 30.5k
checkpoint. The [grid audit](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_30500_voxel_cover_audit/result.json)
finds 12 covered cells, eight occupied in the saved grid and four missing.
The existing sampled admission covers all four missing cells; its one omitted
cover cell is already occupied. The [full-cover render](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_30500_full_voxel_cover/result.json)
therefore tests the effect of admitting every missing cord-tube cell against
the original radius-zero render, with an unchanged field:

| Frozen 30.5k renderer | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Original local sampling | **19.98045** | .812109 | .225064 | **21.71084** | .757066 | .082832 |
| Complete voxel cover | 19.98037 | **.812127** | **.225046** | 21.69865 | **.762503** | **.080458** |

The [GT/baseline/full-cover crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_30500_full_voxel_cover_comparison.png)
shows no visible recovery of the lower strands. The modest SSIM/LPIPS changes
do not justify a longer occupancy warmup as the remedy for this artifact.

## Insights

Filling every cell intersected by the fitted cord tube was tested on both
checkpoints and does not restore the braid. A longer occupancy warmup is
therefore not a useful next experiment for this region. The remaining fade is
better explained by learned colour/depth mixing and insufficient local
contrast: appreciable weight sits outside the likely cord-depth band, and the
band itself is softer than the original braid. The geometric line and its
radius are uncertain around twists, so a stronger hard-depth prior remains
risky. Subsequent trained depth-band results and the actual termination-weight
audit are documented in [the termination-mass experiment](luster_000470_cord_termination_mass.md).
