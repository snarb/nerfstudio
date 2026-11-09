# MVS floor candidates versus unchanged fields

52 train-only HD plane-sweep candidates, observed from MA/NA/NB/LB, were compared against PNG 13k, floor-prior PNG 14k and canonical HDR 6k near=0.052. All 208 actual calibrated rays per model were traced through the native renderer and independently integrated over the entire AABB at 8192 and 16384 intervals; the 16 worst convergence cases also used 32768. No weights, grid bits, source images, geometry constraints or poses changed. Full model-state hashes remained identical.

| Actual checkpoint | MVS cells absent from native core /52 | Dense q50 normal error, median | Dense mass ≥0.5 within ±0.005 /208 | Within ±0.01 /208 |
|---|---:|---:|---:|---:|
| PNG 13k | 19 | −0.00532 | 104 | 107 |
| Floor-prior PNG 14k | 0 | +0.00338 | 125 | 182 |
| HDR 6k near=.052 | 0 | +0.00606 | 79 | 191 |

Units are normalized prepared-world units. Positive error is above the MVS candidate along the fixed upward normal; this is **normal height**, not ray-distance error. Band ray halfwidth is normal halfwidth / abs(normal·ray). Median dense q10/q50/q90 normal heights are respectively: PNG13k −.00253/−.00532/−.00891; PNG14k +.00846/+.00338/+.00037; HDR6k +.00672/+.00606/+.00535. These quantiles describe accumulated first-termination mass rather than a hard surface.

PNG13k has a genuine deep field branch: q50 p10 is −.17288, appearing from all four actual train cameras. Native/full-AABB comparison gives 108/107 rays with majority mass inside ±.01; admitting missing cells alone cannot fix this branch. PNG14k removes most deep geometry but has a thicker layer above MVS: ±.005 median prefix mass .310, band mass .679. HDR6k is much thinner but its layer is usually above MVS: ±.005 median prefix .970, band .030; ±.01 median band .999. This is a remaining geometric discrepancy, not evidence that HDR already exactly matches the candidate depths.

8192→16384 maximum absolute opacity/band-mass differences: .000447/.000744/.004224 for PNG13/PNG14/HDR6. Worst-16 16384→32768 maxima: .000267/.000247/.000817. Dense results converge sufficiently to resolve the large geometric differences. Native versus dense ±.005 mass differences can reach .0718/.0701/.4615; HDR's very thin layer near a diagnostic band boundary is sensitive to native .003 quadrature, but dense integration still places its layer above MVS.

The representative profile image was actually inspected: candidate25 shows PNG near MVS and HDR about +.01; candidates97/123 show PNG13's deep branch, PNG14's broad upper layer and HDR's thin upper layer. Density evidence supports testing local measured depth support in preference to assuming all remaining color error is BRDF, but does **not** show that depth adjustment alone removes mottling. There is no new color comparison or optimization in this measurement.

The 52 candidates share overlapping patches and four cameras; they are not 52 independent physical measurements. LB was withheld from two-view fitting and used as a selection-validation gate, so it is not an unseen test. Their NCC peak widths are not calibrated confidence intervals. The ±.003/.005/.01 bands are explicit diagnostics only. Known sanity errors were .000085 and .002510 in normal height (respectively .000147 and .006129 along the reference ray); the crease sanity failed third-view validation and stayed rejected. No source-camera visibility implies a global free-space certificate. No training constraints or carving were generated.

Artifacts: [/fsx/tmp/marshawn_001660_hd/floor_plane_sweep_HD_PILOT/field_comparison/comparison.json](/fsx/tmp/marshawn_001660_hd/floor_plane_sweep_HD_PILOT/field_comparison/comparison.json), [all 208-ray metrics per field](/fsx/tmp/marshawn_001660_hd/floor_plane_sweep_HD_PILOT/field_comparison/all_ray_metrics.csv), [inspected profiles](/fsx/tmp/marshawn_001660_hd/floor_plane_sweep_HD_PILOT/field_comparison/representative_termination_profiles.png). Per-model audit JSON binds checkpoint/config/helper/MVS hashes and stores every ray/projection/near limit and band integral.
