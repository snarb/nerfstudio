# R0 8k TRAIN surface color versus volume composition

The read-only audit used the existing52soft TRAIN photometric MVS candidates in MA/NA/NB/LB:208rays, current radial PNG5×5means, unchanged native field and actual grid. No evalRGB, SGD, new target, carving or color averaging between views. Exact checkpoint is bound in the audit.

Normal±.005 band contains at least.999762 actual termination mass (median.999998); opacity≈1. Native volume RGB versus band-conditional RGB differs by4.22e−6 mean / .000151 max. Consequently out-of-band tails do not explain the RGB mismatch in this TRAIN subset. Captured weights/colors reconstruct nativeRGB to5.30e−8.

| Channel MAE(range1) | Native volume | Direct MVSXYZ | Weighted band |
|---|---:|---:|---:|
| R | .010448 | .008259 | .010447 |
| G | .008906 | .007594 | .008907 |
| B | .010889 | .010287 | .010887 |

DirectXYZ is closer on114/208rays, while blue bias persists. Its overallMAE is2.222/255 versus2.571/255 native. Source±1pixel footprint range median.00702 is comparable to these errors. Field RGB changes by.0119/.0261 at normal−.001/+.001; the very bright+.005 color has medianσ.000190 and is mostly unobserved air. These measurements do not uniquely identify capacity, BRDF or sample position as the cause. Soft MVSXYZ and5×5pixel means are not infinitesimal color ground truth.

Both actual six-worst/six-typical four-view TRAIN contact sheets were inspected; all view-dependent colors remain separate. [Full audit](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/MVS52_TRAIN_colour_8k/summary.md), [worst contacts](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/MVS52_TRAIN_colour_8k/worst6_TRAIN_contacts.jpg), [typical contacts](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/MVS52_TRAIN_colour_8k/typical6_TRAIN_contacts.jpg). Cache retains actual sample RGB/σ/weights, exact partial-band weights, sourceUV and world-ray primitives. No global floor or clean-quality acceptance is implied.
