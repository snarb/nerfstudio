# Physical-ray comparison of HD and experimental4K

## What was tested

Both models are frozen at15188. The4K arm combines native4K TRAIN RGB/confidence with4000-step frequency-map fits; it is not a pure resolution comparison. Fixed HD-derived semantic geometry and the three FHD evaluation views remain unchanged.

The103 sealed TRAIN rays retain their original origins/directions and witness annotations. Derived4K coordinates are exactly `2*(HD_index+.5)-.5`; no rounding to a4K pixel is allowed. The original query JSON remains sealed and the derived query has a separate hash. Twelve DC evaluation pixels remain unchanged. Actual model camera rays agree with the original geometry within9.69e-8. Both source/weights/schema2 captures and actual field flags are checked.

The first audit generation discovered an integer-coordinate helper shadowing the derived helper. It stopped before replay; a second generation was interrupted to adopt the same stronger absolute-file import and pre-model path/SHA receipt as the signed-floor auditor. These audit-only attempts are retained; the4K producer, checkpoint and source files were never changed. Completed v3 passes CPU imported-module identity and actual per-replay helper bindings.

## Results

| First15188 | PSNR ↑ | SSIM ↑ | Masked LPIPS ↓ |
|---|---:|---:|---:|
|HD reference|31.41439|.841056|.03117262|
|Experimental4K|31.29313|.836767|.03006208|

| Source group | Native witness±.02 mass HD→4K | Median10–90 width HD→4K |
|---|---:|---:|
|Old18|80.23→68.94%|.03433→.05782|
|New66|83.95→81.03%|.01928→.03501|
|Old8 reserves|84.48→74.97%|.02668→.05296|
|New9 patch reserves|95.95→89.25%|.01139→.02265|

All206 clothing rays pass8192/16384 convergence; all24 DC rays converge and match their respective saved FHD PNGs. The coordinate derivation therefore does not explain the geometry regression. The sparse source evidence is still uncertain, and point5568 retains its hair/clothing-overlap annotation.

Manual DC-black,HC-grey and KA-black crop review finds local garment-detail differences while broad exterior haze, hair-adjacent floating material and near-foot floor islands persist. Improved mean LPIPS does not establish better depth geometry; no promotion follows from this gate. The already authorized training trajectory continues unchanged.

[Clothing report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_4k_physical_15188_v3/clothing103/report.md), [signed widths/depths](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_4k_physical_15188_v3/clothing103/depth_residuals.json), [DC replay](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_4k_physical_15188_v3/DC12/report.md), [GT/4K/HD crop](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_4k_physical_15188_v3/DC_black_gate_comparison.jpg).
