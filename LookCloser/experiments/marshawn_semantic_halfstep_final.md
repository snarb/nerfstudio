# Semantic half-step final detail and geometry tradeoff

## What was tested

Cold semantic R0/W64 with only coarse step .00625→.003125, original frequency maps/global update, body80 sampler, .01 distortion, legacy1024 cap. Natural completion131316; historical selector60752 and raw-LPIPS minimum126316 remain distinct. Terminal131316 is retained and visually checked; GPU audits query only the two decision candidates.

## Results

| Model | Step | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| R0_W64 | 111316 | 31.51845 | 0.838073 | 0.02233986 |
| half_selected | 60752 | 31.64736 | 0.838522 | 0.02049509 |
| half_rawmin | 126316 | 31.49708 | 0.833326 | 0.01961766 |
| cold01_terminal | 126316 | 31.65173 | 0.837686 | 0.02135285 |

The half-step raw minimum improves LPIPS4.28% relative to its historical selector, with PSNR lower by.15028dB and SSIM lower by.00520. The original selector and plateau were not changed. Baseline/cold.1 rows reuse previously audited checkpoints; ages and FR values differ explicitly.

| Diagnostic group | R0 band% | Half selected band% | Half rawmin band% | Cold.1 terminal band% |
|---|---:|---:|---:|---:|
| old_prefix_supervised | 80.026 | 82.354 | 82.467 | 91.585 |
| new_prefix_supervised | 83.124 | 84.254 | 84.603 | 96.139 |
| old_local_reserved | 84.218 | 88.713 | 86.668 | 92.540 |
| new_whole_patch_reserved | 98.757 | 97.233 | 98.394 | 99.998 |

All206 new clothing and24 DC rays pass8192/16384 convergence and native-render parity. Actual native cap saturation is0/115 for each half-step checkpoint; maximum cloth/DC counts589/396 at60752 and905/663 at126316. Rawmin maximum dense termination beyond the last native interval is.00306/.00145. This fixed115-ray result is not a whole-grid or future-step certificate.

| Group | Model | Median absolute t50 residual | Median10–90% width | Unknown widths |
|---|---|---:|---:|---:|
| old_prefix_supervised | R0_W64 | 0.001533 | 0.029617 | 0 |
| old_prefix_supervised | half_selected | 0.000819 | 0.033759 | 0 |
| old_prefix_supervised | half_rawmin | 0.000671 | 0.032807 | 0 |
| old_prefix_supervised | cold01_terminal | 0.001895 | 0.021107 | 0 |
| new_prefix_supervised | R0_W64 | 0.002685 | 0.026462 | 3 |
| new_prefix_supervised | half_selected | 0.001514 | 0.018481 | 4 |
| new_prefix_supervised | half_rawmin | 0.001783 | 0.019099 | 4 |
| new_prefix_supervised | cold01_terminal | 0.001873 | 0.008026 | 2 |
| old_local_reserved | R0_W64 | 0.000372 | 0.034799 | 0 |
| old_local_reserved | half_selected | 0.000858 | 0.025976 | 0 |
| old_local_reserved | half_rawmin | 0.000749 | 0.027831 | 0 |
| old_local_reserved | cold01_terminal | 0.000857 | 0.018420 | 0 |
| new_whole_patch_reserved | R0_W64 | 0.000897 | 0.003224 | 0 |
| new_whole_patch_reserved | half_selected | 0.000709 | 0.009157 | 0 |
| new_whole_patch_reserved | half_rawmin | 0.000906 | 0.005744 | 0 |
| new_whole_patch_reserved | cold01_terminal | 0.000639 | 0.002154 | 0 |

Depth units are the accepted scene units, not certified meters. Whole-patch reserves are excluded from auxiliary geometry supervision but remain TRAIN RGB evidence. Point5568 hair/clothing ambiguity is retained unchanged. ±.02 mass is a coarse locality diagnostic, not thin-surface truth.

## Visual inspection and limitations

Personally inspected DC black, HC grey and KA black full-body comparisons for GT, R0selected111316, halfselector60752, rawmin126316 and terminal131316. Clothing details improve; gross hair/shoe disappearance is not apparent in these views. Broad colored exterior haze and floor/bag content remain.

At16HDpx outside the fixed actor masks, alpha>.1 counts DC/HC/KA are R0:185261/107305/74830; halfselector:193349/111919/76948; halfrawmin:192023/110631/76112; terminal:192266/110487/76183. Exterior masks are not certified empty-space labels, but these numbers and visible haze prevent claiming a clean asset from LPIPS alone.

Half-step favors detail, whereas cold.1 favors localized termination and narrower distributions. The results support keeping these quality dimensions separate; neither produces an ideal full-scene occupancy grid. No triple interaction is launched from this review.

Evidence: `C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`; final immutable choices under `C/independent_occupancy/semantic_cold_halfstep_final_choices`, GPU audit under `semantic_cold_halfstep_final_geometry`, CPU reused comparison under `semantic_cold_halfstep_final_vs_R0`, and [DC final comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_halfstep_final_exterior/DC_black_comparison.jpg), [HC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_halfstep_final_exterior/HC_grey_comparison.jpg), [KA](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_halfstep_final_exterior/KA_black_comparison.jpg).
