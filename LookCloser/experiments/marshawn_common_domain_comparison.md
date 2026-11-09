# Canonical PQ versus accepted PNG in one comparison domain

## What was tested

The selected numerical-PQ native NGP checkpoint6000 and the accepted PNG
prediction are scored against the same canonical ground truth at1920×1012.
No field was rerendered and no exposure was fitted for this comparison.

Actual canonical input GT and new prediction codes both follow the pinned
ST2084 EOTF, division by signed gain52.13360318135631 without black subtraction,
AP1-D60→Rec709-D65 Bradford conversion, positive clamp, gain4×2^fixed-old-camera-EV,
Reinhard, sRGB and identical uint8 quantization. Fixed train-derived EV values
are DC−0.18336297, HC−0.21886586 and KA+0.18751854. The accepted PNG prediction
already has these gains baked in and is scored unchanged.

## Results

| Prediction, same canonical GT/domain | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Accepted PNG | **27.415559** | **0.781585** | **0.358962** |
| Numerical PQ, best6k | 26.636813 | 0.764082 | 0.408336 |
| New minus accepted | −0.778746 | −0.017503 | +0.049375 |

The new run has lower PSNR/SSIM and higher LPIPS in every held-out view.
The all-three-view contact was manually inspected: central people remain
coherent, with persistent detail/periphery/floor softness and no consistent
tiny-detail improvement. Retain the accepted PNG release.

This is a comparative choice only. Gray mat/floor patches remain in the best
PNG output too. The scene-cleanliness requirement is unresolved; absence of a
large cloud is not equivalent to removal of those patches. No release ZIP was
modified by this comparison.

![Canonical GT, accepted PNG, new PQ](/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/common_domain_best6000_vs_PNG/common_GT_predictions_contact.jpg)

GT decoded from PQ versus its float32 relative AP1 master differs by at most
one uint8 display code, mean0.00319–0.00356 code units, PSNR72.62–73.09dB.
Intentional floor/peak policy and quantization effects are recorded separately.
The old PNG target versus the new canonical GT has PSNR50.37–53.73dB; neither
prediction is scored against that historical target in this comparison.

Five CPU tests passed, including signed-gain inverse/no black subtraction,
fixed exposure validation, matrix convention/gamut clamp, codec-error separation
and rejection of upscaled previews. All raw producer PQ buffers are preserved
byte-for-byte. The same pinned matrices, vendor source and metric implementation
are used for both models. SSIM has data_range1; LPIPS uses alex/normalizeTrue.

## Insights

The native encoded-space value35.17dB cannot be compared with historical SDR
PSNR. The common-domain result is worse for this numerical-code run and does
not evaluate the held physical-linear HDR proposal. The accepted release and
its checkpoint remain unchanged.

For provenance only, the selected6k numerical-PQ control has native encoded-AP1
code-space PSNR35.1708 / SSIM0.887686 / LPIPS0.257352. These three values use a
different target domain and **were not used to rank it against the PNG output**.
The ranking uses only the common-GT table above. Floor/patch diagnostics and any
paper-derived density regularizer remain separate measured work, not completed
by this color-control experiment.

Helper: [compare_marshawn_common_domain.py](../scripts/compare_marshawn_common_domain.py).
Full report and executed argv:
`/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/common_domain_best6000_vs_PNG/{comparison.json,run_provenance.json}`.
