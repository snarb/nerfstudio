# Marshawn14k same-sample near-prefix causal test

What was tested: exact immutable14k checkpoint `e92e525fe7451ef9cc2959ecca57a09e68925b20143714bf463c54a6ed998ae9`; the same9 KA floor rays and3 fixed centers from predeclared cardboard/paper diagnostic ROIs. Queried actual native packed samples plus16384/32768 uniform density and RGB samples. No camera, near plane, occupancy, NN or gain changes.

For each cutoff .01/.02/.03/.05/.1, integrate prefix optical thickness and premultiplied RGB using the same piecewise-constant sigma/RGB cells, including the cut cell's partial length. The suffix counterfactual is `(Cfull−Cprefix)/Tprefix`, with the corresponding alpha identity. Independent direct suffix recomposition agrees within1e−10. A constant-density analytic partial-cell test agrees within1e−14. Native actual packed weights agree with captured sigma integration. Native near is .01, so density below .01 is outside this test.

Results:

| Samples / ray group | Original median diagnostic EV | Cut.02 median per-ray ΔEV | Cut.1 median per-ray ΔEV |
|---|---:|---:|---:|
| Actual native /9floor | −.52768 | +.00863 | +.03191 |
| Uniform32768 /9floor | −.61068 | +.07002 | +.09876 |
| Actual native /3background controls | +.38824 | +.12428 | +.12428 |
| Uniform32768 /3background controls | +.30165 | +.12293 | +.12238 |

EV is diagnostic inverse-sRGB→inverse-Reinhard→Rec709-Y relative to the same GT; no gain is estimated or applied. Per-ray deltas are reported separately from medians because ray ordering can change.

Actual native floor prefix mass up to .02 is only .000506–.005128; the source grid skips much of the dense near-camera lobe. Removing to .02 changes any floor RGB channel by at most .001105, less than one8bit code. Removing to .1 changes floor RGB by at most .005867 and leaves median suffix error −.50830EV. The background controls already predict too bright and become brighter after removing their dark prefix. Density near the camera is therefore a secondary attenuation source, not a sufficient or floor-specific explanation of the colour gap.

Dense16k/32k full-RGB convergence error is at most .00006772 and suffix-RGB error .00007098. Queries use FP32 positions/output without AMP; native TCNN internals remain unchanged. Integration uses NumPy FP64. All native/core/EMA/NN hashes are unchanged after both probes. No production near-plane change is justified by this evidence.

Evidence:

- [Exact native+dense audit](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/near_prefix14k_NATIVE_AND_DENSE/audit.json).
- [Per-ray prefix RGB/mass and EV summary](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/near_prefix14k_NATIVE_AND_DENSE/summary.json).
- [Raw native and16k/32k sigma/RGB/weights](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/near_prefix14k_NATIVE_AND_DENSE/traces.npz).
- [Executed source snapshot](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/near_prefix14k_NATIVE_AND_DENSE/source_at_probe.py).

Insight: reject near-prefix removal as the main colour fix. The controls are ordinary predeclared ROI centers, not calibrated diffuse references; neither this counterfactual nor existing physical rig support authorizes a clipping change. Source photometry and view/material dependence remain separate questions. Floor cleanliness is not complete.
