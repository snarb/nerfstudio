# Marshawn 1660 canonical colour

## What was tested

Independent CPU reconstruction of 8,192 actual HC pixels from the source Log3G10/RWG TIFF, through the vendor matrix, calibrated linear undistortion, signed luminance scale and PQ16 quantization. The original vendor full-resolution PQ16 image is the comparison target. No model fitting or image-dependent colour matching was used.

## Results

| Candidate transform | Maximum PQ16 code error | Mean absolute code error | Exact channels |
|---|---:|---:|---:|
| Vendor RWG D65 → AP1 D60, Bradford | 0 | 0 | 100% |
| AP1 with an actual D65 white point | 143 | 3.255 | 21.48% |
| D60 matrix, incorrectly clipping negative RWG before conversion | 873 | 0.198 | 97.64% |

Evidence: [hash-bound audit](/fsx/tmp/marshawn_001660_hd/report/canonical_colour_independent/audit.json), [sampled source/target codes](/fsx/tmp/marshawn_001660_hd/report/canonical_colour_independent/sampled_vendor_pixels.npz). Source/vendor/calibration/range file hashes are recorded there. The actual vendor matrix uses AP1 D60 despite the historical `ap1-d65` label. Its luminance weights are `(0.2722287168, 0.6740817658, 0.0536895174)`.

The signed canonical mapping is 52.13360318135631 nits per unbaked linear AP1 unit, with the PQ archive clamped to 0.005–4000 nits. These are distinct from the earlier experimental gain4 Rec709 master with a declared 203 nits per stored unit.

## Fixed display mapping

For a comparison using the earlier gain4 display exposure, transform canonical AP1 nits to Rec709 D65 with the vendor Bradford matrix, multiply by `4 / 52.13360318135631`, clamp negative display RGB, then apply Reinhard and the sRGB OETF. For an unbaked relative AP1 EXR, the factor is simply 4. No eval RGB is used to estimate this transform. A physical preview dividing nits by 203 is also valid, but is 15.5754 times darker before tone mapping than the earlier gain4 look.

Matrix closure agrees with the direct RWG→Rec709 transform to `6.66e-16`. Exact reconstruction of all earlier PNG pixels is not implied: the old helper clipped negative RWG and Rec709 before prefiltering; the vendor instead retains negative RWG until AP1 conversion and clipping. Spatial processing also differs.

The separate direct-PQ numerical control is recorded at [native_direct_pq_STAGED](/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/stage_receipt.json). It fits normalized AP1 PQ16 codes as float32 targets using unchanged native NGP architecture, LOS/surface auxiliaries, scene_loose and core0. Code-space compositing is an explicit surrogate, not physical radiance integration. Its fixed background is the archived floor code `988/65535`.

The [actual CUDA preflight](/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/actual_cuda_preflight.json) used zero optimizer steps: all 128 sampled targets matched EXR values exactly, calibrated ray direction error was below `8.1e-8`, and the combined RGB/LOS/surface backward was finite. The NN parameter/schema comparison and zero-opacity floor passed. An [independent prelaunch review](/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/independent_review.json) additionally checked full-frame PQ16/EXR parity and all66 camera metadata.

The authorized cold run reached its first saved 2000-step gate and paused in place, preserving optimizer/RNG state. [Decoded three-view comparison](/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/preview_2000/all_GT_prediction.jpg): all three views contain recognizable geometry and opaque central actors. Underfit blur and floor/mat haze remain. This passes the initial visual gate only; it does not establish final quality or physical occupancy correctness. Actual PQ-code buffers, depth and opacity are retained alongside the display images. No black-floor subtraction, GT compositing or fitted display adjustment is applied.

After root's independent three-view review, the same process continued. The default quiet plateau rule stopped the run after the 6000-step eval boundary (last compact training row 6420), below the 22,000 cap. No second run or checkpoint resume was started. The strict all-view selector chose step6000, using LPIPS within 0.07 dB of the highest PSNR.

| Step | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 2000 | 33.9699 | 0.860019 | 0.364711 |
| 4000 | 35.1251 | 0.877499 | 0.299011 |
| 6000 | 35.1708 | 0.887686 | 0.257352 |

These metrics are measured in normalized **PQ-code space**, not physical radiance or the earlier SDR target space. They cannot establish a win over the accepted PNG model. That comparison needs the separately declared common display target.

The selected checkpoint's [full-HD actual renders](/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/best6000_HD/all_GT_prediction.jpg) and [completion receipt](/fsx/tmp/marshawn_001660_hd/native_direct_pq_STAGED/completion.json) retain all three raw PQ-code prediction/GT buffers and their hashes. All three native 1920×1012 predictions were manually inspected: the central pair is coherent and opaque; floor/mat haze, fine-detail blur and background softness remain. The accepted SDR release is unchanged. Controller and worker exited, and GPU memory returned to 14 MiB.
