# Marshawn 1660: normal RGB consistency, head-only auxiliary

## What was tested

A 2,000-update continuation from the immutable PNG 14k checkpoint, with one added auxiliary: RGB consistency across a ±0.001 prepared-world normal pair in the finite, manually reviewed floor-prior domain. The pair shares one actual clear TRAIN view direction. Geometry embeddings and SH direction encodings are detached; only the native color head receives this auxiliary gradient. Existing RGB, LOS, SfM surface, distortion, floor prior, random background, sampler and scheduler remain in place.

The pair center follows the conditional termination-weighted median of 32 normal-band density samples, viewed from the chosen TRAIN camera. Maximum density was rejected as a center because its typical bin carried only ~9e-9 termination mass, compared with ~0.19 at the visible selection. The plane/domain remains an explicit scene assumption, not measured depth.

The selected weight, **0.01**, gave auxiliary/RGB-head gradient norm ratios of 0.0142 on the initial 64 rays and 0.1383 on the adapted 16,496-ray / 1.207M-sample packet. Full-state CUDA proofs verified finite combined AMP gradients, no direct auxiliary geometry gradient, unchanged parameter/state schema, and exact native auxiliary RGB forward equality. This isolates the auxiliary gradient, not all subsequent geometry changes from the original training losses.

## Results

**Rejected as a solution for visible floor spots.** Root and independent geometry review found the broad gray/brown floor waviness essentially unchanged. The worker was stopped at the mandatory 16k gate before step 16001; 2,000 new attempts produced 2,000 updates and no AMP skips. The conservative family attempt count is **17,170**, below 22,000.

| Native all-three evaluation | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Parent 14k | 29.23558 | 0.813125 | 0.269608 |
| Head-only auxiliary 16k, final | 29.20931 | 0.813595 | 0.268126 |

The prescribed PSNR / 0.07 dB LPIPS tie rule selects 16k; that formal selection does not constitute visual cleanliness approval. Final native evaluation is retained with the checkpoint.

| Native diagnostic crop | 14k PSNR / SSIM / LPIPS | 16k PSNR / SSIM / LPIPS |
|---|---|---|
| HC floor | 31.965 / 0.86452 / 0.22456 | 32.267 / 0.86659 / 0.21934 |
| KA floor | 31.736 / 0.82777 / 0.32249 | 31.696 / 0.82833 / 0.31689 |
| KA prior + margin | 28.756 / 0.80836 / 0.52967 | 29.141 / 0.80830 / 0.53329 |
| KA former black ROI | 26.266 / 0.81874 / 0.44563 | 26.794 / 0.82032 / 0.44291 |
| DC visible corner | 28.787 / 0.82412 / 0.35417 | 28.531 / 0.82149 / 0.35948 |

[All three views: GT / parent14k / trial16k / safe13k](/fsx/tmp/marshawn_001660_hd/native_png_floor_normalRGB_HEADONLY_STAGED/gate16k_all3_GT_parent_new_safe.jpg), [KA wider floor context](/fsx/tmp/marshawn_001660_hd/native_png_floor_normalRGB_HEADONLY_STAGED/gate16k_KA_prior_boundary_margin.png), [native crop metrics](/fsx/tmp/marshawn_001660_hd/native_png_floor_normalRGB_HEADONLY_STAGED/gate16k_crop_metrics.json).

## Insights

Actual floor observations vary by roughly 1–1.65 EV across vetted TRAIN directions; undoing the known scalar exposure adjustments leaves roughly 1.05–1.47 EV. A direction-variance penalty is therefore unsupported. Some color variation also exists within density-supported layers, but tangential variation is comparable, so the diagnostic did not uniquely establish normal layering as the cause. The controlled normal auxiliary did not deliver a convincing visual fix.

[Configuration, proof and launch plan](/fsx/tmp/marshawn_001660_hd/native_png_floor_normalRGB_HEADONLY_STAGED/preparation_receipt.json), [root visual rejection](/fsx/tmp/marshawn_001660_hd/native_png_floor_normalRGB_HEADONLY_STAGED/root_gate16k_rejected.json), [directional/normal evidence](/fsx/tmp/marshawn_001660_hd/native_png_random10k_floor_band_PRIOR_STAGED/directional_floor14k_DIAGNOSTIC/manual_review_geometry.json).

Bookkeeping issue: the live frozen configuration retained the parent's `final_grid_artifact_dir`; the extra `final_grid_export_dir` setting was unused. The 16k export consequently appended new files to the parent grid directory, without modifying its 12k/14k files. Byte-identical aliases and an explicit [provenance receipt](/fsx/tmp/marshawn_001660_hd/native_png_floor_normalRGB_HEADONLY_STAGED/grid_export_alias_receipt.json) preserve the actual authority. The suspended worker/configuration was not modified.

Terminal native BODY128 audit: 126/128 rays retain at least 50% mass in the witness band; median conditional depth error is +0.0482%. The unguarded core misses 65/9022 known scene points and 11/495 subject points, so this experiment also does not constitute the requested final occupancy release. [Final receipt](/fsx/tmp/marshawn_001660_hd/native_png_floor_normalRGB_HEADONLY_STAGED/finished_receipt.json).
