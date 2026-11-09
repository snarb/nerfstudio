# Frozen16k spatial supersampling diagnostic

What was tested: actual MVS16k checkpoint `5aa08c7cc725cb38e1de33141acc9a5dfd77041d9e808d7d9d97c3832a4c11f2`, default core0, render dt.003. Crops KA[60,630,600,900] and HC[1250,650,1550,900] were rendered at1×,2×,4× raster dimensions (4× uses16 rays per original pixel). K scales exactly: fx/fy×S, (cx−x0)/(cy−y0)×S; pose unchanged. Both encoded-RGB area averaging and inverse-sRGB→inverse-Reinhard linear-light averaging were evaluated against the same original1× GT. No fitting, GT compositing, NN/grid change or worker continuation occurred.

| Crop/method | PSNR | SSIM | LPIPS | Gaussianσ8 residual luma std |
|---|---:|---:|---:|---:|
| KA1× |29.02855|.810591|.444552|.01371583|
| KA4× encoded area |29.03760|.811755|.450543|.01371567|
| KA4× linear area |29.04007|.811762|.450554|.01372008|
| HC1× |33.80633|.894903|.156424|.00834139|
| HC4× encoded area |33.84024|.896817|.160154|.00833463|
| HC4× linear area |33.85257|.896819|.159977|.00832443|

Results: both actual contact pages were manually viewed. Broad KA brown/gray waves and HC mottling remain essentially unchanged with either averaging domain; only fine pixel edges become slightly smoother. Lowpass luma variability changes by approximately0% in KA and less than.21% in HC. This does not support spatial pixel-center aliasing as the main cause of these two broad defects. It does not rule out every possible sampling issue elsewhere.

Validation:1× maxabs parity to saved HD is6.2585e−7 KA and4.1723e−7 HC. Display/inverse roundtrip maxerror2.98e−8, including the white limit. NN/core/EMA digest is unchanged. Raw supersampled float RGB/alpha/depth, both downsampled variants, original GT, Gaussianσ8 residual arrays, crop metrics and source/checkpoint/K receipts are preserved. An initial read-only setup guard expected an LC collider; it was replaced by the existing native-AABB consistency check before successful rendering. No production configuration changed.

Artifacts: `/fsx/tmp/marshawn_001660_hd/native_png_MVSdepth14k_STAGED/spatial_AA16k/` with `audit.json`, `manual_review_occupancy.json`, `SHA256_MANIFEST.json` and exact executed helper source. Audit SHA `cf05975e361ae9142dda1fdefb28ea990d3f1cc6e07a5d0cf1ed6ae418f8265f`. These are crop diagnostics, not global checkpoint-selection metrics or scene-cleanliness acceptance.
