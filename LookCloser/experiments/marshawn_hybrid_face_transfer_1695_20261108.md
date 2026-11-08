# Actual1695 cached face transfer diagnosis

## What was tested

Read-only CPU analysis of canonical0007 saved LC/hybrid PNGs and transfer labels/weights, plus the exact frozen geometric acceptance functions on existing1695 target and16 source caches. Corrected broad head ROIs: black[1900,520,2350,990], gray[1930,1190,2350,1580]. No new renderer, GPU, field query, sweep or production change.

## Results

|Window|Target-valid|Transferred|Fraction|IC share of transfer|
|---|---:|---:|---:|---:|
|Black broad head|142,998|110,520|77.29%|99.61%|
|Gray broad head|122,147|96,425|78.94%|98.09%|
|Black visible-face diagnostic rectangle|28,800|24,718|85.83%|99.55%|
|Gray visible-face diagnostic rectangle|18,843|16,429|87.19%|99.37%|

The broad boxes include background/hair/clothing; diagnostic rectangles are not anatomical masks. Eligibility/source-opacity explain no visible-face control loss. Black/gray fallback comprises1,568/634 pixels failing geometric tests and2,514/1,780 passing geometry but rejected later. The saved audit cannot split native validity, photometric agreement/support and median-label cleanup. All actually accepted pixels pass cached geometric replay.

The shader transfers a single donor's high-pass residual, retaining the LC base and fading at source boundaries. Saved hybrid high-frequency energy increases, but this novel view has no photographic target ground truth. No fidelity gain is inferred from texture energy.

[Full report and feasibility](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/face_transfer_cached_audit_1695_20261108/REPORT.md), [CPU results/proofs](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/face_transfer_cached_audit_1695_20261108/complete.json), [black saved contact](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/face_transfer_cached_audit_1695_20261108/black_head_saved_contact.png), [gray saved contact](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/face_transfer_cached_audit_1695_20261108/gray_head_saved_contact.png). All328 bound inputs, including the prior286, remained unchanged. Runtime9.71s, peak670MiB.

## Insights

No rejection-threshold relaxation is justified. One future-only candidate is fixed sigma4/gain1 versus4/.75 on current1695 HC, excluding HC from all texture donors. Existing1670 native-face scores favor gain1 for LPIPS, with a PSNR/SSIM tradeoff; this remains unproven for current1695. Cached HC neuralRGB/alpha/q10/50/90 and calibrated pose exist at1920×1012, along with all15 other donors and HC's native RAW photo. This permits a bounded HD TRAIN-camera comparison without new GPU geometry, not an independent held-out or native4K geometry claim. KA has no cache. No comparison was launched or adopted.
