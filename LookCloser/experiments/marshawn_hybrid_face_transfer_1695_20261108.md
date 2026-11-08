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

## Authorized finite HC comparison: gain1 rejected

ROOT subsequently authorized one private CPU sigma4/gain.75 versus gain1 comparison, using cached1695 HC1920 geometry and exactly the other15 donors. Cached/prepared camera alignment is exact. Re-decoding actual HC RAW through the frozen linear AREA/remap/k4 display chain reproduces the prepared HC photograph byte-for-byte (maximum difference0). Target photography participates only in reference/scoring, never prediction. This remains an HD TRAIN-camera test.

Each cell below is PSNR↑ / SSIM↑ / LPIPS↓ on actor-supported pixels within photo-reviewed patches:

|Patch|LC|gain.75|gain1|
|---|---|---|---|
|Black face|50.8954 / .997848 / .000166|38.4302 / .963903 / .009543|36.0306 / .941732 / .021924|
|Gray face|47.2564 / .996414 / .000288|37.0852 / .944462 / .007205|34.7437 / .911198 / .015452|
|Equal mean of8 face/hair/shirt/boot patches|47.8441 / .995730 / .001004|35.7166 / .920048 / .011407|33.4266 / .877301 / .022775|

Gain1 worsens all three metrics in all eight patches versus.75; LC alone is best in all eight, consistent with its excellent in-sample fit. Native contacts show added fine-pattern/color variation rather than demonstrated facial recovery. Acceptance/donor labels are identical between gain variants and fallback remains exactly LC. This rejects the candidate on current HC; it does not measure novel-view fidelity or establish a safe alternative.

[Full finite report](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/private_HC_gain_1695_20261108/REPORT.md), [native contacts](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/private_HC_gain_1695_20261108/all_native_contacts_legible.png), [metrics and proofs](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/private_HC_gain_1695_20261108/complete.json). Worker46.30s, peak2709.89MiB; hard limits300s/12GiB/3CPUthreads. All351 bound inputs, including the previous286, remain unchanged; no GPU, model query, depth interpolation, production change or adoption. No additional test ran.

ROOT personally inspected the native contact: LC is close to actual RAW_AREA, hybrid is noisier especially on dark cloth, and gain1 offers no clear visual win. Production is retained. The [personal visual receipt](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/private_HC_gain_1695_20261108/ROOT_personal_visual_review.json) binds all351 inputs and complete SHA `fb032f5408bb951a0c24be2875e8139cb1a1708bafb1321736c5589eff8ac447`; receipt SHA `21115582ae7f9453f3273ba855d7dd4c43aa5197763ef80c9535448e23e9b5fb`. This is a1920 HD/in-sample TRAIN-camera comparison against canonical RAW_AREA. It does not establish native4K novel-render LPIPS or justify blanket disabling hybrid. Exact RAW bytes matching prepared HC strongly check photometry/alignment. Work stopped with no further experiment or promotion.
