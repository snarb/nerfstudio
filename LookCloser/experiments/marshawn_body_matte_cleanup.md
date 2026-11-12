# Marshawn previous-flight body matte cleanup, 2026-11-12

## What was tested

MatAnyone2 inference on all70 **previous low-flight** neural frames (not the rejected wide flight), using raw NeRF black composites as RGB and frame000 strict accumulation as the soft initial0–255 prompt. Ten first-frame warmups, official first-frame reset convention, native1920×1080, FP16, GPU4. Existing isolated MatAnyone2 environment and weights from the hair experiment were reused; no training or source/delivery edits.

Silver root `E=/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/operations/body_matte_cleanup_20261112`. Source `../flying_strict_low_20261112/neural/{000..069}`. Code: [inference](../scripts/marshawn_sequence/body_matte_neural.py), [conservative hybrid](../scripts/marshawn_sequence/body_matte_hybrid.py).

Three outputs were inspected:

- `sequence/mat_black`: unrestricted rawRGB × Mat2 alpha.
- `sequence/safe_black`: baseline strictRGB, Mat2 reduction only within4px of strict support edge.
- `hybrid/black`: stronger reduction confined to explicit lower-body rectangles interpolated across time;8px boundary fully eligible,12px interior protected,16px rectangle feather. Upper bodies, hair and bat regions excluded. Frames054–069 unchanged. Rectangles are conservative editing regions, not semantic ground truth.

## Results

| Output | Observation / measured invariant |
| --- | --- |
| Unrestricted | Strong rim cleanup; **reject globally: erases bat at035**, modifies/adds hair from raw rendering |
|4px boundary | Keeps interior but leaves substantial ochre leg rims and lying-actor attached ghosts |
| Lower-body hybrid | Visibly reduces leg rims,54/70frames modified;454,397 changed RGBpixels across sequence;8,764 at035 |
| Protected areas | Maximum RGB change0 outside edit regions or at distance≥12px from strict edge, checked across all70frames |

Evidence:

- [Native-body pilot035: strict / unrestricted /4px](../lookcloser_debug_outputs/body_matte_cleanup_20261112/body_pilot035.jpg): unrestricted bat failure is clearly visible.
- [Frame035 strict / conservative hybrid](../lookcloser_debug_outputs/body_matte_cleanup_20261112/body_hybrid035.jpg): leg rims improved, bat retained.
- [Lying actor temporal comparison](../lookcloser_debug_outputs/body_matte_cleanup_20261112/body_lying.jpg): unrestricted matte removes attached ghosts better than4px restriction. This overview is resized; use native panel for anatomy.
- [Native lying035 raw / strict / unrestricted](../lookcloser_debug_outputs/body_matte_cleanup_20261112/body_lying035.jpg).
- Silver `hybrid/comparison_24fps.mp4`: full70frame side-by-side; `hybrid/review/` sampled contacts; `hybrid/receipt.json` all per-frame protection checks and ROI coordinates.

All inference finished; GPU4 returned to0MiB, no OOM observed. This is matte/postprocess evaluation without trusted alpha ground truth: PSNR, SSIM, LPIPS against the imperfect input matte would not establish correctness and are not claimed.

## Insights and integration

The useful result is the **conservative hybrid**, not unrestricted Mat2. Hair, props, and deep body interiors remain baseline in its chosen regions. This is intentionally limited: upper-body halos remain, and ambiguous real boundary pixels cannot be proven correct without annotation. It does not fix missing geometry or other actors.

For integration, `hybrid/factor/{000..069}.png` stores exact uint16 multiplicative factors. Apply `round(strict_rgb * factor/65535)`; use with the matching previous-flight frame only. Existing source renders were never overwritten.

Mat2 also gives promising lying-actor cleanup at020–040, but rawRGB×alpha can reintroduce gray between legs that strict already removed. For a separately reviewed lying-actor region, use `strictRGB * min(1, Mat2alpha/max(strict_accumulation,1e-4))`, not rawRGB×alpha; preserve head/hair and evaluate foot edges. This candidate was passed to the parent agent for combination with their independent lying-actor investigation.

## Follow-up: torso, shoulder, and arm cleanup

The lower-body-only candidate did not cover the complete user request. A separate `hybrid_body` extends cleanup to both standing actors' torsos/arms/shoulders, using **strict RGB only**, with the smaller4px-full/8px-zero boundary band. Its larger≥12px core is unchanged. Manual review-space rectangles at frames000,005,…065,069 specify the two actor regions and protect heads/hair, the bat, and the other tumbling actor. Coordinates interpolate linearly; guard falloff extends20native pixels outside each guard. Explicit guards override even the earlier lower-body factor. Code: [body_matte_upper.py](../scripts/marshawn_sequence/body_matte_upper.py).

This is the preferred body candidate for integration, replacing the lower-body-only factor. It gives821,832 changed RGBpixels across70frames, versus454,397 for the earlier restricted output. These are edit counts, **not accuracy scores**. Maximum RGB change inside protected guards or≥12px body interiors is0 across allframes. At035 the extended candidate differs from lower-only at4,327pixels; at060 at11,619pixels. It retains the baseline bat and hair in inspected samples and reduces outer torso/sleeve haze. The smaller upper-body band deliberately leaves ambiguous thick halos rather than permitting deeper cloth removal. There is no ground-truth annotation guaranteeing all boundary edits.

- [Native035 strict / body hybrid](../lookcloser_debug_outputs/body_matte_cleanup_20261112/body_upper035.jpg): bat and hair retained, leg/torso/arm rims reduced.
- [Native060 strict / body hybrid](../lookcloser_debug_outputs/body_matte_cleanup_20261112/body_upper060.jpg): modest sleeve/torso contour cleanup; source cloth and hands remain recognizable. Upper hair falls outside this crop; full-frame guards are in the receipt.
- Silver `hybrid_body/comparison_24fps.mp4`: full70frame temporal comparison.
- Silver `hybrid_body/factor/{000..069}.png`: exact uint16 integration factors, `round(strictRGB * factor/65535)`.
- Silver `hybrid_body/receipt.json`: interpolated guard coordinates, boundary settings, all per-frame edit/protection counts.

Lying-actor refinement remains a separate parent-agent integration so the same frame is not inadvertently reconstructed from raw RGB. Prior variants and existing deliveries remain unchanged.
