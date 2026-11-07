# Separate source annotation diagnostic

## What was tested

Original frame1660 KA source imagery contains a knitted floor bag touching the grey actor's boot. Root personally inspected the native RAW source, original segmentation and calibrated HD proposal. Twenty-eight fully contained HD pixels on the distal bag, separated from a conservative shoe/contact guard, qualify for a **separate source annotation diagnostic**. DC/HC and the other actor receive zero exclusions. Mixed contact pixels, hair and footwear remain unchanged.

The original dataset masks, fixed crops, training labels, historical V1 scores and checkpoint selection remain unchanged. Every compared source/P3 render receives the identical diagnostic mask. Both domains use the existing Alex LPIPS helper, no resizing, equal weighting of two actors and three views, and neutral0.5 outside each mask in prediction and target. This manual semantic annotation does not provide optical alpha or hidden-space geometry.

## Results

The CPU comparison uses unchanged full float render archives. CPU results may differ slightly from earlier GPU LPIPS calculations; each row below is recomputed using the same CPU backend in both domains.

| Render | Domain | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| Selected126316 source | Historical V1 |31.639727|.831264|.01877399|
| Selected126316 source | Bag28 diagnostic |31.649476|.831269|.01876577|
| p=.01 hard≥.1 | Historical V1 |30.887904|.828332|.02216182|
| p=.01 hard≥.1 | Bag28 diagnostic |30.899217|.828360|.02215528|
| p=.01 hard≥.5 | Historical V1 |30.723478|.827095|.02334031|
| p=.01 hard≥.5 | Bag28 diagnostic |30.733495|.827122|.02329351|
| p=.99 hard≥.1 | Historical V1 |30.973597|.828647|.02139289|
| p=.99 hard≥.1 | Bag28 diagnostic |30.985316|.828675|.02140198|

[Complete nine-variant comparison and input hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/source_annotation_diagnostic_CPU_v1/metrics.json). [Root source review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/root_manual_source_review.json). [Native source proposal](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/KA_native_contact_page.png).

The evaluator verifies the exact approved proposal and all original render/mask hashes after computation; CUDA remains uninitialized. Its first attempt stopped at dynamic module registration before scoring; the corrected retry registers the existing metric module for Python dataclasses and completes. No model or metric formula changed.

## Insights

This conservative bag correction produces only small metric changes. It does **not** explain the isolation variants' overall regression: source-to-hard≥.1 LPIPS remains approximately.00339 worse in the diagnostic domain. Continue checking genuine hair, contour and contact fidelity. Keep historical V1 as the primary published comparison; this diagnostic neither reselects a leader nor authorizes EVAL-based fitting.

Reproduce with `scripts/score_marshawn_source_annotation_diagnostic_cpu.py`, specifying the campaign root and a new output JSON. The script refuses to overwrite an existing result.
