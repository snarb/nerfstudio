# Held-out actor-mask source audit

What was tested: CPU-only source annotation audit of frozen DC/HC/KA V1 scoring masks against native 8192×4320 RAW photographs and original native SAM masks. The proposed V2 is an **unapproved, separate exclusion-only scoring diagnostic**; no training label, field, camera, historical score, crop, or leaderboard changed. Local paper design does not establish semantic-mask truth.

Results:

| View / actor | V1 pixels | Proposed exclusions | Remaining if approved |
|---|---:|---:|---:|
| DC black / gray | 50,554 / 32,212 | 0 / 0 | 50,554 / 32,212 |
| HC black / gray | 39,768 / 30,426 | 0 / 0 | 39,768 / 30,426 |
| KA black / gray | 31,759 / 29,297 | 0 / 28 | 31,759 / 29,269 |

The KA proposal covers only distal knitted bag interior beside the near gray shoe, HD bbox `[926,488,943,501]` (exclusive upper bounds). Native RAW polygon vertices are `[(3949,2094),(3972,2079),(3986,2087),(3998,2106),(4024,2122),(4031,2137),(4006,2141),(3969,2132),(3950,2126)]`. Its 3,013 native pixels yield 138 fully contained HD pixels; only 28 intersect V1. Their calibrated source-polygon coverage is exactly 1. The minimum excluded HD cell-corner distance to the manually conservative shoe/contact guard is 12.7548 pixels. This guard is source annotation pending manual review, not calibrated silhouette or optical-alpha truth.

[Native KA source + SAM + proposal, 1:1](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/KA_native_contact_page.png) · [KA mapped HD overlay, 3× nearest display](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/KA_HD_contact_3x.png) · [KA wide source context](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/KA_wide_source.jpg).

[DC native contact page](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/DC_native_contact_page.png) · [HC native contact page](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/HC_native_contact_page.png) · [Exact vertices, guard, counts, source hashes and proposal NPZ bindings](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/proposal_manifest.json) · [Reproducible CPU builder](/fsx/tmp/marshawn_001660_hd/tuning_20261106/eval_actor_mask_source_audit_CPU_v1/build_audit.py).

Original TIFF/display/native-mask/HD/V1 hashes were verified. Native TIFF gain4 Rec709 display replay has zero channel-code difference in every contact ROI. Explicit OPENCV projection agrees with remap tables within 0.0005 RAW pixel. The exact existing AREA-prefilter plus calibrated bilinear remap is used; distorted masks are never naively resized. All original V1 semantic archives, ready receipt, transforms and source-image hashes match afterward. Raster remains 1920×1012 and all six crop boxes remain exact. No GPU, fitting, rendering, or PSNR/SSIM/LPIPS computation occurred.

Insights: A small unambiguous portion of KA bag is demonstrably included in actor support. This audit does not certify a complete actor-only mask: larger shoe/bag contact, wrist/sleeve, hair and mixed pixels remain untouched. DC/HC contact boundaries do not support a comparably separated exclusion in this review. Root must inspect native contact pages and exact vertices before any use. If separately approved, apply `maskV2 = maskV1 & ~exclusion` with identical historical neutral-0.5 compositing and original crops to every compared baseline/candidate render, report alongside unchanged V1, and never retroactively reselect the leader. Visible bag membership does not imply anything about hidden actors or 3D free space.

Blinding limit: An initial broad source-path grep inadvertently exposed historical metric text and a source-location discussion in an existing experiment note; therefore this is not a pristine blinded study. No prediction/render file or candidate-metric artifact was opened, and model scores/differences were not used to choose the polygon. The proposal was drawn from independently viewed native source photographs and native SAM masks only.
