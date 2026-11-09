# Marshawn: two-actor black-background video

## What was tested

Clean the previously delivered LookCloser and hybrid clips without changing their original rendered body RGB or camera path. Both have ten frames, source IDs1660–1705 at stride5, native3840×2024 and24fps (10/24 seconds). The user accepts4K delivery. Original masters and videos remain preserved.

The hybrid colour lookup actually samples native8192×4320 RAW TIFF donors. All160 donor headers and the ten original output hashes were checked; the shader converts projected UV to calibrated native coordinates and decodes four native RAW taps before linear-light bilinear interpolation. HD geometry/eligibility buffers do not imply HD donor colours. [Donor audit](/fsx/tmp/marshawn_4k_actor_cutout_20261108/ops/delivered_native_donor_audit.json).

## Results

| Method | Actual visual result | Decision |
|---|---|---|
| Global SAM3 person proposals | Correct two actors in all10 frames of each variant; removes large floor fragments. Global encoder1008 is insufficient to establish native contour precision. | Identity/semantic guidance only. |
| Native1008 SAM3 interactive tiles, bidirectional logit blend | Sharper hair boundary, but punctures face/hands and removes black boots. | Rejected; no full-batch promotion. |
| BiRefNet HR matting, native2048 padded actor ROI | First-frame face/hands and four boots survive; soft floor/reflection residue remains under grey front boot. Source pixels are copied1:1, without shrinking. | Native contour candidate; semantic support and final visual review required. |
| SAM3 semantic guidance + native BiRef, core8/outside8 | All20 preserve the two actor silhouettes and four boots; distant floor/reflected-toe residue removed. | Selected, with the four explicitly reviewed interior repairs below. |

The selected trimap forces opacity only inside an8-native-pixel erosion of the two-actor semantic union, makes pixels farther than8native pixels outside that union exactly black, and retains the native BiRef alpha in the uncertain band. This is an explicit opaque-body/support assumption, not measured alpha ground truth. The initial20 outputs preserve10,746,725 opaque-core RGB pixels byte-for-byte and zero142,603,598 far-background pixels. No components were discarded merely for being small: reviewed tiny threshold islands lie on real hair and boot contours. [Policy review](/fsx/tmp/marshawn_4k_actor_cutout_20261108/ROOT_first2_trimap_policy.json).

A final cross-render enclosed-hole audit caught a false428px LookCloser trouser hole that the body thumbnails did not resolve. All10 frames were inspected for current-frame hybrid-supported holes. Root approved exactly four false components (428px cloth,226px braid,12px cloth,1px shoe); their supported8px rims were restored as well. This changes1205alpha pixels in LookCloser frames0/4/6 and restores original LookCloser RGB, without generating colour or changing other pixels. Larger anatomical openings and uncertain hair gaps remain untouched. Hybrid output is unchanged. [Exact approval](/fsx/tmp/marshawn_4k_actor_cutout_20261108/ROOT_validated_internal_hole_repairs.json), [repair/difference proof](/fsx/tmp/marshawn_4k_actor_cutout_20261108/ops/enclosed_hole_repair_v2/repair_proof.json). Final mask manifest SHA: `017bfc450a674852e0150160417cf3d0a97e4f8d5848d230ecd37248855197e6`.

Root reviewed all30 coarse identity contacts, including optional GT. Only the two fighting actors were selected; frame-to-frame candidate IDs are tracked rather than assumed constant. The final delivery scope is two clips, hybrid and LookCloser. [Identity review](/fsx/tmp/marshawn_4k_actor_cutout_20261108/ROOT_all30_identity_review.json), [native-tile rejection](/fsx/tmp/marshawn_4k_actor_cutout_20261108/ROOT_native_v2_rejected.json).

All NeRF training and automatic continuations were stopped before video masking. Video inference uses a separate campaign and qualified physical GPU leases. Existing66-camera metrics remain in-sample; future tuning uses the separately requested63/3 split. [Training stop receipt](/fsx/tmp/marshawn_native_crop_tuning_20261108/ops/stop_66train_20261108/complete.json).

## Insights

Native resolution alone does not establish a better mask: the tiled interactive model visibly damages real bodies despite finer hair contours. Broad semantic identity and native matting need separate checks. Detached floor debris and real whipping braids must not be treated alike. Do not propagate masks unchanged between moving novel views or smooth them into motion trails.

The compositor preserves exact source RGB wherever alpha is fully opaque and writes exact black at alpha0. Lossless PNG/alpha masters accompany single-generation H.264 clips (CRF12, all intra, no B frames, native raster,24fps). All20 body silhouettes, native hair sequences, hands and four-boot sequences were visually reviewed. First/middle/last decoded clips were compared with their PNG masters. Synthetic helper checks alone do not establish final video quality. [Hair review](/fsx/tmp/marshawn_4k_actor_cutout_20261108/review_hair20_native/manual_review.json), [hands/boots review](/fsx/tmp/marshawn_4k_actor_cutout_20261108/review_final20_band8_cpu/manual_hands_boots_review.json).

## Final delivery

- [Hybrid MP4](/fsx/tmp/marshawn_4k_actor_cutout_20261108/delivery/Marshawn_Hybrid_4K_Black.mp4):2,733,786bytes; SHA256`c9888c188ab2d825704a30d21124535d2a0b0190c60d2284febe281abbf74b83`.
- [LookCloser MP4](/fsx/tmp/marshawn_4k_actor_cutout_20261108/delivery/Marshawn_LookCloser_4K_Black.mp4):2,146,684bytes; SHA256`a895f80fe775d750d09ef0265c92ff440002c5f69d6d4cd5e33c4532c143e8eb`.
- [Final lossless PNG/alpha ZIP](/fsx/tmp/marshawn_4k_actor_cutout_20261108/outputs/two_clips_v2/TwoActor_4K_PNG_Alpha_Masters.zip):25,209,274bytes; SHA256`825603a86862a1f853ee159612821d97e9c1f9c5d5b7f1c90458cb8a57b92a86`.

Both clips retain3840×2024,24fps, ten frames and exactly5/12seconds; no extra frames are invented. All frames decode successfully. First/middle/last decoded far-background maxima are RGB0 for both clips. Root reviewed the four repaired native patches after correction; the larger true openings remain. Actual final masters prove all10 hybrid RGB/alpha pairs and the other seven LookCloser pairs hash-identical to the preceding reviewed version; only the three approved LookCloser frames change within the1205supported pixels. Original/staged20 source RGB files are unchanged. [Publication receipt](/fsx/tmp/marshawn_4k_actor_cutout_20261108/delivery/complete.json), [actual-master parity](/fsx/tmp/marshawn_4k_actor_cutout_20261108/ops/enclosed_hole_repair_v2/actual_masters_parity.json).

These masks remove exterior render debris; they do not reconstruct missing NeRF detail or establish perfect physical alpha. All ten GPUs were free after the finite mask jobs, and no old training restarted. [Final shutdown receipt](/fsx/tmp/marshawn_4k_actor_cutout_20261108/ops/final_CPU_shutdown_receipt.json) retains the original audit SHA `38cac2aeb557c7b2c4c52fad61ce8a6c75acaa2f683b11a190474d995bda54b3`. During the subsequently authorized native63 campaign, a hardware helper accidentally replaced the old `final_GPU_audit.json` with a newer idle-GPU snapshot; video, masks, models and source bytes are unaffected. [Recorded audit-path incident and replacement SHA](/fsx/tmp/marshawn_native63_24h_20261108/ops/old_audit_output_incident.json).
