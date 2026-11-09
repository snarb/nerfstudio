# Diagnostic radial cold native control

The fixedk4 + jointly fitted TRAIN scalar dataset preserves66 camera geometry/splits/masks. R0 is a cold native run with frequency OFF, loose AABB, near.052, true-gradient SfM LOS.01 clipped to that near, measured SfM opacity band.01, bounded-linear distortion.01 and soft photometric MVS52 distribution.01. The assumed finite floor-plane prior is removed. No NN layers/parameters were added; no eval RGB was used to fit the preprocessing.

This is a multi-factor test relative to actual trained MVS16k: radiometry, cold optimizer history, near/LOS trim, removal of assumed plane and E[rho] versus rho(E) MVS differ. It does not isolate radial correction alone.

Results below use actual1920×1012 float predictions and one comparison engine throughout: PSNR(range1), SSIM with explicit data_range1 and AlexNet LPIPS. The native controller's SSIM uses its default range; its values are deliberately not mixed into this table. Formal selection still uses native own-domain all3 PSNR with LPIPS tie within.07dB.

| Run | GT / prediction domain | PSNR | SSIM(range1) | LPIPS |
|---|---|---:|---:|---:|
| R0 2k | NEW corrected radial | 29.10916 | 0.789472 | 0.334303 |
| R0 2k | OLD scalar common | 29.46557 | 0.799420 | 0.327743 |
| R0 4k | NEW corrected radial | 29.88679 | 0.804012 | 0.292185 |
| R0 4k | OLD scalar common | 30.25527 | 0.813703 | 0.285716 |
| Actual MVS16k reference | OLD scalar common | 29.19285 | 0.819852 | 0.268069 |

The common comparison applies the known inverse radial/new-EV transform to raw float candidate outputs, then OLD scalar metadata; it fits no RGB. Both candidate and actual16k reference are scored against identical OLD GT. Inverse quantized NEWGT differs from OLDGT by about.33/255 mean and<1.20/255 max at2k; saturated predicted channels were0. Own corrected-domain metrics remain separate.

Root viewed all three full-view contacts and floor/black-region crops. From2k to4k, edges and common-domain metrics improve, but pale mat/floor patches and fine-detail blur remain. The4k common PSNR exceeds actual16k; its SSIM/LPIPS are still worse. This is continuation evidence, **not acceptance as clean**. Root is preparing an8k gate on the same R0 worker; R1 is a separate guarded-grid comparison under supervision. No new color variant is introduced.

Actual read-only readers preserved the field and stored active occupancy grid at both checkpoints. This does not certify unseen physical geometry or an ideal grid. R0 checkpoint SHAs:2k `f2765413898ecbf25cc4b40b7338a99ebfa8f39d7dbf8af6a2a241d27e510e87`;4k `01451388a276e01b0a8fa220d621ffdfe6e7334c2faa72c53bd164515998fc5c`.

[2k metrics and per-view crops](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/gate2000_HD_common/report.json), [4k metrics](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/gate4000_HD_common/report.json), [4k DC common](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/gate4000_HD_common/contacts/DC_common_old.jpg), [HC common](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/gate4000_HD_common/contacts/HC_common_old.jpg), [KA common](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/gate4000_HD_common/contacts/KA_common_old.jpg), [HC floor](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/gate4000_HD_common/contacts/HC_floor.jpg), [KA blackwide](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/gate4000_HD_common/contacts/KA_black_wide.jpg).

A bounded independent fit on the existing26,030 TRAIN observations /5,287 tracks gives unconstrained sharedk3.67354, originalfive disjoint TRACK-fold range3.652–3.697. Held-out mean-fold medianEV is.11147(k0),.05044(fixed4),.04798(free); RMSE.22279/.12974/.12911. Tail p90 does not improve over4. Fixed4's small remaining partial radial trend implies slight overcorrection (unweighted effectivek3.72867), not evidence for increasingk towards the separate static-card5.92. Materials/coverage confound optical interpretation. Keep fixed4; do not create another correction. [Read-only TRACK-fold report](/fsx/tmp/marshawn_001660_hd/report/shared_radial1660_train_PRECHECK/free_k_track_test.md).

[Stage README](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_STAGED/README.md) documents the22,000-attempt cold cap (terminal21999), same-process gates and actual zero-update proof. Native cold AMP65,536 overflowed its RGB branch;32,768 was finite without optimizer updates. Production retained automatic native scaler backoff and counts skipped attempts. Frozen code/config/data bindings remain in the stage receipts; no checkpoint or image was altered for this report.
