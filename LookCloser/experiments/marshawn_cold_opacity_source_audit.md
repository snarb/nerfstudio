# Marshawn cold8k KA opacity-hole source audit

What was tested: independently project actual frozen-field density peaks and conditional termination medians for KA pixel `(242,747)` into all 63 training cameras. Compare actual source GT, nearest 9022 SfM witnesses, trusted free intervals, and exact continuous positive surface bands. No model, optimizer or occupancy changes.

Results: the cold batch1M/scheduler22k checkpoint `6962dca6…fe1` has missing floor opacity. Native opacity is 0.16054, all-cells native quadrature 0.51173, converged uniform32768 0.525427; the uniform13k checkpoint `26f6c484…dc4` is approximately opaque on this same ray. Conditional cold radiance is ordinary brown/gray, so black background through incomplete opacity causes the dark smudge. Grid culling amplifies the deficit; it does not fully explain it.

| Actual traced sample | Independent source evidence | Nearest positive-band distance, prepared world units |
|---|---|---:|
| Cold near-camera maximum sigma | Same camera/rig housing in AC, AD, BD, DD; nearest tracked witness p770 is on its mount | 0.04975 |
| Cold conditional termination median | Projects onto ordinary floor or foreground boots; no distinctive dark object. Physical visibility of deep XYZ is unproven | 0.23722 |
| Uniform13k conditional median | Clear floor in AA, AB, BB and additional training views; BD is boot-occluded | 0.09376 |

The near-camera sigma maximum is distinct from the main termination mass and adds opacity. It is not evidence that the black floor smudge is an extra dark physical object. Cold median z is −0.4453, below the observed-point minimum z −0.335855. This is a gap in observed support, not proof that the volume is empty. No nearby erroneous positive track was identified. The nearest trusted-free segment is 0.2434 from the cold deep median; no direct negative-prior conflict was established either.

The approximate 15-point cloth-plane reference is extrapolated here: all nine ray/plane intersections lie outside its support XY hull. Other uniform13k rays can also terminate below that plane. Do not use this plane for hard depth labels or carving.

Evidence and actual manual review:

- [Source audit and all projections](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/train_source_audit/audit.json).
- [Near-camera peak source GT](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/train_source_audit/cold8k_242_747_uniform32768_peak.jpg).
- [Cold conditional median source GT](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/train_source_audit/cold8k_242_747_uniform32768_q50.jpg).
- [Uniform13k conditional median source GT](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/train_source_audit/uniform13k_242_747_uniform32768_q50.jpg).
- [Manual receipt, bound to all 63 actual train PNG hashes](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/train_source_audit/manual_review_occupancy.json).
- [Exact continuous positive-band proximity](/fsx/tmp/marshawn_001660_hd/native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/train_source_audit/positive_band_support.json).

Track-camera membership comes from the actual SfM observation graph, but `observations.npz` UV are calibrated projections of triangulated XYZ, not independent raw feature detections. Cyan markers show those reprojected tracked points. A zero reprojection residual against this cache verifies the calibration bridge only; it does not independently validate track geometry. GT frustum inclusion likewise does not prove visibility.

Insight: retain uniform13k as the safer current candidate. The measured cold-field opacity failure is real despite stronger global metrics; its exact optimization cause remains unresolved. No additional training or carving follows from this audit. Floor cleanliness remains incomplete.
