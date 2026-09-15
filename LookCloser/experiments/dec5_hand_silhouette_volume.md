# DEC5 hand silhouette envelope: useful constraints, rejected surface replacement

## What was tested

Address the large hand/wrist failure at actual time **001037**, rather than
another small jaw-pixel change or camera crop. Construct a bounded implicit
occupancy envelope from six real train views (G/H columns, A/B/C rows), then
add six wider views: E/A, E/B, E/C, F/A, F/C, F/D. F/B remains held out and is
not read. No new PatchMatch job, model training, texture synthesis or movie
mutation occurs.

Fixed-profile train RGB, existing person masks, a fixed `R-B > 8` test, largest
connected component and radius-three closing define a coarse warm-object mask
against blue clothes inside portrait `[0,1400,500,1920]`. This includes the held
lipstick and some narrow room leakage; **it is not certified skin anatomy**.
The main agent inspected both six-view GT sheets and both mask sheets before
interpreting the surfaces. Strong motion blur is visible in the actual GT.

Previous triangulated hand landmarks supply only a loose world box, padded by
0.025 normalized units. Their earlier correspondence errors do not become
surface-confidence evidence. Grid spacing is 0.0004, with at least three
available train masks and no available negative silhouette vote. Compare zero
and three-pixel silhouette margins in the initial six-view pilot. AABB caps
are discarded. The target movie camera never defines the volume.

### New-prototype domain error and its guard

The initial implicit field used a negative sentinel where fewer than three
annotations were available. Marching cubes could turn that **unknown** region
into a false closing wall. More generally, an available-camera set can change
at a field-of-view/annotation edge without an anatomical surface there.
`silhouette_domain_surface.py` retains only triangles in cells with identical
camera-availability bits at all eight corners and at least three known views.

This is a conservative rejection guard, **not a complete surface estimator**:
removing those faces can leave open slits. The signed silhouette distance still
uses finite masks and does not independently certify contour uncertainty or
anatomical boundaries near an annotation edge. This new-prototype issue is
not asserted to be the cause of the original PatchMatch/TSDF movie holes.

## Results

| Envelope | Occupied voxels | Retained triangles | Domain-edge faces rejected |
|---|---:|---:|---:|
| Six cameras, raw, margin 0 | 176,426 | 102,732 | 0 |
| Six cameras, raw, margin 3 px | 191,564 | 107,116 | 0 |
| Same six-camera field, domain guard | 176,426 | 69,937 | 32,795 |
| Twelve cameras, raw, margin 0 | 125,054 | 86,365 | 0 |
| Same twelve-camera field, domain guard | 125,054 | 58,637 | 27,728 |

The repeated six-camera raw mesh is **byte-identical** to the initial zero-margin
mesh. Thus the removed wall faces are not explained by a different meshing run.
The audit independently resamples 2,048 field/availability values per inventory
and reconstructs every raw/guarded face in all four wide-study outputs.

The main agent inspected three initial geometric panels and all five wide-study
panels (G/A, H/A, E/C, F/D and the actual movie pose), in addition to the four
GT/mask sheets. The six-view envelope fits part of the front contour but inflates
badly from extra E/C and F/D views, which were **not used in its construction**.
Twelve views constrain the gross extent more tightly. Neither result restores
separate fingers or a coherent wrist: fused forms, unobserved surfaces and open
slits remain. Both are **rejected as a production surface replacement**.

The initial masks' rendered-depth coverage counts are fitting diagnostics only;
less black area does not establish correct anatomy. In particular, production
depth outside the warm-object mask mostly belongs to clothing, not false hand
geometry. No new face/full-frame PSNR, SSIM or LPIPS is computed, and no good
average metric is used to override the visible failure.

- [Actual additional train views](/mnt/data/dec5_wrist_wide_observations/001037/six_train_views_native.png).
- [Original six-view masks](/mnt/data/dec5_hand_silhouette_volume/001037/masks_native.png).
- [Additional six-view masks](/mnt/data/dec5_hand_silhouette_extra/001037/masks_native.png).
- [Extra E/C view exposes the six-view inflation](/mnt/data/dec5_hand_silhouette_wide/review/E004_C005_1210YM.png).
- [Actual moving-view geometry comparison](/mnt/data/dec5_hand_silhouette_wide/review/elevated_workaround_00099.png).
- [Explicit visual verdict](/mnt/data/dec5_hand_silhouette_wide/visual_review.json).
- [Replayed audit](/mnt/data/dec5_hand_silhouette_wide/audit.json).
- [Retained/input hash inventory](/mnt/data/dec5_hand_silhouette_wide/artifact_manifest.json).

Six focused tests cover unknown versus negative evidence, exact camera-bit
availability, insufficient views, partial annotation footprints, explicit mask
margin, artificial box caps and mixed-availability cells. The first additional
camera selector correctly failed its six-camera assertion because F/B is held
out; the fixed explicit six-train list uses F/D instead. That failure log is
retained. No held-out image was staged by the failed selector.

## Insights

The expanded train views provide genuinely useful constraints missing from the
original narrow selection. But an intersection of partial, blurred silhouettes
is not a reliable completed hand surface. It should constrain a depth/shape
proposal, not replace PatchMatch geometry by itself. More mask dilation cannot
recover unseen finger concavities and is not a justified cure for wrist holes.

For the next geometry attempt, use the broader real-view inventory as a check
on a depth-anchored local shape proposal, retaining reliable original fingers.
Do not spend a full 150-frame replay on these rejected shells. The published
dynamic-actor/dynamic-camera movie and prior verified jaw repair remain
unchanged; the full artifact-free objective remains open.
