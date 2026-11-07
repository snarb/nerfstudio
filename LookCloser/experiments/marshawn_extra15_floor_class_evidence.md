# Proposed fifteen-row TRAIN floor-class evidence

What was tested: whether already reviewed, actually captured TRAIN class labels expose a missing signal for the retained floor/fringe. This CPU-only proposal leaves all69,888 original queries, native weights, targets and four UNKNOWN confidences unchanged. All standalone21 source fit weights remain0. No optimizer, field, GPU or new head checkpoint was created.

The packet has12 visible-floor class0 labels and3 MD shoe class1 labels. The three reviewed boot pixels have zero nominal original-mask coverage; their new positive class labels do not change those masks or any original target. Nearby original-label conflicts must be reported, never silently relabeled. The FD hair query remains diagnostic with weight0. Each of the16 actual native rays has nonzero native opacity and in-box capacity>.95. Pixel/source-review provenance, actual c83 native visibility, packed weights and the existing standalone21 packet context are bound; this is not a second capture and does not claim original128-ray allocator equivalence.

Results, current residual256 head88871: Qactor/nativealpha is the native-visible conditional actor posterior. Selected opacity is recomputed from original sigma/intervals at the fixed per-sample hard thresholds. These opacity values are UNLABELED diagnostics, never opticalalpha targets or evidence that a whole ray contains no actor. The existing native MD far-shoe, MDfloor04 and FDfloor03 review panels were visually inspected: the boot is real despite the nominal mask omission, while the selected floor patches show visible floor. This supports their class labels, not a ray-wide occupancy statement.

| TRAIN query | Class | Native alpha | Qactor/nativealpha | Selected opacity .1 | Selected opacity .5 |
|---|---:|---:|---:|---:|---:|
| FD_upper_hairtip_01 | 1 (diagnostic0) | 0.801968 | 0.729374 | 0.624551 | 0.621554 |
| FD_floor_01 | 0 | 0.842674 | 0.037672 | 0.032051 | 0.032051 |
| FD_floor_02 | 0 | 0.806891 | 0.046443 | 0.177220 | 0.000000 |
| FD_floor_03 | 0 | 0.662638 | 0.167082 | 0.191972 | 0.096500 |
| FD_floor_04 | 0 | 0.580156 | 0.001170 | 0.000000 | 0.000000 |
| MC_floor_01 | 0 | 0.830636 | 0.000073 | 0.000000 | 0.000000 |
| MC_floor_02 | 0 | 0.612905 | 0.000306 | 0.000477 | 0.000000 |
| MC_floor_03 | 0 | 0.485216 | 0.000386 | 0.000000 | 0.000000 |
| MC_floor_04 | 0 | 0.595586 | 0.003181 | 0.000000 | 0.000000 |
| MD_black_farshoe_01 | 1 | 0.766854 | 0.955060 | 0.758414 | 0.740349 |
| MD_black_farshoe_02 | 1 | 0.993845 | 0.994642 | 0.993414 | 0.992815 |
| MD_black_farshoe_03 | 1 | 0.980462 | 0.913468 | 0.972834 | 0.970449 |
| MD_floor_01 | 0 | 0.696809 | 0.108397 | 0.195398 | 0.088189 |
| MD_floor_02 | 0 | 0.699383 | 0.106676 | 0.148402 | 0.100230 |
| MD_floor_03 | 0 | 0.997840 | 0.052564 | 0.125590 | 0.037846 |
| MD_floor_04 | 0 | 0.826322 | 0.102564 | 0.479028 | 0.000000 |

The strongest observed floor responsibilities are FDfloor03 and MDfloor01–04; MCfloor rows are already nearly rejected. For example, MDfloor04 still produces.479028 selected opacity under hard.1 despite native conditional posterior.10256. The selected floor tau at these12 rays comes from source-visible weights; measured selected tau at EXACT zero native weight is0. This motivates new visible-class evidence rather than assigning arbitrary empty-ray labels or changing native sigma. It does not prove coverage of every heldout fringe.

An algebraic first-Adam-update probe (LR.03, epsilon1e-8, zero moments) compares the unchanged old camera objective with the proposed extra class sum divided by the ORIGINAL camera query count. No optimizer runs or parameters are changed. At the proposed initial learned7937+zero256 function:

| Camera | Mean floor posterior change, old only | With proposed classes | Added109-point threshold crossings |
|---|---:|---:|---:|
| FD | -0.000244434 | -0.000605317 | 0 |
| MC | 0.000000000 | -0.000028324 | 0 |
| MD | -0.000001177 | -0.000943560 | 0 |

All3 MD shoe posteriors rise with the added evidence (+.00175059/+.00074989/+.00196922). The minimum109-reference posterior under the MD update is.51103836 in BOTH old-only and augmented probes; the extras add no measured reference damage in this single step. This is a local diagnostic, not a claim about multi-step convergence, native physical quality, or whole-foot safety.

Insights and smallest proposed fit: run ONE new fresh2064 frozen7937+zero256 residual candidate, reusing the measured88871 control. Match seed42, Adam0/LR.03, all60 original camera choices and original objective exactly. On FD/MC/MD only, add `sum(extra nativealpha-weighted semantic CE)/old_camera_ray_count` as a separate reduction. The sole scientific factor is the15 additional reviewed class rows. No extra camera draw, ray reweighting, depth/opacity target, mask change, field update or heldout fitting is proposed. Retain the FD hair row at0.

First implement an explicitly data-bound schema2 fitter and qualify extrasOFF32 against the old worker, then extrasON32/resume32/independent64 with exact restored state and measured CUDA drift. Check all15 class rows, all109 reference points, true shoes and source/cache bytes before ROOT authorizes the single fresh2064 fit. Afterwards a separately released schema2 constructor must repeat the same3body+9detail physical gate and PSNR/SSIM/LPIPS. Current flags remain false and this proposal has no executable GPU route.

[Sealed CPU measurements, all16 rows and counterfactual probes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_extra15_floor_evidence_plan_v3/CPU_measurements_and_proposal.json), [explicit one-candidate fit proposal](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_extra15_floor_evidence_plan_v3/single_candidate_fit_proposal.json), [false-only review template](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_extra15_floor_evidence_plan_v3/root_review_TEMPLATE.json). Source script/helpers and input hashes are archived alongside the proposal. Existing original-class and floor/actor provenance is preserved; ROOT must review qualification before any new fit.
