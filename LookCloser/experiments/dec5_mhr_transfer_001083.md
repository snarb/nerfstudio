# DEC5 001083: fresh-pose transfer of frozen local MHR completion

## What was tested

A third, substantially different head pose, using the same fitting and local measured-confidence admission protocol as001193/001195. No fitted pose or prior geometry was copied. Only neutral topology/model assets and frozen functions were reused. This bounded test leaves the delivered6K video, original images, camera path, production meshes and defaults untouched.

Fresh input/fitting roots: [/mnt/data/dec5_mhr_transfer_001083_inputs](/mnt/data/dec5_mhr_transfer_001083_inputs), [/mnt/data/dec5_mhr_transfer_001083](/mnt/data/dec5_mhr_transfer_001083). Production-base completion: [/mnt/data/dec5_mhr_completion_001083](/mnt/data/dec5_mhr_completion_001083). Explicit specifications: [prior](/mnt/data/dec5_mhr_transfer_001083_spec.json), [completion](/mnt/data/dec5_mhr_completion_001083_spec.json).

Production mesh SHA-256 `46bb17f3e4cf9b85602154610a09e3086e546cc0b02511e09ea69236034164fd`, metadata/calibration, all62 geometric depth maps and existing independent D/D measured-mask override passed preflight. The override already exists from the same depth/color/three-other-foreground algorithm; it was not hand-substituted or regenerated. A first preflight failed because the new wrapper indexed the mask path instead of the cached model path during a hash check. That orchestration bug was corrected before creating the preflight root; the failed log is retained, and no model/data changed.

The fitting chain is fresh canonical similarity from001083's calibrated train-only RGB landmarks; MHR similarity/head20/named neck6 controls; measured smooth100 conformance; identical10-step and100-step silhouette fitting from the original per-time conformance base. No model-z is used as metric depth. All fitting/depth-voting/silhouette constraints exclude the same eight reserved train cameras. Original COLMAP and the measured override retain all62 provenance; subsequent admission uses all62. The true three held-out views never supply input or selection.

The generic [completion CLI](dec5_reusable_local_mhr_completion.md) is used without editing it. Same anatomical band135..153, all unsafe parent facets excluded, subdivision ≤.00075, surface/boundary locality .002/.003, normal dot≥.25 and centroid gap≥.00002. Both strict actual multiview-depth support and certified observed-seed interpolation are followed by the unchanged62-camera×integer/half-pixel measured-free-space veto. No target-defined fit/candidate domain or threshold adjustment.

Review uses the **actual current cinematic request's001083 moving camera**, not the old elevated001193/001195 stress pose, plus F/E,M/B,C/E,G/B train cameras. Moving crop is projected from fixed neutral head/neck coordinates under the fresh initial alignment; train crops come from fresh landmarks. These crops and enclosed ray-miss components are posthoc diagnostics only. A ray miss—even enclosed—is not automatically missing anatomy; real background gaps and cast shadows remain legitimate.

## Results

Fresh face landmarks detected62/62 cameras. Canonical alignment used7,522 fitting and1,077 reserved measured anchors. MHR anchors total12,400, with1,600 reserved and6,200 coarse neck-group candidates. Actual final neck6 associations retain5,400 fitting face and5,398 fitting neck-group samples. The inspected G/B anchor panel has a few ear samples and cyan points across neck/clavicles/chest: that group must not be described as6,200 guaranteed underside measurements.

| Stage | Reserved neck surface-distance P90 |
|---|---:|
| Head20 | .004704 |
| Head20 + neck6 | .002836 |
| Measured smooth100 | .000156 |

Initial and fitted G/B native panels, final C/E,M/B,E/D clay and G/B projected silhouettes were actually inspected. Pose alignment is plausible, but full-prior facial identity remains poor, with eye/nose folds; no whole-prior replacement is accepted. The same100-step silhouette fit reaches its cap, **not convergence** (unconstrained step .000474). Maximum displacement is .031534. Final topology has155 strict crossing pairs (7new) and85 normal changes over90°; these are different diagnostics and all affected candidate parents are excluded.

On the fixed11,498-sample reserved silhouette cohort, mean excess over the unchanged2px tolerance falls16.6675→.001754px; outside samples2,295→23. Fitting-cohort mean17.1036→.016657px, outside16,685→669 on77,992 fixed samples. These nearest-surface/silhouette results do not themselves prove a missing-surface repair.

| Local admission stage | Count |
|---|---:|
| Unsafe anatomical parent facets excluded | 72 |
| Subdivided facets | 741,376 |
| Local facets before centroid gate | 92,789 |
| Centroid rejections | 1,773 |
| Raw candidates | 91,016 |
| Semantic-admitted | 31,314 |
| Strict initial → final | 20,367 → 20,277 |
| Certified interpolation initial → final | 20,928 → 20,838 |
| Independently validated observed seeds | 1,753 |
| Certified vertices / queries | 9,011 /17,657 |

Admission took54.89s. Both branches remove90 triangles then0, checking all124 camera/offset pairs in the terminal pass. The original61,940 vertices and120,503 triangles remain exact prefixes. Independent replay passed313,140 support/footprint samples,17,657 certificates and248 final native checks. Head audit replays all12,400 measured anchors and three model forwards; conformance arrays and all100 silhouette iterates replay exactly. Four bounded tests pass.

All five native clay triplets were actually inspected by this agent and independently by the parent. No convincing visible repair appears; no broad new fold appears either. Underchin fringe remains in F/E and C/E. Current-moving enclosed misses3→3 with no new full-frame hits; F/E2→2 with2new hits; C/E has no enclosed misses and2new hits; M/B1→1 and G/B1→1 with no new hits. No-loss is not counted as repair.

All 15 matched current-policy RGB renders completed serially on CPU with two threads (about 11 minutes total; not a CPU/CUDA-equivalence benchmark). All five native triplets and all six interpolated side-effect crops were actually inspected. Both branches have identical aggregate counts below; their meshes are not identical.

| Current-policy RGB view | Enclosed misses, baseline → either branch | New colored hits | Newly black pixels | Nearer depth >.003 | Maximum nearer depth |
|---|---:|---:|---:|---:|---:|
| Current moving | 3 → 3 | 0 | 2 | 0 | .002382 |
| F/E | 0 → 0 | 2 | 0 | 3 | .003578 |
| M/B | 1 → 1 | 0 | 0 | 0 | .002939 |
| C/E | 0 → 0 | 2 | 0 | 1 | .003807 |
| G/B | 1 → 1 | 0 | 0 | 0 | .001848 |

No branch loses an existing current-policy depth hit, introduces untextured new geometry, or moves common hits farther by more than .003 in these five views. The two moving newly black pixels are at portrait coordinates (96,1389) and (96,1391), on the already jagged shoulder/neck silhouette—not an interior jaw defect. The four >.003 nearer pixels across F/E and C/E are shoulder/collar fringe changes, not evidence of a useful under-cheek repair. Shadow and existing underchin fringe remain. Tiny contour regression is retained as a measured failure; no broad new fold was visible in the reviewed views.

The F/E clay/RGB enclosed-count difference is **not hole closure or raycast roundoff**. The existing production wrapper applies source foreground masks when the target is a named train camera; unmasked clay does not. Read-only replay found 10,330 original F/E hits excluded by that mask and exact agreement of all common depths. The two clay enclosed misses at (830,1031),(831,1031) are still zero in RGB depth, but now connect to the excluded background component and cease to be *enclosed*. The final audit replays this policy exactly for all five baselines and all 15 RGB metric rows. These metrics must be read within each matched pipeline, not across masked and unmasked denominators.

Native evidence: [current moving](/mnt/data/dec5_mhr_completion_001083/native_rgb/current_moving.png), [F/E](/mnt/data/dec5_mhr_completion_001083/native_rgb/F004_E.png), [M/B](/mnt/data/dec5_mhr_completion_001083/native_rgb/M004_B.png), [C/E](/mnt/data/dec5_mhr_completion_001083/native_rgb/C004_E.png), [G/B](/mnt/data/dec5_mhr_completion_001083/native_rgb/G004_B.png). Side effects: [moving contour pixel 1](/mnt/data/dec5_mhr_completion_001083/native_rgb/current_moving_interpolated_newly_black_1.png), [pixel 2](/mnt/data/dec5_mhr_completion_001083/native_rgb/current_moving_interpolated_newly_black_2.png), [F/E nearer 1](/mnt/data/dec5_mhr_completion_001083/native_rgb/F004_E_interpolated_nearer_1.png), [2](/mnt/data/dec5_mhr_completion_001083/native_rgb/F004_E_interpolated_nearer_2.png), [3](/mnt/data/dec5_mhr_completion_001083/native_rgb/F004_E_interpolated_nearer_3.png), [C/E nearer](/mnt/data/dec5_mhr_completion_001083/native_rgb/C004_E_interpolated_nearer_1.png). See the [actual visual-review manifest](/mnt/data/dec5_mhr_completion_001083/visual_review.json), [generic replay audit](/mnt/data/dec5_mhr_completion_001083/audit.json), and [final input/output seal](/mnt/data/dec5_mhr_completion_001083/final_seal.json).

### Reproduction and provenance

Use `/home/brans/repos/nerfstudio/.venv/bin/python`, `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1`. Only `infer` and `semantic` use the existing isolated `/home/brans/lookcloser_temp/mediapipe_hand_env/bin/python`. No new package or model download was needed.

Run `scripts/transfer_local_mhr_prior.py STAGE --spec PRIOR_SPEC` with stages `preflight`, `stage_rgb`, `infer`, `init`, `semantic`, `anchors`, `fit`, `articulation`, `review_head`, `conform`, `silhouette10`, `silhouette100`, `review_final`, `audit_head`, `audit_conformance`, `audit_continuation`. Actual prior inspection and a hash-bound `prior_visual_review.json` are required before `seal_prior`. `scripts/review_local_mhr_transfer.py initial --spec PRIOR_SPEC` adds the initial alignment canary.

Then use `run_local_mhr_completion.py init --spec COMPLETION_SPEC --output NEW_ROOT`, followed by `build`, `check`, `admit`, `audit` with the same output. `review_local_mhr_transfer.py {clay,render,rgb} --spec PRIOR_SPEC --completion NEW_ROOT` generates five matched views; completed per-view RGB reviews must not be overwritten. The optional `--views` selects already declared review cameras, never fitting data. Final `audit_local_mhr_transfer.py --spec PRIOR_SPEC --completion NEW_ROOT --report REPORT --tests tests/test_local_mhr_transfer.py` requires actual final visual review and verifies all15 render receipts, calibration/source policy, geometry prefixes, priors, adapters and images.

## Insights

**Negative transfer for visible repair at 001083.** Fresh-pose fitting and unchanged confidence admission execute successfully, but neither branch visibly improves the reviewed head/neck defects. The 20,277/20,838 added triangles yield only tiny fringe changes in these cameras and two newly black moving contour pixels. Nonempty admitted geometry and low reserved-fit residuals are not evidence of useful repair. No broader temporal usability or production acceptance is inferred, and no additional fitting variant was launched.

The positive local-hole results at 001193/001195 do not generalize automatically to this substantially different pose. This prior supplies inferred geometry, not measured truth; original geometry remains intact. The report deliberately separates method execution, geometric safeguards and visible benefit. Hair is neither fitted nor evaluated; original texture seams/fringe are not solved by this anatomical prior. Independent PSNR/SSIM/LPIPS are N/A because no held-out reference fidelity evaluation is performed. Delivered 6K artifacts and all production defaults remain unchanged.
