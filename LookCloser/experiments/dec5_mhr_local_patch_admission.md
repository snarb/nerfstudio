# Measured-depth admission of local MHR jaw/neck proposals

## What was tested

Frame 001193, three frozen conformance strengths (0.25/1/4), original COLMAP unchanged. Candidate extraction is separately frozen at `/mnt/data/dec5_mhr_local_patch_candidates`: neutral lower-head/neck band 135..153 cm, unsafe prior facets excluded, fine subdivision, original-surface/boundary proximity, normal agreement, and proximity to an **actual opposite open edge**. No target camera constructs candidates or admission evidence.

This study compares direct two-view depth admission with bounded observed-seed interpolation. Result: 99/89/66 triangles survive, but **none repairs the requested 44-pixel hole**. Interpolation adds zero. No production promotion or model refitting.

Root: [/mnt/data/dec5_mhr_local_patch_admission](/mnt/data/dec5_mhr_local_patch_admission/request.json).

Important explicit provenance correction: the existing mask-control request binds repaired geometry SHA `46718391862351a38212e5602af4b685d424f48c2ab2f9099517af5cfa63d1f0`; these candidates instead preserve original TSDF SHA `316dc2b6a82c0d99a6e399d15d941a69d491896921bc82e4d88ea307bb358b03`. The old geometry receipt is **not** reused as if matching. Only its verified 001193 depth maps, calibrated train masks and the D_D measured-foreground override are rebound. That override was produced from train depth/color support without loading a candidate mesh. Actual candidate geometry is independently hash-bound and its original vertex/triangle prefix checked exactly.

Common guards, fixed across all three arms:

- Foreground mask support in at least two available train cameras, with no available-camera mask disagreement, at all three triangle vertices and centroid. Outside the renderer's native footprint domain is unknown. The mask override affects geometry admission only, not texture masks.
- Ten fixed barycentric depth samples per triangle. Direct admission requires at least two vertices with two observed-view votes and median sample votes at least two. Deterministic physical-camera depth references use the existing round-trip/parallax support rules, not frustum counts.
- No trusted free-space contradiction at samples: all four enclosing native depths must be valid, farther by over 0.003, and each corroborated by three other train cameras before a fractional sample vetoes the triangle.
- Interpolation is not the older nearby-anchor-vote shortcut. Original-surface neighborhoods select actual native measured depth samples, deduplicated by physical camera and integer pixel, then revalidated in at least three depth views. Seed-to-prior distance ≤0.0005; seed-to-original distance ≤0.001. Each query uses at most 24 seeds within 0.003 and normal dot ≥0.5. At least eight seeds, query inside their projected hull, a well-conditioned quadratic fit, leave-one-out P90 ≤0.0005 and predicted offset ≤0.0005 are required. All three proposed vertices must pass; semantic and free-space vetoes remain in force. These seed positions are actual unprojected depth samples, not merely nearby original mesh vertices.
- Both branches then undergo the same native 62-camera × integer/half-pixel ray veto. A visible added triangle is removed when a query-camera depth is over 0.003 farther and has three other corroborating depth views. Iterate to no implicated triangle, capped at eight passes. Original triangles cannot be removed.

All scene-distance thresholds use the calibrated normalized COLMAP gauge. The eight cameras previously reserved from **fitting** now participate in this explicitly 62-train-camera admission check; the final admitted mesh is therefore not independent of them. The three dataset held-out RGB cameras remain excluded. Missing measured depth remains unknown, not proof of free space or missing anatomy.

## Results

| Stage | Strength 0.25 | Strength 1 | Strength 4 |
|---|---:|---:|---:|
| Raw local triangles | 11,606 | 18,193 | 24,057 |
| Pass foreground masks | 1,489 | 1,540 | 1,526 |
| Direct sample admission | 99 | 89 | 66 |
| Verified observed seeds | 156 | 158 | 154 |
| Candidate vertices checked for interpolation | 1,012 | 1,042 | 1,019 |
| Certified interpolation vertices | 0 | 0 | 0 |
| Additional interpolated triangles | 0 | 0 | 0 |
| Final triangles, either branch | 99 | 89 | 66 |
| Native pruning passes | 1 | 1 | 1 |
| Requested original hole misses remaining | **44 / 44** | **44 / 44** | **44 / 44** |

Every branch's first 124 native checks has zero trusted free-space veto pixels; no triangle requires subsequent removal. Strict and interpolated mesh files are byte-identical within each strength. This is a passed measured-depth guard, **not** proof that all added surfaces are correct or useful.

Interpolation fails before quadratic residual evaluation: insufficient local usable seeds in 726/763/749 queries, and outside the seed hull in 286/279/270. Therefore no leave-one-out value is reported as zero or as a successful check. The result concerns this finite, original-vertex-guided measured seed pool; it is not proof that no denser measured neighborhood could exist.

Main's posthoc native clay review was independently inspected here for the requested hole and G_B. All three retain the original 44 missing pixels; G_B/M_B/E_B have zero visible added pixels in the reviewed face crops. The admission controls are a negative repair result, despite retaining small depth-compatible fragments elsewhere.

Main additionally inspected all four native panels: raw neck fringe islands disappear, the face is unchanged, and the verdict is **failure to repair the hole**, not catastrophic new geometry. Posthoc [facet attribution](/mnt/data/dec5_mhr_local_patch_admission/target_admission_diagnosis/result.json) finds zero matching raw facets for strengths 0.25/1 and 14 for strength 4. Every one of those 14 has at least two foreground-supporting views but also a mask disagreement; none reaches direct-depth or interpolation evaluation. This is not solely an open-edge bottleneck.

Native train [C_E](/mnt/data/dec5_mhr_local_patch_admission/mask_veto_witnesses/C004_E005_1210X7.png) and [E_D](/mnt/data/dec5_mhr_local_patch_admission/mask_veto_witnesses/E004_D005_1210L4.png) were independently inspected: the rejected sample cluster lies in visibly real background beyond the neck silhouette. These two views support keeping the rejection; a shadow or mask-error explanation is not justified there. The [witness receipt](/mnt/data/dec5_mhr_local_patch_admission/mask_veto_witnesses/result.json) retains all 62 camera counts and four marked native RGB panels. This diagnostic uses the requested location posthoc only and does not alter masks or admission.

![Original requested hole and three depth-admitted controls](/mnt/data/dec5_mhr_local_patch_admission/native_clay_review/requested_hole.png)

The main agent's [native review receipt](/mnt/data/dec5_mhr_local_patch_admission/native_clay_review/result.json) binds the actual original mesh, admitted meshes and diagnostic camera; that camera is posthoc only. RGB reconstruction was not run after the failed geometric usefulness gate. PSNR/SSIM/LPIPS are N/A for this clay-only admission study.

## Insights

The close, coherent MHR surface did not translate into a useful depth-admitted patch under this generic actual-open-edge proposal rule. Many raw facets fail real-view foreground or direct-depth support. The guarded survivors do not cover the requested hole, and local seed hulls do not justify extrapolation.

Preserve this negative control. Do not weaken a per-frame threshold, credit nearby votes as interior measurements, or treat native-veto passage as successful repair. A separately declared proposal-domain study can test whether requiring the nearest point's opposite edge to be open is overly restrictive; this experiment does not perform that change.

### Reproduction and checks

Producer: `scripts/admit_mhr_local_patch_depth.py`; replay: `scripts/audit_mhr_local_patch_depth.py`. Run in reconstruction Python with `OPENCV_IO_ENABLE_OPENEXR=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`. The explicit output root must be fresh. A two-thread scene wrapper preserves existing ray/pixel conventions without changing shared helpers. No GPU inference or rendering is used.

Admission completed in 71.76 seconds. Four synthetic tests cover direct evidence requirements, all-vertex certification, free-space/semantic vetoes and hull/offset rejection. Run `python -m pytest -o addopts='' -q tests/test_mhr_patch_admission.py`.

The [audit](/mnt/data/dec5_mhr_local_patch_admission/audit.json) replays 45,550 sample votes and footprint checks, 3,073 vertex certificates, measured seed pixels/votes, and all 744 final native camera/offset checks. It verifies all 62 source geometric depth hashes, exact saved original prefixes and strict/interpolated identity. Helper hashes, separate geometry/mask provenance, seed neighborhoods, certificate rejection reasons, retained proposal IDs, native-check logs and failed workspaces are retained.

The supplementary [final seal](/mnt/data/dec5_mhr_local_patch_admission/final_seal.json), produced by `scripts/seal_mhr_patch_admission_review.py`, verifies the audit snapshot and binds subsequent main native review, facet attribution, mask witnesses, their producers, this report, and the full final artifact inventory. Four synthetic tests passed.
