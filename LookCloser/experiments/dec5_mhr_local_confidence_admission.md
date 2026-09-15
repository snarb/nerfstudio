# Wider local MHR proposals still do not repair the jaw hole

## What was tested

Frame 001193, the same three frozen measured-conformance priors and original COLMAP mesh. This separate control removes only the candidate requirement that the nearest original surface point lie by its actual opposite open edge. All other local geometry constraints, train masks, observed-depth guards, interpolation certificates and native vetoes are unchanged from the [open-edge admission control](dec5_mhr_local_patch_admission.md).

**Result: all six guarded outputs leave the requested 44 missing pixels unchanged.** Thousands of depth-compatible triangles mostly overlap existing local geometry. Certified interpolation contributes 11/15/34 additional final triangles, but no requested-hole benefit; tiny occlusion changes elsewhere are retained as failures to establish usefulness. Nothing is promoted.

Output: [/mnt/data/dec5_mhr_local_confidence_admission](/mnt/data/dec5_mhr_local_confidence_admission/request.json). Candidate domain: [/mnt/data/dec5_mhr_local_confidence_domain](/mnt/data/dec5_mhr_local_confidence_domain/request.json). The wrapper binds its own hash, the frozen original admission producer/helper hashes, both candidate receipts and explicit paths. It extends request provenance only; it does not substitute an admission function or threshold.

The original vertex and triangle prefix is exact. Surface distance ≤0.002, open-boundary vertex distance ≤0.003, normal dot ≥0.25, neutral band 135..153 cm, maximum subdivided edge 0.00075, minimum centroid distance 0.00002, and exclusion of prior crossing/reversed facets remain fixed. These normalized scene-distance limits are not millimetres.

Direct admission still requires real depth agreement at ten samples: at least two vertices with two observed-view votes and median votes ≥2. Masks require ≥2 available foreground views and no available disagreement. Interpolation requires all three vertices inside a local hull of ≥8 actual native measured seeds, each revalidated in ≥3 depth views; radius ≤0.003, seed-to-prior ≤0.0005, seed-to-original ≤0.001, normal dot ≥0.5, quadratic conditioning and leave-one-out P90/offset ≤0.0005. Fractional footprint and native free-space contradictions remain vetoes. Both branches undergo the same 62-camera × integer/half-pixel native guard until no added triangle is implicated, at most eight passes.

Only the verified mesh-independent measured-foreground override and depth/mask artifacts are rebound from the older mask-control request. Its repaired mesh receipt is not inherited: actual original mesh SHA starts `316dc2b6`, not `46718391`. Admission uses all 62 train cameras, including the eight formerly reserved from prior fitting; these are no longer independent validation views for the admitted result. The three dataset held-out cameras and target RGB remain excluded. Requested-camera rays below are posthoc diagnostics only. Missing depth is unknown, not proof of missing skin.

## Results

| Stage | Strength 0.25 | Strength 1 | Strength 4 |
|---|---:|---:|---:|
| Raw wider-domain triangles | 76,507 | 95,295 | 106,807 |
| Pass train masks | 20,077 | 18,636 | 17,154 |
| Direct initial triangles | 12,628 | 10,924 | 9,326 |
| Verified measured seeds | 1,979 | 1,856 | 1,694 |
| Certified vertices / queried | 6,750 / 11,618 | 6,054 / 10,726 | 5,358 / 9,852 |
| Certificate-only initial triangles | 11 | 15 | 37 |
| Native removals, direct / interpolated | 143 / 143 | 339 / 339 | 186 / 189 |
| Final direct triangles | 12,485 | 10,585 | 9,140 |
| Final interpolated triangles | 12,496 | 10,600 | 9,174 |
| Native passes, either branch | 2 | 2 | 2 |
| Requested original misses, either branch | **44 / 44** | **44 / 44** | **44 / 44** |

Admission took 168.64 seconds, CPU only. The final pass has zero trusted native free-space contradictions. That is a guard pass, not proof of geometric correctness or successful completion.

Main inspected all four strict native panels (G_B/M_B/E_B/requested hole): the changes are mainly lower-neck facet shading; no requested-hole improvement. Independent inspection here of G_B and the requested-hole panel agrees. Original facial detail is preserved, but duplicate local surfaces are not a useful repair.

![Original and three guarded wider-domain controls](/mnt/data/dec5_mhr_local_confidence_admission/native_clay_review/requested_hole.png)

The [strict-versus-interpolated comparison](/mnt/data/dec5_mhr_local_confidence_admission/branch_difference/result.json) tests three native face crops and the complete moving frame for each strength, with a 1e-8 depth-difference threshold. All fixed-ROI depths are identical between branches. No comparison gains or loses a hit.

| Branch difference | Strength 0.25 | Strength 1 | Strength 4 |
|---|---:|---:|---:|
| Changed pixels G_B / M_B / E_B crops | 0 / 0 / 0 | 0 / 0 / 0 | 3 / 1 / 2 |
| Changed pixels, full moving frame | 2 | 0 | 3 |
| Largest common-ray depth change, moving frame | 0.015220 | 0 | 0.001045 |
| Changed pixels, fixed requested-hole ROI | 0 | 0 | 0 |

Strength 0.25's two changed portrait pixels `(541,1048)` and `(541,1049)` shift from depths 0.69013/0.69028 to 0.67506. Native inspection places this at a ragged local occlusion boundary; **the 0.0152 difference is not numerical roundoff**, despite only two pixels changing. Strength 4 changes `(517,1102)`, `(610,1218)` and `(611,1218)`. Exact locations, depths and component crops are retained in the [supplementary location receipt](/mnt/data/dec5_mhr_local_confidence_admission/branch_difference_locations/result.json). These are no demonstrated repair benefit.

![Two-pixel interpolation occlusion change, strength 0.25](/mnt/data/dec5_mhr_local_confidence_admission/branch_difference_locations/smooth025_component1.png)

No RGB render is justified after this geometric usefulness failure. PSNR, SSIM and LPIPS are N/A, not zero.

## Insights

Removing the open-edge restriction expands the domain and produces actual passing seed certificates, but does not make the admitted surface cover the missing ray region. This separates **certificate implementation working on supported neighborhoods** from **a useful missing-jaw proposal**.

In the previous control, main inspected all four native mask-witness panels (C_E/D_E/E_D/E_E), and this agent independently inspected C_E/E_D: rejected first-prior-hit points project into real background beyond the neck contour. Do not relax that guard or call the disagreement a shadow segmentation error. Conversely, a wrong front layer on a ray does not prove every deeper layer is background. The distinct main ray-interval diagnostic must be interpreted separately; this control tests unchanged priors, not silhouette-aware refitting.

Preserve all six outcomes and both branches. More accepted triangles and zero measured free-space vetoes do not establish useful inferred anatomy. No mask edits, new fit, temporal transfer, production geometry or active 6K changes were made.

### Reproduction and audit

Run `scripts/run_mhr_confidence_domain_admission.py` in a fresh fixed output root, then the same command with `--audit`, using reconstruction Python and `OPENCV_IO_ENABLE_OPENEXR=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`. The original admission producer and helper files must match the original control's hashes. `scripts/review_mhr_admission_branch_difference.py` and `scripts/localize_mhr_branch_difference.py` generate posthoc branch evidence; main's frozen `scripts/review_mhr_depth_admitted_patches.py` provides the original/strict comparison.

Six tests pass across `tests/test_mhr_patch_admission.py` and `tests/test_mhr_confidence_domain_wrapper.py`, including explicit verification that the wrapper preserves admission functions, constants and original request data. The frozen [numeric replay](/mnt/data/dec5_mhr_local_confidence_admission/audit.json) passed: 558,670 sample votes/footprints, 32,196 vertex certificates, real measured seed positions/votes, exact original prefixes and 744 final native checks. The [final seal](/mnt/data/dec5_mhr_local_confidence_admission/final_seal.json) additionally asserts identical admission protocols after excluding candidate/provenance fields, verifies unchanged candidate locality constants, and binds review producers, all native panels, both-branch differences, this report and the final artifact snapshot.
