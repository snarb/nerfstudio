# Independent review: measured MHR conformance

## What was tested

Read-only review of the main agent's three fixed conformance strengths (0.25, 1, 4), not a new fitting variant or patch. Source: `/mnt/data/dec5_mhr_measured_conformance`; independent outputs: [/mnt/data/dec5_mhr_conformance_independent_review](/mnt/data/dec5_mhr_conformance_independent_review/result.json). Original COLMAP and source outputs were not modified.

Questions: are reserved-camera observations excluded from fitting, and are reported negative normal dots actual under-jaw folds or defects elsewhere?

## Results

Validation exclusion passes an explicit replay: remove all 1,600 reserved-camera anchor records **before** any correspondence, weighting or linear solve, then independently replay the strength-4 fit. Every output vertex is identical to the saved result (maximum difference 0). The producer also excludes these eight cameras from depth-support voting inherited from the earlier study.

This is a reserved-fitting-camera check, not unseen-from-baseline evaluation: the unchanged original COLMAP mesh used all 62 train cameras. The three dataset held-out RGB views were never used. Dense nearest-surface errors can improve through correspondence sliding, smoothing or overlapping surfaces; they do not establish semantic identity, visibility, free-space consistency or safe patch attachment. No independent confidence interval is inferred from those residuals.

### Normal changes and actual self-intersections are different

A negative dot between original and deformed triangle normals means a change of at least 90 degrees; in a 3D embedded surface it is not alone proof of triangle inversion or self-intersection. A rigid 180-degree rotation can also give a negative dot without any fold. The fitting method preserves triangle indexing but has no explicit nonintersection or orientation constraint.

We separately enumerate nonadjacent triangle intersections, excluding every shared-vertex pair, and verify each pair with a strict noncoplanar edge-through-triangle-interior witness (barycentric and segment margins 1e-6). Thus the crossings below are not merely normal changes, coplanar contacts or shared-edge adjacency.

| Surface | Negative normal dots | Strict transverse crossing pairs | Anatomical location of crossings |
|---|---:|---:|---|
| Input articulated prior | 0 by definition | 109 | Mouth/lips |
| Strength 0.25 | 32 | 183 | Mouth 120, nose 55, eye 3, ear-side 5 |
| Strength 1 | 32 | 163 | Mouth 118, nose 45 |
| Strength 4 | 15 | 114 | Mouth/lips only |

The input prior already has real lip crossings; these are **not** treated as valid. There are 163/158/111 changed pair IDs relative to the input at strengths 0.25/1/4, but a changed pair ID is not a new anatomical defect region: strength 4 changes the already intersecting lip contact pattern. We do not claim 111 newly defective regions.

All normal reversals lie above neutral y=153 cm. Strength 4's 15 reversed triangles span neutral y=155.464..158.323 cm around the nose/lip region. Some weaker-arm reversals extend to eye/ear regions. None lies in the proposed neutral 135..153 cm lower-head/neck band. Normal reversal and crossing sets are not interchangeable: none of strength 4's reversed triangles participates in its changed intersection-pair set.

### Clearance from the requested under-jaw

There are zero strict crossing pairs touching the neutral 135..153 cm band in the input or any conformed arm. More strongly, the minimum neutral vertex y among **all vertices of intersecting triangles** is 155.211/155.077/155.077 cm for strengths 0.25/1/4; these are not triangles straddling the lower-band cutoff.

Minimum actual intersecting-triangle surface distance to the 24 original visible rim samples at the requested hole is 0.006739/0.006734/0.006727 calibrated scene units. Strength 4's reversed-triangle surface is at least 0.010100 scene units from that rim. These measurements use surface closest points, not only centroid distances.

Six native diagnostic panels were inspected: strength 0.25 G_B; strength 1 G_B/M_B; strength 4 G_B/M_B/E_B. Red marks visible negative-normal triangles; amber marks triangles in changed intersection pairs. They confirm nose/eyelid/ear distortions in weaker variants and lip crossings in strength 4, away from the requested under-jaw. All are prior-only diagnostics, not accepted replacement renders.

![Strength 4 native G_B: normal changes red, changed intersection pairs amber](/mnt/data/dec5_mhr_conformance_independent_review/smooth400_G004_B005_1210FG.png)

![Strength 0.25 G_B: additional nose/eye defects](/mnt/data/dec5_mhr_conformance_independent_review/smooth025_G004_B005_1210FG.png)

## Insights

Do not replace the whole face with any of these priors. The conformance fit improves geometric agreement but does not prevent existing or new self-crossings.

The detected folds/crossings do **not** establish a need to change the conformance rule merely to address the requested under-jaw: they are outside that bounded anatomical region with measured clearance. A separately constructed local candidate still needs its own original-mesh intersection, attachment, visibility, free-space, semantic and native-view checks. This review grants no patch or production approval and does not reuse baseline lip crossings as valid geometry.

Three synthetic tests pass: a genuine transverse crossing, disjoint/coplanar controls, and a rigid normal reversal without intersection. Replay, triangle IDs, area ratios, pair lists, neutral-coordinate localization and strict witnesses are retained. The [seal](/mnt/data/dec5_mhr_conformance_independent_review/seal.json) binds reviewed source fits and independent artifacts. No PSNR/SSIM/LPIPS applies to this geometric-only review.

Reproduce with reconstruction Python and `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1`: run `review_mhr_conformance_independent.py` in a fresh root, `check_mhr_conformance_crossings.py`, then `seal_mhr_conformance_independent_review.py`. Tests: `python -m pytest -o addopts='' -q tests/test_mhr_conformance_crossings.py`.
