# DEC5 001193: surface-interior silhouette fitting

## What was tested

The [previous control](dec5_mhr_clearance_and_silhouette_weight.md) exposed a
specific sampling gap: in train camera B/B all six vertices of the affected
faces project inside the silhouette, but points between them project outside.
Vertex-only penalties therefore do not represent those surface excursions.

Two isolated controls use the same warm geometry, anatomical active domain,
54 fitting / eight reserved cameras, masks, measured-depth anchors, 40 outer
iterations, .001 displacement bound, area/normal safeguards and all-pair guard:

- `surface`: vertex weight16 plus surface weight16. Every face touching an
  active vertex contributes its three edge midpoints and centroid. Samples are
  fixed in barycentric coordinates and selected without target rays or RGB.
- `vertex32`: the previous vertex-only objective with weight32, a strength
  control against attributing a larger silhouette penalty to surface sampling.

Each surface sample's projection/SDF derivative is distributed to its incident
active vertices through its barycentric weights. Fixed vertices still contribute
to sample position but have no variable columns. Positive SDF excess uses the
existing robust scale8px, sigma2px, and normalization by fixed sample count times
fit camera count. Samples are equally weighted, not area weighted; shared-edge
midpoints occur once per incident face. This changes spatial weighting compared
with the vertex control and is explicitly not an identical-quadrature ablation.

Entry point: `scripts/run_mhr_surface_sampling_control.py --arm surface` (or
`vertex32`). New roots are `/mnt/data/dec5_mhr_sampling_surface` and
`/mnt/data/dec5_mhr_sampling_vertex32`; old geometry/video and runner defaults
remain unchanged. `--dry-run` validates exact-source adapters without fitting.
Executed optimizer text and new helper/wrapper hashes are retained in each root.

Finite sampling is not a continuous silhouette-containment guarantee. A sample
at a stationary SDF maximum can detect an excursion with zero local gradient;
the regression test explicitly preserves that limitation instead of claiming
that detection always implies a corrective direction.

## Results

The surface control completes 40 iterations in 255.67 seconds. It stops at the
iteration cap, not demonstrated convergence; late steps are heavily reduced by
the unchanged nonlinear signed-area guard. The vertex32 control stops after 11
recorded iterates in 78.28 seconds because every permitted backtracking step
would reduce a triangle's area below .25 of its warm area. Consequently this
comparison includes different guarded stopping trajectories; it does not compare
fully converged optima.

The [fixed vertex review](/mnt/data/dec5_mhr_sampling_review/result.json) compares
54,193 common fitting and 8,048 reserved vertex/camera samples. The extra
[surface-cohort review](/mnt/data/dec5_mhr_sampling_surface_cohort/result.json)
uses exactly the same canonical barycentric samples and common availability
across vertex16, vertex32, and the surface control: 465,172 fitting and 69,109
reserved camera/sample pairs.

| Fit | Vertex outside: fit / reserved | Surface outside: fit / reserved | Surface mean positive SDF: fit / reserved (px) | Residual first-hit binary passes /30 |
|---|---:|---:|---:|---:|
| Vertex16, previous control | 299 / 13 | 2836 / 199 | .050184 / .007487 | 14 |
| Vertex32, strength control | 489 / 26 | 3942 / 294 | .054320 / .009461 | 7 |
| Vertex16 + surface16 | 116 / 8 | 1859 / 120 | .043133 / .005272 | 23 |

Surface-sample maximum positive SDFs remain 99.64014 / 46.77860 px in all arms.
The fitting domain includes faces touching active vertices; some sampled edge
points depend only on fixed vertices. Whole-domain statistics include inherited
uncorrected excursions, and the experiment does not certify the whole prior.

Six target-ray parent-triangle checks now pass versus zero in both vertex-only
controls. These are checks at 30 diagnostic rays, not a count of distinct parent
triangles or admitted production pixels. Surface fit final minimum relative
area is .57609, normal cosine .52832, maximum displacement .001, no new original
normal reversal, and 182 strict/all reported pairs (unchanged warm inventory).

Parent directly viewed all four PNG comparisons in the vertex review: the
residual crop plus C/E, G/B and M/B. The local under-jaw shape changes modestly,
without an obvious new large fold in those views. Existing eyelid/lip folds,
triangulation and neck irregularities remain; this is not full-prior approval.

The [remaining-point attribution](/mnt/data/dec5_mhr_sampling_residual_diagnosis/result.json)
replays the saved mask veto counts exactly. Seven rays still fail: A/C vetoes
two and B/B six, with one overlap, all in fitting cameras. A/C's two vetoes have
nonpositive bilinear SDF, exposing the known binary/bilinear boundary mismatch;
B/B still has five positive-SDF points, reaching 1.04962px. Therefore that
mismatch alone does not explain the remaining error. Local vertex displacement
is only .00017494; none approaches the .001 cap.

37 focused tests pass, including the new barycentric chain rule with fixed
vertices, finite differences through actual camera projection, an interior
excursion missed by vertices, and both explicit source adapters. The executed
optimizer records were checked: only `surface` calls the new surface term;
`vertex32` actually uses weight32 in the numerical expression.

### Measured admission and actual native RGB

The local builder preserves the original 61,860 vertices / 120,073 triangles
exactly, adding 700,108 raw proposals. Semantic-only filtering keeps 557,515;
on the fixed 30 rays this mesh hits 22 and misses eight. The unchanged measured
admission then runs for 259.01 seconds: 305,316 strict initial proposals and
16,974 additional certified proposals. Both branches require four native
passes. Final strict / interpolated triangle additions are 305,126 / 322,028,
with zero final measured-free-space violations. Certificates use 3,362 verified
observed seeds and accept 154,334 of 293,397 queried vertices.

`admit_mhr_corrected_candidates.py` explicitly binds this new prior to the frozen
production base; it does not bypass the old recipe pin. The independent
`audit_mhr_corrected_candidates.py` replays **5,575,150** depth/footprint samples,
**293,397** certificates and **248** final native checks. Its output inventory
deliberately excludes `rgb*` paths so geometry replay can run concurrently with
rendering. RGB receipts/hashes are checked separately, not implicitly covered
by the geometry audit.

`render_mhr_corrected_candidate.py` uses the **current production** texture policy
(incidence power2, unwarped and source-quality wrappers), not the older power8
renderer. Baseline/interpolated variants use identical cameras, source RGB,
color profiles and fixed exposure. Two isolated spawned CPU processes handle
the F/E and old-moving views concurrently; each uses two compute/raycast threads
and one source loader. This changes scheduling, not numerical rendering.
The baseline frames take 46.2 / 42.4 seconds; F/E interpolation takes 82.8s.
This is a measured parallel execution, not CPU/CUDA-equivalence evidence.

These are **1080×1920 diagnostic renders**, not replacements for the delivered
3456×6144 video. No delivered video frame or production mesh is modified.

| Native view | Fixed hole colored hits /30 | New hit pixels / uncolored | Newly black RGB | Lost geometry | Common pixels with nearer geometry |
|---|---:|---:|---:|---:|---:|
| F/E baseline | 0 | — | — | — | — |
| F/E corrected | **20** | 36 / 0 | 0 | 0 | 1301 |
| Old moving corrected | N/A, different rays | 102 / 10 | 3 | 0 | 1654 |

The [native F/E panel](/mnt/data/dec5_mhr_sampling_candidates/admission/rgb_review/F004_E_native.png)
visibly shows the black puncture shrinking, with **ten fixed rays still missing**.
All twenty filled rays have nonzero train-textured RGB. The candidate therefore
makes an actual rendered partial repair, not merely a mask-score improvement.
The [moving panel](/mnt/data/dec5_mhr_sampling_candidates/admission/rgb_review/old_moving_native.png)
and two full overviews show no obvious broad new face/color defect, but inherited
ragged hair/neck boundaries remain. Three separately viewed native crops locate
new single-pixel black changes at the clothing edge, neck margin and hair.
Ten new mesh-hit pixels in that view have no RGB. These are retained limitations,
not silently discarded pixels. Stable-geometry source labels change at 155 F/E
and 116 moving-view pixels; the repair is not texture-neutral.

Parent actually viewed four clay comparisons, two native RGB panels, two actor
overviews and all three new-black crops. The [final review](/mnt/data/dec5_mhr_sampling_final_review/visual_review.json)
records **partial_not_promoted**, not a full repair or all-frame-video pass.
The final check revalidates retained input/output hashes, measured audit, matched
render settings, twenty colored residual hits, no lost geometry, new-black counts,
and the unchanged delivered 6K video hash.

PSNR/SSIM/LPIPS are N/A: these are matched train-textured diagnostics, not a
held-out image-quality comparison. No full-frame image metrics are substituted.

## Insights

The decisive comparison is the actual admitted surface and rendered hole, not
only a smaller vertex SDF statistic. Raw prior intersections with target rays
are insufficient: semantic and measured-depth admission still apply unchanged.

The surface term improves both fitting and reserved-camera agreement beyond
the tested weight-only trajectory. It reduces but does not eliminate the local
silhouette failure. Four samples per face are not guaranteed to resolve a narrow
nonconvex boundary excursion between sample positions. Admission and a native
RGB comparison remain necessary before treating this as a practical repair.

Those gates now demonstrate a partial rendered improvement, while the remaining
hole and rendering side effects prevent full acceptance. A direct read-only
quadrature check on the four residual parents finds a maximum sampled B/B SDF
of only .11322px versus 1.04962px at the unsampled diagnostic ray points; A/C's
sampled values are all negative (largest −.15678px). Thus the four-point rule
still under-resolves this boundary. A subsequent geometry-selected denser or
adaptive surface quadrature is justified, without using the diagnostic target
rays as fitting locations or relaxing masks/depth guards.
