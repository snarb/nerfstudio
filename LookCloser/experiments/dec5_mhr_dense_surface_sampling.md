# DEC5 001193: denser surface silhouette quadrature

## What was tested

The [four-sample surface control](dec5_mhr_surface_sampling.md) repairs 20/30
fixed under-jaw hole pixels after measured-depth admission and actual texturing.
Ten remain missing. Its B/B quadrature sees only .11322px positive SDF while
unsampled diagnostic surface points reach 1.04962px, motivating a density test.

`run_mhr_dense_sampling_control.py --order 4` and `--order 8` use nested
barycentric lattices: 13 and 43 samples per face, respectively, versus the
previous four. Triangle corners are excluded because the independent vertex16
term already includes them; each lattice includes the centroid and all points
of the previous lattice. Faces touching an active vertex are sampled equally,
not by target coordinates or observed defect location.

The auxiliary surface weight remains16, normalized by the actual sample count
and camera count; vertex weight16, robust scale8px, zero silhouette offset,
warm start, train/reserved split, 40 outer steps, depth anchors, .001 displacement
limit and all geometry/contact guards remain unchanged. These are quadrature
controls, not a relaxation of mask/depth admission or a per-frame exception.

`mhr_dense_surface_silhouette.linearizer` rebinds only the association function
in a private copy of the frozen numerical function; it does not monkeypatch
shared globals. The order2 control reproduces the original matrix and RHS
exactly in the regression test. Both new fits run in separate CPU processes.
Outputs: `/mnt/data/dec5_mhr_sampling_dense4` and
`/mnt/data/dec5_mhr_sampling_dense8`.

`score_mhr_surface_cohort.py` compares terminal fits on one common denser
quadrature and availability cohort. It reports both all samples and samples
depending on at least one active vertex, without hiding the fixed-only residuals.
These are fit diagnostics, not PSNR/SSIM/LPIPS or admitted-hole counts.

## Results

The order4 fit completes 40 iterations in259.48s. The original order8 run stops
after28 saved iterates when its next cone solve fails the unchanged primal
certificate: 1.1569735e-8 violation against a1e-8 limit. It has no accepted final
fit and is retained intact, not silently treated as complete.

The [exact failed-problem replay](/mnt/data/dec5_mhr_dense8_solver_precision/result.json)
changes only numerical solver settings, never the stored matrices, RHS, radius
or acceptance thresholds. Tighter solve tolerances and a smaller termination
step return the same rejected result. Lower internal KKT regularization,
disabled equilibration, and tighter iterative refinement each pass the original
certificate. The selected refinement-only change uses relative1e-15 / absolute
1e-14 and at most30 internal refinement iterations; the replay's primal
violation is2.3144e-10, stationarity1.30e-16, complementarity4.23e-10.

`refined_conic_surface_step.py` preserves the exact numerical solve function and
certificate function, changing only those three internal refinement settings.
`run_mhr_dense_refined_control.py` replays order8 from the original warm start in
the separate `/mnt/data/dec5_mhr_sampling_dense8_refined` root. It completes40
iterations in272.61s. Both terminal fits stop at the outer iteration cap, not a
demonstrated convergence criterion. This numerical-setting difference is part
of the dense8 experiment and is not concealed as an identical solver trajectory.

| Control | Vertex outside fit / reserved | First-hit point mask passes /30 | Parent-triangle checks passed /30 |
|---|---:|---:|---:|
| Four samples, previous | 116 / 8 | 23 | 6 |
| 13 samples | 91 / 8 | 24 | 6 |
| 43 samples, refined solve | 90 / 8 | 28 | 25 |

The parent-triangle column counts diagnostic rays, not distinct triangles.
All comparisons use54,193 fitting and8,048 reserved common vertex/camera samples.
The parent directly viewed four native clay comparisons for each fit; no new
large fold is obvious locally, while inherited whole-prior eye/lip folds remain.
The 13-sample semantic-only candidate hits22 of30 rays, identical to the earlier
four-sample candidate: its extra point pass is not an actual patch-coverage gain.

No production mesh or delivered3456×6144 video is modified. Completed depth
admission and actual RGB **reject the43-sample candidate as a regression**.

On a shared43-point quadrature, all arms have5,000,727 fitting and742,907
reserved camera/sample pairs. All-sample mean positive SDFs for sparse / dense4 /
dense8 are .04587485 / .04582977 / .04587745px (fit) and .00676938 / .00662082 /
.00662435px (reserved). Dense8 therefore does **not** uniformly improve the
whole-domain objective despite better local residual coverage. The companion
movable-only cohort is also retained, along with fixed-only samples; no large
inherited excursion is silently excluded from the all-sample table.

The dense8 semantic-only patch hits24 of30 residual rays, versus22 for both
sparse and dense4. This is an upper-stage result, not a textured repair.
Its original geometry prefix is byte-exact; 700,789 raw proposals yield563,402
semantic passes. All40 dense8 iterates pass the independent guard replay and
rehashed seal. The audit execution handle returned143 after emitting its complete
verified result; its termination cause is unknown. The result was independently
checked and sealed from retained arrays/hashes, not assumed from a clean exit.

Actual admission takes293.28s: strict307,203→307,007 added faces and
interpolated324,690→324,433, six native passes each. Its independent completed
audit replays5,634,020 depth/footprint samples,296,037 vertex certificates and
248 final native-camera/lattice checks. This admission audit passes; the
unknown143 termination above concerns the separate prior-fit audit execution.

| Fixed30 diagnostic rays | Previous four samples | Dense43/refined |
|---|---:|---:|
| Raw proposals | 30 | 30 |
| Semantic-only | 22 | 24 |
| Strict initial / final | 9 / 9 | 6 / 6 |
| Interpolated initial / final | 20 / 20 | 13 / 13 |
| Final colored RGB | 20 | 13 |

`diagnose_mhr_admission_stages.py` rehashes both terminal admission inventories,
casts the exact fixed rays, maps every stage triangle back to its original
proposal ID, and checks counts against the existing probe/RGB receipts.
Root: `/mnt/data/dec5_mhr_admission_stage_diagnosis`.
Seven formerly covered rays are lost; none are gained. The loss occurs at
initial measured/certified admission, **not final native free-space carving**.
Of24 dense semantic first-hit triangles,11 fail initial interpolation versus
two of22 for the sparse candidate. None of these failed first hits has a
footprint veto; every rejected vertex certificate reports`outside_seed_hull`.
This attributes particular first-hit facets, not every possible layer on a ray.

The main agent viewed both F/E comparisons, the dense moving native panel and
all three new-black crops. The hole remains visibly larger in dense than sparse;
the moving view additionally has12 newly exposed but uncolored geometry pixels
and three newly black old-surface pixels. Its 103 new geometry hits and no lost
geometry do not offset the hole regression. The previous20-pixel repair remains
the better local candidate, itself **not production-approved**.

The separate original-surface backoff restores only three old-surface RGB
regressions; it does not improve inferred geometry or this hole.
50 focused tests pass across stage indexing, quadrature, solver guards, internal
refinement and [texture backoff](dec5_inferred_visibility_backoff.md).

## Insights

More samples may represent the nonconvex silhouette more faithfully, but they
do not guarantee a feasible deformation, continuous containment, better depth
admission or textured coverage. Compare actual guarded terminal fits and then
the surviving patch; do not infer a rendered repair from a lower fit objective.

The next hypothesis is evidence selection, not ever-denser silhouette sampling:
the24-nearest seed cap can leave the query outside a one-sided hull although
other verified measured seeds within the unchanged radius surround it. The
post-hoc neighbor diagnostic replays every old certificate on66 sparse and72
dense queried vertices. Using all radius/normal-eligible seeds (71–98 in dense)
recovers23/38 previously rejected vertices with no losses on these cohorts.
That is **not a new mesh result**; whole-candidate admission and RGB validation
are required. See `/mnt/data/dec5_mhr_certificate_neighbors/result.json`.
