# DEC5 001193: guarded warm-start MHR correction

## What was tested

This is an isolated **prior-fitting** experiment for the remaining under-chin
hole, not a replacement of the COLMAP mesh or an accepted video revision.
The selected margin-two MHR fit is the new displacement reference. Training
uses the same 54-camera anchor/silhouette subset; eight reserved train cameras
are scored afterward. The upstream COLMAP/depth override still has all-62
provenance, so this split is not independent of every upstream artifact.
No target RGB, residual pixel coordinates or depth-interval scan enters fitting.

The warm correction uses zero positive-SDF allowance, 40 outer iterations,
maximum step .00025 and total per-vertex displacement .001 in **normalized
scene units**, magnitude sigma .001. Existing data and Laplacian terms remain.
Every accepted iterate must preserve inactive vertices exactly, relative area
≥.25, warm-normal cosine ≥.5, no newly reversed previously positive original
normal, and no new transverse intersection pair. All warm intersections are
inherited, not considered a valid full-face replacement. These are discrete
iterate checks, not continuous collision detection or an anatomical guarantee.

Success would require an admitted, textured repair of the actual hole without
new artifacts. A better silhouette objective or a first-hit mask test alone
does **not** establish that.

### Bounded proposal controls

1. `fit_mhr_guarded_correction.py`: LSMR proposal plus whole-step backtracking.
2. `fit_mhr_constrained_correction.py`: signed-area lower bounds inside the QP;
   nonlinear signed area must also remain ≥1% of its warm positive value.
3. `fit_mhr_contact_correction.py`: add separating planes for newly detected
   triangle contacts and resolve; exact geometry checks still decide acceptance.
4. `fit_mhr_bounded_contact_correction.py`: additionally approximate displacement
   balls by accumulated supporting planes. The exact ball remains guarded.
5. `run_mhr_osqp_correction.py`: same ball-plane problem with private OSQP 1.0.4.
6. `fit_mhr_conic_correction.py`: replace the ball-plane approximation with one
   exact Lorentz cone per active vertex, private Clarabel 0.11.1. The radius,
   least-squares objective, contact constraints and discrete guards do not change.
7. `run_mhr_certified_conic_correction.py`: allow `AlmostSolved` only if the same
   independent primal, dual, stationarity and complementarity checks pass.
   Reject other non-solved statuses. This changes a status acceptance rule, not
   any numerical tolerance. Nonfinite certificate inputs are additionally rejected.
8. `run_mhr_anatomical_correction.py`: restrict the correction to the existing
   model-space anchor width `abs(x)<12 cm`, in addition to the correction height
   band; reject **all** newly reported nonadjacent intersection/contact pairs.
   This deliberately changes the active domain and its count-based normalization.
   It is not a claim of an identical objective or a per-frame target exception.

Dependencies are isolated in `/home/brans/lookcloser_temp/{osqp,clarabel}_correction_20260915`;
main environment/model/runner defaults are untouched. Requests pin private
module/binary hashes. Cone equations follow the
[official Clarabel interface](https://clarabel.org/stable/python/getting_started_py/):
`x=.001*y`, scaled Hessian `H/max(diag(H))`, linear term
`-g/(.001*max(diag(H)))`, and affine slack
`[radius/.001, (current-base)/.001+y]` in each Lorentz cone.

## Results

### Earlier terminal controls

All roots below are under `/mnt/data/`. A stopped fit is never called converged.

| Root suffix (`dec5_mhr_…`) | Accepted iterates | Outcome | Residual first-hit points passing binary masks /30 |
|---|---:|---|---:|
| `guarded_correction` | 5 | Backtracking stalls on new original-normal reversal | 0 |
| `constrained_correction` | 5 | Backtracking stalls on new intersection | 0 |
| `contact_correction_v2` | 12 | Backtracking stalls at displacement bound | 1 |
| `bounded_contact_correction` | 9 | Active-set iteration cap; no final fit/result | N/A |
| `osqp_correction` | 7 | OSQP iteration cap; no final fit/result | N/A |
| `conic_correction` | 3 | `AlmostSolved` status rejected; failed subproblem retained | N/A |
| `certified_conic_correction` | 40 | Iteration cap, not convergence; rejected by wider topology check | 8 |
| `anatomical_correction` | 40 | Iteration cap, not convergence; all-pair guard retained | 9 |

The contact first attempt failed only while serializing NumPy pair indices;
`dec5_mhr_contact_correction` and its log are retained. Its explicit integer-list
fix ran in a new `_v2` root, with no numerical change.

The three completed guarded controls independently replay every saved geometry
check (`guard_audit.json`). The independent review also found exactly the same
182 Open3D intersection pairs in warm/contact-final/bounded-009; all are inherited
transverse pairs. A synthetic coplanar overlap can pass the transverse-only
guard, so no general intersection-free claim is made. See
[independent solver review](dec5_mhr_solver_independent_review.md).

### Fixed-cohort comparison, before the exact-cone control

These are **fit diagnostics**, not image-quality metrics. Only samples available
in every compared fit are included: 82,864 fitting and 12,409 reserved samples.
Do not compare these means with changing-availability means in individual fit
`result.json` files or with a different review cohort.

| Prior | Fitting samples outside / mean positive SDF (px) | Reserved outside / mean positive SDF (px) |
|---|---:|---:|
| Warm margin two | 1812 / .079642 | 268 / .083968 |
| Guarded | 1768 / .074806 | 257 / .078797 |
| Area QP | 1727 / .071511 | 250 / .075364 |
| Contact QP | 1382 / .049643 | 180 / .052365 |

The original 30 residual rays hit every prior, but only 0/0/0/1 first-hit points
pass the binary multi-camera mask test; **no full parent triangle passes**.
This is not a candidate-subdivision/admission test. It establishes that the
earlier global silhouette gain did not translate into a useful hole repair.

Parent viewed the four actual comparison images at
[/mnt/data/dec5_mhr_guarded_correction_review](/mnt/data/dec5_mhr_guarded_correction_review):
`residual_clay.png`, `C004_E005_1210X7.png`, `G004_B005_1210FG.png`, and
`M004_B005_12109O.png`. Small neck movement is visible, without an obvious new
large fold; the inherited eye/lip/full-prior folds remain. None is a textured
or anatomically approved full-face result.

### Exact-cone numerical diagnosis

The strict-status cone run retained the failing sparse matrices and offsets in
`dec5_mhr_conic_correction/solver_failure`. Read-only replay in
`dec5_mhr_conic_solver_diagnosis` found that its `AlmostSolved` result already
passes the unchanged independent certificate: scaled primal violation
2.05e−11, stationarity 2.38e−12, complementarity 5.45e−11; maximum physical
displacement .001000000000020455. This motivated the explicit certified-status
control rather than weaker geometry or numerical thresholds. The replay itself
does not return geometry to fitting and makes no repair claim.

### Exact-cone actor result: numerical progress, still rejected

The certified-status control completed 40 iterations in 210.72 s. On the same
82,864/12,409 shared camera/vertex cohorts, fitting outside samples fall to 642
and reserved to 56; mean positive SDF is .028857/.032019 px. Eight of the 30
residual first hits pass binary masks, versus zero warm and one contact control;
still no complete parent triangle passes. These are not eight rendered repairs.
All inner solves pass the stated certificate thresholds; the largest scaled
stationarity/complementarity residuals are 3.79e−9/3.46e−10.

Crucially, full Open3D pair replay contradicts a broader interpretation of the
transverse-only guard: beginning at iterate023, **11 new nontransverse pairs**
appear. The final has 187 total pairs, of which 176 are strict, versus warm
182/182. It removes six old strict pairs but adds eleven different pairs.
The prior seal/candidate probe correctly refuses it; no candidate or RGB repair
was produced from this fit. The earlier spoken “no new intersections” update
was too broad: it applied only to the transverse check, and was corrected after
the independent all-pair audit. The limitation previously found synthetically
has now appeared on actual fitted geometry.

All eleven new pairs involve triangle12512 with neutral vertices around
`x=19.34..20.99 cm, y=143.05..144.00 cm`: the **shoulder**, not the jaw.
Eight were already in the separating-plane contact set. The y-only active band
therefore allows unwanted shoulder deformation. The original anchor protocol
already restricts `abs(x)<12 cm`; the later correction omitted that width.
The corrected anatomical domain has 1006 rather than 1566 active vertices and
retains every vertex of all three residual parent triangles8113/8226/8316.
This motivates the next matched control, not relaxing the topology check.

Parent viewed all four exact-cone comparison panels in
[/mnt/data/dec5_mhr_certified_conic_review](/mnt/data/dec5_mhr_certified_conic_review),
plus the fixed neutral-model front render. The local neck shape changes without
an obvious large new face fold; eye/lip/full-prior defects remain. The all-pair
failure overrides this limited visual impression. The all-contact guard has a
synthetic regression test reproducing a coplanar overlap accepted by the older
guard and rejected by the new one. The independent
[cone review](dec5_mhr_conic_independent_review.md) checks solver algebra, not
actor quality, and must not be cited as approval of this rejected fit.

### Anatomical-domain and all-contact correction

This control completed 40 iterations in 349.32 s. The final keeps the warm
182/182 total/strict pair inventory, with no new all-pair contacts, no new
original-normal reversal, minimum relative area .63424 and normal cosine .90266.
Maximum displacement is .000734488 normalized units. Late iterations repeatedly
backtrack to avoid a remaining new contact; this is **not convergence**.
The old contact-plane proposal still detects transverse contacts only, while
the final all-pair guard is wider; that mismatch remains a possible step-size
bottleneck, not permission to skip the guard.

`review_mhr_anatomical_correction.py` explicitly intersects active **vertex IDs**
before comparing camera samples, so removing shoulder vertices cannot by itself
improve the following table. The fixed cohort is 54,193 fitting and 8,048
reserved samples and is different from both earlier tables.

| Prior on shared anatomical vertex IDs | Fitting outside / mean positive SDF (px) | Reserved outside / mean positive SDF (px) |
|---|---:|---:|
| Warm margin two | 754 / .028241 | 60 / .010425 |
| Contact QP | 613 / .016273 | 38 / .005668 |
| Anatomical + all-contact guard | 337 / .005812 | 15 / .002289 |

Nine of the 30 first-hit points pass binary masks, but zero complete coarse
parent triangles pass. This is useful geometric/semantic alignment progress,
not a filled hole. Parent viewed all four comparison panels in
[/mnt/data/dec5_mhr_anatomical_review](/mnt/data/dec5_mhr_anatomical_review).
The neck is modestly changed without an obvious new large fold; the inherited
full-prior face/eye/lip defects remain. No RGB/anatomical full-prior approval.
One prematurely attempted review failed before creating an output because the
fit was still live; its log is retained. The actual fit was not restarted.
Review now explicitly requires a terminal result before reading fit arrays.

The optional `probe_mhr_conic_candidates.py` requires a passed all-pair replay,
explicitly seals only the prior audit (not anatomy), then uses the unchanged
locality/subdivision and multi-camera semantic gates on the actual production
base. It does not pretend the new fit satisfies the older frozen-margin-two
CLI recipe. A semantic-only candidate is explicitly not publishable without
the independent measured-depth and RGB gates.

### Subdivided proposal probe and final bounded audit

The anatomical prior passed the independently replayed all-pair guard for all
40 iterates and was sealed **for proposal testing only**. The unchanged builder
on the actual production base generated 698,658 local proposals; 463,789 pass
the multi-camera semantic test. On the fixed 30-pixel residual:

| Geometry used for the probe | Hits | Remaining misses |
|---|---:|---:|
| Unchanged production base | 0 | 30 |
| Raw local proposals | 30 | 0 |
| Semantic-only surviving proposals | 6 | 24 |

The last row is **not** measured-depth-admitted geometry and has no RGB render.
It cannot be published as a successful repair. The result narrows the remaining
failure to fitted surface/semantic agreement, with additional rejection at the
subtriangle footprint level (9 first-hit points but only 6 surviving ray hits).
Merely showing the raw filled surface would hide the failed consistency checks.

[`audit_mhr_correction_study.py`](/home/brans/repos/nerfstudio/LookCloser/scripts/audit_mhr_correction_study.py)
rehashes 117 declared bindings, verifies the exact original mesh prefix,
proposal arrays and recorded mask/ray counts, and confirms the delivered 6K
video SHA is unchanged. Evidence:
[/mnt/data/dec5_mhr_correction_study/audit.json](/mnt/data/dec5_mhr_correction_study/audit.json).
This is a bounded fit/proposal audit, not a measured-depth admission replay.

Reconstructing the last rejected full step from two saved iterates and its
backtracking factor identifies one new nontransverse contact, pair11510/11596.
It is **already present** in the proposal's contact set. Thus at least this
last bottleneck is not an undiscovered pair: weak separating inequalities
permit touching, while the all-pair endpoint guard rejects it. The saved
reconstruction is explicitly a floating-point diagnostic, not the exact solver
array. A small positive separation constraint is a justified next numerical
test; it has not yet been validated or used here.

27 focused parent tests pass, including the malformed-certificate and coplanar
contact regressions. Independent solver reviews have their own separately
reported tests and should not be added to this count as unique coverage.
All fit/probe/audit workers are terminal. No production mesh, source RGB,
calibration or delivered video was changed; all failed workspaces/logs remain.

## Insights

The initial safety fixes exposed distinct optimization bottlenecks: globally
scaling a step because one facet reverses, then because another contacts, then
because one vertex reaches its displacement cap. Moving those constraints into
the proposal problem permits more progress but is not proof of better anatomy.
Many nearly dependent supporting planes also exposed numerical solver limits;
an exact cone removes that particular approximation, not the need for admission.

The depth-interval study independently found mask-feasible offsets on all 30
residual rays ([report](dec5_mhr_residual_depth_intervals.md)); those offsets are
posthoc diagnostic evidence, **not fitting targets** and not measured depths.
If the guarded prior still cannot repair the hole, the next question is local
surface representation/coverage, not automatically relaxing masks or declaring
success from the silhouette average. PSNR/SSIM/LPIPS are N/A until an actual
RGB candidate is rendered. Delivered dynamic 6K video remains unchanged.
