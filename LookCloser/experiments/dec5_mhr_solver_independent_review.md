# Independent review of constrained MHR correction helpers

## What was tested

Read-only review of `constrained_surface_step.py`, `mesh_contact_constraints.py`, the guarded/constrained/contact/bounded-contact correction wrappers, and subsequently `osqp_surface_step.py`. No helper, running input, original mesh or fitted result was edited. Synthetic problems are not DEC5 fits. Evidence: [/mnt/data/dec5_mhr_solver_independent_review/result.json](/mnt/data/dec5_mhr_solver_independent_review/result.json).

## Results

### Math checks passed

- **Lower-bound KKT sign is correct.** For `Ax≥b`, stationarity is `Hx−g−Aᵀλ=0`, with `λ≥0`. The active-set direction `p₀+H⁻¹Aᵀλ` and removal of negative multipliers have the correct signs. Across 100 seeded small positive-definite QPs, no solve failed; the maximum objective difference versus SLSQP was 9.38e-11 and maximum KKT stationarity residual 1.18e-13. This does not certify numerical robustness of the large DEC5 system.
- **Signed-area gradient is correct.** With unit reference normal `n`, derivatives for triangle `(a,b,c)` are `g_b=(c−a)×n`, `g_c=n×(b−a)`, and `g_a=−g_b−g_c`. Row normalization and the lower-bound sign are consistent. One hundred non-axis-aligned checks with a permuted active-vertex subset gave maximum centered-difference error 4.73e-10.
- **Ball-plane sign is correct.** The row enforces `n·(current+step−base)≤limit`. Finite supporting planes form an outer approximation, not the ball itself: a synthetic endpoint with norm .001281 satisfies a tangent plane to the .001 ball. The retained exact displacement guard is therefore necessary; nine inner rounds are not a feasibility guarantee.

### A general contact limitation was reproduced, but not observed in these saved outputs

The SAT rows preserve weak ordering along the selected fixed axis. For coplanar triangles, the enumerated axes can all be perpendicular to the common plane, with zero gap. Then the rows permit an in-plane overlap. A synthetic pair moved .00035 into overlap while retaining unit area ratio and normal cosine; Open3D reported an intersection, but the transverse-only predicate and `StepGuard` accepted it. This is an endpoint coplanar-overlap limitation, distinct from the already documented absence of continuous collision detection. It does not invalidate the narrower “no new transverse pairs” claim.

The requested actual-output check found **no manifestation of that limitation**:

| Saved geometry | All Open3D intersection pairs | Transverse pairs | New all-pairs versus warm | New nontransverse pairs |
|---|---:|---:|---:|---:|
| Warm margin-two prior | 182 | 182 | 0 | 0 |
| `contact_correction_v2` final | 182 | 182 | 0 | 0 |
| Bounded-contact last accepted iterate `009` | 182 | 182 | 0 | 0 |

The pair **sets**, not only counts, are identical. All inherited 182 intersections remain; this is not a globally intersection-free mesh. Exact arrays and input hashes are retained in the evidence root. The bounded worker later raised its active-set iteration-cap exception; no approximate QP result was returned. These checks do not identify the numerical cause of that cap, because the failing QP matrices were not part of this review.

### OSQP scaling is correct; its independent KKT checks are incomplete

For `x=u·y`, `u=.001`, dividing the objective by `scale·u²` gives `P=H/scale`, `q=−g/(scale·u)`, and bounds `Ay≥b/u`. Restoring lower-bound multipliers as `−dual·scale·u` is correct. Four system/RHS scales (.001, 1, 1000, 1e6) matched the reference solution within 6.51e-19; restored stationarity divided by Hessian scale was ≤3.35e-19. [Supplementary code and results](/mnt/data/dec5_mhr_solver_independent_review/osqp_scaling_check.json) preserve the calculation.

**Validation gap:** `scaled_complementarity_inf` is recorded but not enforced. Feasibility, stationarity and dual sign alone are not a complete independent KKT certificate. An isolated mocked backend returned `x=2` for `min .5(x−1)², x≥0`; current independent checks passed despite complementarity 2,000,000 and true optimum 1. This is deliberately a mock, **not an observed OSQP failure**. OSQP's own solved status remains required. Any claim of a complete independent KKT check needs a complementarity acceptance condition, with its scaling stated explicitly. No running helper was changed.

### Input and reporting invariants

No reviewed correction path reads the target residual pixels or the new depth-interval scan. Fit anchors and silhouettes receive the 54-camera fitting subset; the eight reserved cameras are only scored. The original COLMAP and measured D/D override retain their already disclosed all-62 provenance, so the reserved cohort is not independent of every upstream artifact. Inactive vertices are checked exactly against the warm base. The warm prior—not original smooth100—is explicitly the new displacement reference.

Descriptive silhouette means use the currently available samples, not a frozen common cohort: contact train availability changes 82,864→82,889 and reserved availability 12,409→12,411. Solver normalization is still fixed, but those reported means should not be called fixed-denominator comparisons. This is a reporting-scope limitation, not target leakage.

Ten existing active-set/contact/guard tests and two existing OSQP tests passed. No new fit, parameter sweep, target-conditioned geometry or production change was performed.

## Insights

No sign or scaling error was found in the requested algebra. The main concerns are the general coplanar-contact blind spot, incomplete independent complementarity enforcement, and changing descriptive-statistic cohorts. The actual contact final and bounded iterate009 preserve the warm intersection inventory exactly. None of these numerical checks establishes anatomical quality or repair success. PSNR/SSIM/LPIPS are N/A for this solver review.
