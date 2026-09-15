# Independent exact-ball solver review

## What was tested

Read-only review of `conic_surface_step.py`, `fit_mhr_conic_correction.py`, and the separate `certified_conic_surface_step.py`/execution adapter. No DEC5 fit, actor images, target rays, geometry changes or production changes were performed. Clarabel 0.11.1 remains in its isolated environment. This is a numerical solver review; image metrics are not applicable.

Three seeded, fully coupled 2-/3-vertex problems were solved at system scales 0.01, 1 and 100, with offset radius-0.001 balls and linear lower bounds. One case requires nonzero feasible motion through an inter-vertex constraint. Independent SLSQP uses analytic derivatives, up to 2,000 iterations and tolerance 1e-13. Its solution is a numerical reference, not a symbolic optimality proof.

Reproduction: run `scripts/review_mhr_conic_solver.py --output NEW_ROOT`, then `scripts/review_certified_conic_adapter.py --input NEW_ROOT --output NEW_ROOT/certified_adapter`. Evidence is retained at [the review root](/mnt/data/dec5_mhr_conic_independent_review/result.json); snapshots preserve the exact reviewed producers. `audit_mhr_conic_independent_review.py` verifies this recorded root and writes its final seal.

## Results

| Check | Result |
|---|---:|
| Coupled optimization comparisons | 9/9 pass |
| Maximum coordinate difference from SLSQP | 3.134e-9 scene units |
| Maximum scaled objective difference | 1.563e-10 |
| Maximum restored stationarity / (Hessian scale × 0.001) | 1.116e-11 |
| Maximum SOC complementary inner product | 1.107e-10 |
| Maximum physical displacement | 0.0009999999999912 |
| Minimum original linear slack | 6.567e-15 |
| Original-versus-certified synthetic solutions | 9/9 byte-identical |
| Existing original/adapter unit tests | 11 passed |

The cone signs and scaling are consistent. With `x = u y`, `u = 0.001`, and `s = max diag(MᵀM)`, the backend uses `P = MᵀM/s`, `q = −Mᵀr/(su)`. Linear cone slack is `Ay − b/u`; each Lorentz slack is `[R/u, offset/u + y]`, exactly encoding the displacement ball. Restored linear multipliers are positive `z_linear su`; the SOC vector contribution is `−z_vector su`. Direct assembly assertions and original-coordinate stationarity checks cover both signs. Objective scaling changes no minimizer.

Complementarity is actually enforced per linear coordinate and per Lorentz cone, not merely logged. A feasible/stationary but noncomplementary synthetic SOC certificate is rejected. **One original defensive defect was confirmed:** a malformed NaN objective vector can produce NaN stationarity that passes the original `>` comparison. This is not evidence that the real solver emitted such a result. The separate certified adapter checks finite inputs and finite stationarity and rejects the same probe.

The adapter changes only accepted backend status (`Solved` or `AlmostSolved`) and finite certificate validation. All original numerical tolerances remain unchanged. Controlled status substitution verifies: a genuine solution accepts under either status; a feasible but deliberately nonstationary `AlmostSolved` solution rejects; `MaxIterations` rejects even with otherwise valid solution data. See [adapter evidence](/mnt/data/dec5_mhr_conic_independent_review/certified_adapter/result.json).

The original actor run stopped at outer iteration 3 on `AlmostSolved`. The parent's separate [matrix replay receipt](/mnt/data/dec5_mhr_conic_solver_diagnosis/result.json) records primal 2.046e-11, stationarity 2.376e-12, complementarity 5.450e-11 and displacement 0.001000000000020455, within the unchanged gates. This review read and hash-binds that receipt; it does **not** claim an independent replay of those actor matrices.

## Insights

Exact SOC balls remove the finite-support-plane approximation of displacement bounds. Backtracking between feasible current/proposed points also preserves these convex balls. The read-only driver review found no target-ray/target-ROI input introduced by the conic substitution; existing fitting cameras, anchors and downstream guards remain inherited.

This does not certify nonlinear silhouette convergence, anatomical accuracy, continuous collision-free motion, or absence of every self-intersection. The inherited global guard checks new transverse crossings, not all possible coplanar overlaps; the previous independent review distinguishes that theoretical limitation from unchanged actual Open3D pairs. Solver certificate acceptance is not surface or patch promotion. No new fitting variant was run by this review.
