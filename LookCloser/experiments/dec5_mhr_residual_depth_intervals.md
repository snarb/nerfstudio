# All 30 F/E residual rays have a nearby silhouette-feasible depth interval

## What was tested

A read-only feasibility scan for the existing 30-pixel residual in frame 001193, F/E train camera. Each ray is the exact calibrated native production ray, centered at its hit on the frozen full MHR prior. The queried range is ±.003 in its non-unit `t_hit` parameter; positive offsets go farther from the camera. Original stored depth, triangle ID and barycentric point replay exactly. No target scan enters fitting, active-region selection, geometry, admission or production.

The scan uses all 62 existing train foreground masks with the unchanged measured D/D override. Nearest binary masks (`np.rint`) and bilinear signed distance (outside EDT minus inside EDT, threshold zero) use the same frozen guard projection and footprint domain. No dilation or 2-pixel fitting allowance is added. All 62 cameras are available at every sampled point—feasibility is not obtained by leaving a camera's field of view. True held-out images are not used.

Initial step .00001 gives 601 positions per ray. A separate .000001 verification grid gives 6,001; each observed transition is bisected to a bracket ≤.00000001. This is a finite dense scan with refined boundaries, not a proof excluding arbitrarily thin unsampled intervals. Positive feasibility is directly witnessed in both grids. Ray direction norms are 1.000195–1.000224, so Euclidean movement is the reported offset times that factor.

## Results

**The local intersection is nonempty for all 30 rays, under both classifiers.** At the original prior hit, all 30 fail. Each residual ray has one observed feasible interval extending from the entry below through the +.003 scan endpoint; that endpoint is truncated, not an identified exit.

| Classifier | Feasible residual rays | Entry offset min / median / max | Feasible at prior hit |
|---|---:|---|---:|
| Nearest binary | 30/30 | +.00013074 / +.00022237 / +.00032600 | 0/30 |
| Bilinear signed distance ≤0 | 30/30 | +.00014213 / +.00022006 / +.00032600 | 0/30 |

No camera combination makes the entire ±.003 intersection empty. The cameras rejecting the *original prior hits* are:

| Camera | Binary rejected rays | Bilinear-SDF rejected rays |
|---|---:|---:|
| A/C (`A004_C005_121008`) | 30 | 30 |
| B/B (`B004_B005_1210Z3`) | 27 | 27 |
| A/B (`A004_B005_1210RN`) | 12 | 12 |
| C/B (`C004_B005_1210ER`) | 4 | 3 |

At the refined entry boundary, the last remaining binary veto is A/C for 16 rays and B/B for 14. For SDF it is A/C for 15 and B/B for 15. Audit receipts contain the actual camera lists on both sides of every transition bracket. These point-ray counts differ legitimately from the earlier **29 proposal facets × multiple barycentric samples** attribution; those are not the same denominator.

### Nearby existing-hit controls

Six control pixels were fixed without looking at mask feasibility: `(682,1144)`, `(689,1144)`, `(696,1144)` above the puncture and `(682,1154)`, `(689,1154)`, `(696,1154)` below. They have an existing original-mesh depth hit and nonblack train-textured RGB; this does not make their geometry ground truth.

All six have a feasible interval for both classifiers, but none is feasible at its **prior** hit. The three below already pass every mask at the **original mesh** hit, lying respectively +.00047183, +.00073266 and +.00085539 beyond their prior hits. The three above fail 6, 8 and 10 masks at their original hits, respectively, and lie on the nearer side of their priors. These negative controls are retained: the mask/prior disagreement is not unique to missing pixels, and apparently filled geometry is not automatically multiview-consistent. No claim that those masks or original surfaces must be wrong is inferred from this test.

### Native evidence and verification

Actually inspected [native underchin context](/mnt/data/dec5_mhr_residual_depth_intervals_v2/native_underchin_context.png), [6× nearest-neighbor pixel localization](/mnt/data/dec5_mhr_residual_depth_intervals_v2/native_ray_locations_6x.png), and [ray/offset feasibility chart](/mnt/data/dec5_mhr_residual_depth_intervals_v2/verification_intervals.png). Magenta denotes the 30 residual pixels, cyan the six controls; the chart's green portions are feasible. The images locate the small puncture separately from the larger background opening and continuous shadow. Background gaps are not labeled missing anatomy.

[Per-pixel intervals and camera vetoes](/mnt/data/dec5_mhr_residual_depth_intervals_v2/result.json), [refined boundary witnesses](/mnt/data/dec5_mhr_residual_depth_intervals_v2/boundary_audit.json), [exact rays, prior triangle IDs and 6,001-position scan](/mnt/data/dec5_mhr_residual_depth_intervals_v2/verification.npz), [camera replay provenance](/mnt/data/dec5_mhr_residual_depth_intervals_v2/cameras_replay.json), and [final seal](/mnt/data/dec5_mhr_residual_depth_intervals_v2/final_seal.json) preserve reproducible evidence. The audit replays 14,735,664 camera/sample classifications across both grids; an independent SciPy bilinear interpolator gives zero value/sign differences. Four tests cover ray indexing, exact stored ray/depth construction, interval segmentation and transition refinement.

Two orchestration failures are retained outside the successful root. The first preflight compared ray-coordinate points to double-vertex barycentric points too strictly; their maximum discrepancy is 1.0632e-7 scene units. The successful run verifies the old barycentric points exactly and separately uses the actual calibrated ray. A subsequent completed scan failed JSON serialization of NumPy booleans; its arrays, images, request and exact producer snapshot remain in `/mnt/data/dec5_mhr_residual_depth_intervals`. The corrected serializer reran in the separate `_v2` root. Neither correction changes masks, fitting or geometric gates.

## Insights

The original prior surface is slightly too near along these rays to satisfy the existing silhouettes; a modest deeper point can satisfy them. This rules out “no local mask-feasible depth exists” for this residual. It does **not** prove that a continuous, topology-safe, measured-depth-compatible surface can occupy those intervals. No pointwise scan result is prescribed as a fitting target, and no patch is admitted here.

Any independent correction still needs its existing anchor, topology, measured free-space and native-view gates. This report uses the reporting workflow to separate observed feasibility from anatomical interpretation. PSNR/SSIM/LPIPS are N/A because this is a mask-feasibility diagnostic, not held-out image-fidelity evaluation. Production, delivered 6K, original geometry and masks remain unchanged.

Reproduce with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1 /home/brans/repos/nerfstudio/.venv/bin/python scripts/probe_mhr_residual_depth_intervals.py --output NEW_ROOT`. Audit with `audit_mhr_residual_depth_intervals.py --output NEW_ROOT`; after actual visual review, pass `--report experiments/dec5_mhr_residual_depth_intervals.md --tests tests/test_mhr_residual_depth_intervals.py` to seal. Tests: `python -m pytest -o addopts='' -q tests/test_mhr_residual_depth_intervals.py`.
