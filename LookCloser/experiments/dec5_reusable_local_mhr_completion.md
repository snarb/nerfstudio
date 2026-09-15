# Reusable, opt-in local MHR completion for the calibrated DEC5 rig

## What was tested

Converted the already measured 001193/001195 local-prior method into an explicit frame-configured entrypoint, without changing fitting, locality, semantic, measured-depth or interpolation thresholds. This is an implementation-reuse test, not a third-time result or a new model variant.

[run_local_mhr_completion.py](/home/brans/repos/nerfstudio/LookCloser/scripts/run_local_mhr_completion.py) accepts one JSON specification and a new private output directory. No frame-specific transfer runner is imported. The frozen numerical functions are reused with three exact-count source adaptations: inherited-conformance path, candidate frame label and mask-parent request path. Their original/generated source and hashes are retained in separate receipts, avoiding the overwritten adapter-receipt problem found in the preceding wrapper.

Required specification fields are `frame`, `production_request`, `mesh`, `metadata`, `prior`, `inherited_fit`, `measured_control`, `mask_parent_request`, and `frozen_admission_request`. Unknown fields, including proposed threshold overrides, are rejected. All file paths must be explicit absolute paths. Outputs cannot overlap any input tree. [Executed 001195 specification](/mnt/data/dec5_mhr_reusable_001195_spec.json) provides a complete concrete example.

Initialization verifies the prior's full seal, exact frozen 2px/100-step recipe, same original regularization reference, per-time raw mesh/normalization, production mesh/request hashes, inherited conformance membership, the 54-fitting/8-reserved camera partition, no actual held-out cameras, and frozen guard producers. `check` subsequently loads and verifies all62 actual geometric depth maps/calibrations and independent measured-foreground override, and requires the same depth receipt used by the prior. Geometry is not admitted before this check.

Candidate construction retains the exact measured original mesh prefix and excludes unsafe prior parent facets. All prior locality gates, including the `.00002` centroid-gap rule, remain unchanged. Strict observed-depth and certified-interpolation branches both run, followed by the unchanged 62-camera × two-ray-offset veto until zero removals (maximum eight rounds). Prior geometry is inferred, not measured truth. An audit pass is explicitly not a visual or production acceptance.

## Results

Private root: [/mnt/data/dec5_mhr_reusable_001195](/mnt/data/dec5_mhr_reusable_001195).

The initial implementation has five passing tests: invalid frame/extra threshold rejection, output/input overlap rejection, sealed-input tamper detection, exact path-only adapter and unchanged locality expressions, and rejection of the failed zero-margin recipe. Initialization pins393 inputs. All62 native depth maps and masks passed the input check. The regenerated raw candidate PLY is byte-identical to the sealed production001195 control (669,088 proposals, 88 unsafe parents excluded).

Both admission branches finished in **195.74 s**. Strict retained **215,322** added triangles after native removals `268,7,1,0`; certified interpolation retained **221,945** after `286,7,1,0`. Both completed all124 unique final `(camera, offset)` checks with zero trustworthy free-space contradictions. Strict and interpolation PLYs are already byte-identical to the independently sealed production001195 meshes; the original61,703-vertex/119,830-triangle prefixes are unchanged.

The independent numerical replay passed: **3,422,820 support/footprint samples, 180,069 certificates and 248 native checks**. [audit.json](/mnt/data/dec5_mhr_reusable_001195/audit.json) binds the replay. [equivalence.json](/mnt/data/dec5_mhr_reusable_001195/equivalence.json) separately checks all six array archives and all three PLY files against the sealed earlier production study and binds this report/tests.

Since the meshes are byte-identical, the [previous matched native RGB review](/mnt/data/dec5_mhr_production_patch_001195/admission/rgb_review/old_moving_native.png) remains applicable to those exact meshes; it was not rerendered here. That panel was inspected again during this interface check. The earlier local result remains73→1 missing pixels in both branches, with known two newly black moving-view contour pixels and seven moving/one M/B untextured collar pixels. This entrypoint neither removes those defects nor grants new visual acceptance. Exact earlier reviewed-image hashes are retained as reference provenance, not relabeled as newly generated evidence.

### Reproduction

Use the existing `/home/brans/repos/nerfstudio/.venv/bin/python`, with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1`. Run each stage in its own process; this intentionally configures frozen helper globals before importing their audit. The entrypoint also fixes its CPU thread environment to2.

```text
scripts/run_local_mhr_completion.py init --spec SPEC.json --output NEW_ROOT
scripts/run_local_mhr_completion.py build --output NEW_ROOT
scripts/run_local_mhr_completion.py check --output NEW_ROOT
scripts/run_local_mhr_completion.py admit --output NEW_ROOT
scripts/run_local_mhr_completion.py audit --output NEW_ROOT
```

The config and source hashes are rechecked at each stage. Existing candidate/admission geometry and audit outputs are not overwritten; interrupted workspaces are retained and need a new output root rather than an implicit restart. A minor lifecycle limitation remains: retrying `admit` writes its deterministic `admission_adapter.json` before the frozen producer refuses an existing admission directory. Such a retry does not resume or change meshes; use a new root, not repeated admission. The pinned runner is not silently changed after execution. Final geometry is under `NEW_ROOT/admission/silhouette100/{strict,interpolated}/mesh.ply`. `audit.json` records exact candidate replay, full measured-support/certificate replay, final native checks, mesh-prefix verification and an artifact inventory.

For the separate equivalence check used here:

```text
scripts/audit_local_mhr_reuse.py --output NEW_ROOT --reference /mnt/data/dec5_mhr_production_patch_001195 --report experiments/dec5_reusable_local_mhr_completion.md --tests tests/test_local_mhr_completion.py
-m pytest tests/test_local_mhr_completion.py -q -o addopts=''
```

This comparator requires an independently sealed legacy reference, rehashes it, and compares all candidate/domain/admission/certificate/retained-index arrays and all three raw/strict/interpolated PLYs. Any linked RGB review remains explicitly the old review, not a new render. PSNR/SSIM/LPIPS are N/A for this implementation-equivalence check.

## Insights

The reusable boundary is **fresh per-time sealed MHR prior + matching calibrated depth/mask evidence → separately admitted local geometry**. It is not an automatic prior fitter, arbitrary-rig API, renderer, batch-video integrator or production promotion tool. The fixed DEC5 source/calibration interface and native1920×1080 depth convention remain dependencies. Admission uses all62 train views; the eight prior-fitting-reserved views are not independent admission validation or unseen original-COLMAP evaluation.

Only approved final priors for001193 and001195 are currently present; the other continuation roots are rejected001193 fitting controls. A fresh third-frame check therefore still requires a new per-time prior fit and matching sealed observations. No pose/geometry is copied to an untested time, and no new fitting/GPU job was launched during 6K delivery. The build-report skill keeps this code-equivalence result separate from previously measured local image improvements and unresolved contour defects. Current video, source data and production defaults remain unchanged.
