# DEC5: measured-support neighborhood instead of nearest24 seed cap

## What was tested

The [dense silhouette control](dec5_mhr_dense_surface_sampling.md) loses
coverage at initial measured/interpolated admission, not final carving.
The fixed query can fall outside the convex hull of its24 nearest measured
seeds despite more surrounding verified seeds inside the unchanged radius.
This motivates changing **evidence selection**, not loosening confidence gates.

`diagnose_mhr_certificate_neighbors.py` first replays the actual nearest24
certificates at semantic first-hit vertices, then uses all eligible seeds in
radius. The post-hoc cohorts contain66 sparse and72 dense vertices. All-radius
selection recovers23/38 rejected vertices respectively, with no losses in those
cohorts. This diagnostic uses target-selected queries and is not a repair.

`run_mhr_radius_seed_control.py` then applies the same rule to **all296,037
candidate-query vertices**, never using target rays to select them. The frozen
dense43/refined prior, proposals, actual measured seed pool and masks are reused.
Every seed remains a deduplicated actual native measured depth point validated
by at least three views, within .0005 of the prior and .001 of the old surface.
Radius .003, normal dot .5, at least8 seeds, hull containment, quadratic
conditioning, leave-one-out P90 and predicted offset limits .0005 are unchanged.
All three proposed vertices still need certification. Direct measured admission,
semantic gates and free-space vetoes remain unchanged. Final native checks cover
all62 train views at both ray lattices. No per-frame or target-hole exception,
new prior fit, color fit, RGB averaging or held-out RGB is introduced.

This is a matched whole-candidate neighborhood control, **not** the union of old
and new passes: if extra measurements disprove an old certificate, it is lost.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENCV_IO_ENABLE_OPENEXR=1 \
../.venv/bin/python scripts/run_mhr_radius_seed_control.py \
  --candidate-root /mnt/data/dec5_mhr_dense8_candidates \
  --output /mnt/data/dec5_mhr_radius_seed_control
```

Use a new root for a new run; existing roots are not overwritten. The first
pre-output attempt incorrectly treated an audit count as a dictionary and exited
before creating the control root. The corrected worker and both logs are retained;
no partial numerical result was reused.

## Results

The global test certifies167,064 versus155,456 old vertices:14,089 newly pass,
but2,481 old passes are lost. Thus the selected diagnostic's no-loss result does
**not** generalize to the full proposal domain. Initial interpolated faces rise
324,690→331,340. Six native passes leave331,062 additions. Original geometry
is exact. Construction and admission complete in108.69s.

Root: `/mnt/data/dec5_mhr_radius_seed_control`.
`render_mhr_radius_seed_control.py` uses the frozen current incidence2,
hard-source, unwarped CPU recipe in two independent processes; compares baseline
and new interpolated geometry at F/E and the old moving stress view.
`audit_mhr_radius_seed_control.py` independently selects seeds using exhaustive
distance evaluation (not the worker KD-tree), replays every certificate,
initial admission and final62×2 native checks. Quadratic/veto mathematics reuse
the established helpers; these are not falsely claimed independent derivations.

Actual final RGB covers **24/30** fixed hole pixels, versus13 for the same
dense prior with nearest24 seeds and20 for the previous sparse-prior best.
Six pixels remain absent. F/E has40 newly visible geometry pixels, none uncolored,
no lost geometry and no newly black old RGB. The moving stress view has105 new
geometry pixels, eight uncolored new hits and three newly black original-surface
pixels. It also changes131 source labels at common near-equal-depth pixels;
global texture equivalence is not claimed.

Main LLM actually viewed both native comparisons, both overview pairs and all
three new-black crops. The hole is visibly smaller; no broad new face distortion
is obvious in those two views. Existing jagged neck/hair margins, a small remaining
puncture and an inherited protruding chin fringe prevent full visual acceptance.
These two views are not a full temporal or6K evaluation.

The [verified original-surface backoff](dec5_inferred_visibility_backoff.md) was
also rerun without code changes at `/mnt/data/dec5_mhr_radius_seed_texture_backoff`.
Its reviewer was invoked with only its output-root constant rebound. It restores
exactly the same three old-surface pixels; no other RGB/source pixel changes.
Main viewed all three baseline/raw/backoff panels. The eight uncolored new
surface hits and six missing hole pixels remain. This is a separate display
control, not a geometry repair or promotion of the new mesh.

The auditor's first execution was deliberately interrupted after identifying
repeated whole-array NPZ decompression inside its per-vertex loop. Materializing
the same arrays once fixes runtime without changing arithmetic; both execution
logs are retained. The corrected audit completes in70.36s: all296,037
certificates and124 final native-camera/lattice checks pass, with exact original
geometry prefixes. Its terminal process exits0. Render workers also exit0.
Verified arithmetic does not upgrade the partial visual verdict to full success.

52 focused tests pass, including a planar synthetic case where nearest24 lie
on one side but same-radius measured seeds surround the query; a larger seed
set still fails when the predicted depth offset exceeds the unchanged tolerance.
No production mesh or delivered3456×6144 video is changed. These are diagnostic
1080×1920 RGB renders, not replacements for the user's6K output. PSNR/SSIM/LPIPS
are N/A here; no full-frame quality metric or held-out comparison is claimed.

## Insights

A nearest-neighbor count can be a computational shortcut with geometric side
effects: the closest measurements may lie on only one side of the query.
Increasing confidence by gathering surrounding measurements is different from
extrapolating outside their hull or counting cameras merely seeing a point.
Extra measurements can also reveal inconsistency, as the2,481 lost certificates
show. A global rerun and actual RGB review are essential; the post-hoc local
success cannot justify a production change on its own.

The next checks must cover other frames and more challenging camera positions,
and separate the remaining semantic/proposal coverage limit from seed selection.
Simply increasing silhouette sampling or relaxing admission thresholds is not
supported by this experiment. The much more conspicuous lipstick fin is a
separate geometry/layer-ownership problem, not fixed by the local neck result.
