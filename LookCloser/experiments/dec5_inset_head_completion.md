# DEC5: silhouette-constrained inset surface prior

## What was tested

2026-09-15. The [uniform measured-mask experiment](dec5_uniform_measured_head_masks.md)
recovered real hair boundary pixels but hardly repaired the visible crown. Test a
different **surface proposal**, not a further per-frame mask exception: slightly inset
the existing raw Poisson head surface toward the median original head vertex, then
admit it only inside the 62-camera refined foreground silhouettes.

Four identical arms at actual times `001083` and `001123`: radial inset 0, .001,
.003 and .006 normalized units. Original vertices/triangles remain exactly unchanged;
only new proposal vertices move. The fixed proposal screen requires surface distance
and original-boundary distance <= .006, head coordinate x > -.03, maximum triangle
edge .0015 and normal dot >= .25. All four arms use this same expanded proposal rule.
Masks require support >= 2 and zero outside-camera vetoes. The same raw Poisson mesh
is reused, so the comparison does not include Poisson solver variation.

This is a silhouette-constrained geometric prior, **not measured depth** and not the
earlier observed-neighborhood certificate method. The .006 locality and absence of
those certificates are explicit algorithm changes. A mask and absence of a free-space
contradiction do not prove the surface correct. No held-out RGB, target pose, generated
RGB, new exposure adjustment or texture-source averaging constructs the geometry.

Select .001 for both times from the native train-view screen, then prune triangles
that contradict corroborated PatchMatch free space. Use the unchanged native query
guard at offsets 0/.5 in all 62 cameras: depth separation > .003, supported by at least
three other physical cameras. Iterate to convergence, at most eight rounds. Independently
replay shifted vertex geometry, semantic votes, retained triangle assembly and 124 fresh
native ray checks per final mesh.

Six fresh RGB controls use the current fixed-profile/exposure, incidence-2, unwarped,
hard-source renderer; moving production baselines are reused. Exact native calibration
is retained, with the diagnostic target-name alias preventing accidental target masking.
**Texture source masks remain original**, not expanded. The native images are train
diagnostics, not independent held-out validation.

## Results

### Pre-guard geometry screen

| Frame | Inset | Semantically admitted triangles | New depth pixels, native train | New depth pixels, moving |
|---|---:|---:|---:|---:|
| 001083 | 0 | 192,894 | 170 | 7 |
| 001083 | .001 | 224,437 | 779 | 18 |
| 001083 | .003 | 209,589 | 516 | 39 |
| 001083 | .006 | 100,559 | 0 | 25 |
| 001123 | 0 | 214,829 | 225 | 44 |
| 001123 | .001 | 280,369 | 1,387 | 51 |
| 001123 | .003 | 266,695 | 1,370 | 34 |
| 001123 | .006 | 181,089 | 395 | 9 |

These are raw coverage diagnostics, not anatomical missing-surface counts. The unchanged
coarse train hair polygon contains some true background/strand gaps. Increasing the inset
does not monotonically improve coverage: the surface eventually retreats behind old geometry.

### Guarded .001 arm and matched RGB

001083 retains **218,938** added triangles after seven pruning rounds; 001123 retains
**276,107** after four. Both independent audits pass all 124 native checks. Much of this
shell is hidden or overlaps existing geometry; the large face count is **not** a large
visible improvement. Surfaces are appended, not welded, and watertightness is not claimed.

| Frame / view | New depth pixels | Lost depth pixels | Changed RGB pixels | Black removed / introduced |
|---|---:|---:|---:|---:|
| 001083 moving | 16 | 0 | 3,213 | 29 / 6 |
| 001083 native train | 772 | 0 | 1,047 | 763 / 9 |
| 001123 moving | 43 | 0 | 3,670 | 66 / 0 |
| 001123 native train | 1,380 | 0 | 1,792 | 1,375 / 0 |

No full-frame PSNR/SSIM/LPIPS, loss or face-score improvement is claimed. Most new native
surface pixels receive real train RGB; at 001123 only nine of the 1,380 newly covered
pixels remain black. Remaining visible crown holes are therefore not explained solely
by absence of texture on the successfully added part.

Visual review inspected two four-arm native clay panels, four matched crown panels, two
native jaw panels, and the 001123 native clay/full RGB images. The native crown gap partly
fills, especially at 001123, but the upper opening, detached fringe and ragged brown hair
rim remain. Moving crown views show no substantial repair. No conspicuous new face/jaw
defect was observed in the inspected native crops; this is not temporal/held-out acceptance.

Verdict: **partial native-view improvement, not promoted**. No 150-frame reconstruction
or video is replaced. The separately rendered camera-path choices use unchanged production
geometry and cannot be credited with this two-frame experiment.

- [083 native GT comparison](/mnt/data/dec5_inset_head_completion/review/001083/native_unmasked_crown.png)
- [123 native GT comparison](/mnt/data/dec5_inset_head_completion/review/001123/native_unmasked_crown.png)
- [123 moving comparison](/mnt/data/dec5_inset_head_completion/review/001123/moving_crown.png)
- [Geometry audit](/mnt/data/dec5_inset_head_completion/geometry_audit.json)
- [Visual verdict](/mnt/data/dec5_inset_head_completion/visual_review.json)
- [Experimental 123 mesh](/mnt/data/dec5_inset_head_completion/001123/guarded/mesh.ply)

The saved mesh is an untextured geometry diagnostic, not a textured laptop-ready asset
or raw TSDF volume. Source EXRs, calibration, existing geometry and defaults are untouched.
The radial helper test passes (six tests with the existing measured-mask controls).
All 438 sealed hashes were rechecked. Artifact checks also verify actual train EXR and dense-depth
hashes, exact matched camera/source/exposure settings, original texture masks, and finite RGB.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/freeze_inset_head_completion.py --check
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python -m pytest -q -o addopts='' tests/test_inset_head_completion.py
```

## Insights

1. Moving an inferred shell inward can recover part of a mask-rejected hair opening;
   merely expanding masks was less effective. A larger inset is not automatically better.
2. This does not repair the original detached/ragged crown structure. Preserving every
   original triangle also preserves potentially erroneous structures. A useful next
   diagnostic is to evaluate the original crown fringe's multiview support before deciding
   whether to replace or remove any of it; do not silently delete it just to reduce black pixels.
3. The useful native coverage comes with a much larger shell and no observed-neighborhood
   certificate. Do not promote this inference solely because measured free-space checks pass.
   Remaining holes, temporal transfer, texture seams and geometry accuracy still need work.
