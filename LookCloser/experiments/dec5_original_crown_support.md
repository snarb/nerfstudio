# DEC5: evidence for existing crown/fringe geometry

## What was tested

2026-09-15. The [inset surface prior](dec5_inset_head_completion.md) partly filled
native-view crown gaps but retained the original detached fringe. Diagnose whether
preserving every original triangle also preserves unsupported geometry. This study
does **not** delete triangles, change a mesh or rerender a production video.

At actual frames 001083/001123, raycast the original production mesh in the previously
selected real train hair camera. Query all triangles visible in the frozen manual
train hair polygon. Define a diagnostic crown band as the top quarter of that polygon's
portrait bounding box, using the same rule at both times. These are regions for evidence
inspection, not candidate-dependent metric masks or a future deletion policy.

For each triangle, query its three vertices and centroid against the 62 full-resolution
PatchMatch depth maps. The nearest agreeing physical camera supplies an anchor; other
cameras must meet the existing .001 depth, 1.5-pixel roundtrip and 1-degree parallax rules.
Also compute original and measured-refined foreground-mask support/vetoes.

Descriptive labels: median four-sample depth support below two, at least three mask
vetoes, or both. **Missing depth alone is not an error certificate.** Refined masks retain
the limitations documented in the earlier study; they are not perfect semantic ground truth.

Select three spatially separated high-veto/low-support crown triangles per time for
six-camera native RGB contact sheets. Save all per-camera observations. A stronger
background-conflict indicator requires all four projected points' 9×9 neighborhoods to
contain no refined foreground-mask pixel. Native RGB inspection checks whether those
mask classifications actually correspond to background or uncertain hair boundaries.
Use the exact fixed-profile uint8 display images, no new color response or held-out RGB.

## Results

| Frame / diagnostic region | Queried triangles | Median depth-support votes | Low-depth triangles | ≥3 mask-veto triangles | Both |
|---|---:|---:|---:|---:|---:|
| 001083 hair | 18,328 | 15 | 1,723 | 2,990 | 815 |
| 001083 crown | 5,097 | 9 | 842 | 1,313 | 559 |
| 001123 hair | 17,590 | 15.5 | 1,495 | 2,505 | 722 |
| 001123 crown | 3,921 | 7 | 823 | 1,234 | 571 |

The crown has weaker support than the broader hair region, but is far from uniformly
unsupported. Do not trim all hair fringes or infer that every dark gap is missing anatomy.
These are evidence distributions, **not PSNR/SSIM/LPIPS, face metrics or coverage truth**.

| Frame / selected triangle | Four depth votes | Mask-veto cameras | All-four-point clear-background cameras |
|---|---|---:|---:|
| 001083 / 101418 | 0, 0, 0, 0 | 51 | 25 |
| 001083 / 58658 | 0, 0, 0, 0 | 48 | 30 |
| 001083 / 97093 | 0, 0, 0, 0 | 47 | 40 |
| 001123 / 6675 | 0, 0, 1, 0 | 54 | 30 |
| 001123 / 14352 | 0, 0, 0, 0 | 53 | 36 |
| 001123 / 10611 | **5**, 1, 1, 1 | 48 | 17 |

The main agent inspected both native crown label panels and all six projection sheets
(36 displayed camera crops). Several selected triangle projections clearly fall in room
background beyond the hair in separated cameras; other views overlap hair/curls or mixed
boundary pixels. This supports testing targeted fringe correction, not automatic deletion
of all labelled triangles. The last example has positive multiview support at one vertex;
a median-only deletion rule would ignore that contradictory evidence.

The latest pre-existing head-boundary repair added 1,537/886 triangles. Only 47/66 of the
5,097/3,921 queried crown triangles belong to those additions. All six selected examples
predate that repair. The diagnosed fringe is therefore not explained solely by that last
hole-closing step. Earlier geometry was preserved in position, even where topology splitting
changed vertex indices.

- [083 crown labels](/mnt/data/dec5_original_crown_support/001083/native_crown.png)
- [123 crown labels](/mnt/data/dec5_original_crown_support/001123/native_crown.png)
- [083 six-camera example](/mnt/data/dec5_original_crown_support/001083/projection_00.png)
- [123 six-camera example](/mnt/data/dec5_original_crown_support/001123/projection_00.png)
- [Example with a supported vertex](/mnt/data/dec5_original_crown_support/001123/projection_02.png)
- [Visual findings](/mnt/data/dec5_original_crown_support/visual_review.json)
- [Independent depth/mask replay](/mnt/data/dec5_original_crown_support/audit.json)

Three focused tests cover descriptive labels, zero-displacement topology splits and the
already-quantized display-image contract. Independent replay recomputes all four-point
depth votes and refined-mask votes, verifies source RGB/mesh hashes, and seals accepted
evidence. No geometry changed and no production default was edited.

Two diagnostic implementation errors were caught and corrected before accepting results:
an initial provenance assertion compared vertex IDs instead of triangle coordinates after
zero-displacement splitting; the next attempt incorrectly rescaled already-uint8 RGB in
review panels. Both attempts are preserved under explicitly failed/invalid subdirectories,
excluded from accepted evidence. The corrected runs reproduce the numeric depth/mask counts;
only their corrected RGB panels are used for the visual verdict.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/audit_original_crown_support.py --check
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python -m pytest -q -o addopts='' tests/test_original_crown_support.py
```

## Insights

1. Addition-only repair is insufficient when a visible rim contains unsupported geometry.
   Some existing triangles project outside the real hair in many views, not merely outside
   a single imperfect mask. This is a concrete reason to test replacement/removal of weak
   original fragments, rather than only adding another shell behind them.
2. A future edit must preserve positive depth evidence and distinguish mixed hair boundary
   pixels from confident background. A plausible conservative test is unanimous weak support
   at all four samples plus several clear-background camera conflicts, applied uniformly to
   the head—not just the manually inspected triangle IDs. It remains **untested** here.
3. Deletion can enlarge holes. Evaluate any such edit with and without the constrained
   completion surface, across native and moving RGB and multiple times, before promotion.
   This diagnosis does not fulfill the artifact-free dynamic-video objective.
