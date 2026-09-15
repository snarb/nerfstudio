# DEC5: why the inferred crown shell still has openings

## What was tested

2026-09-15. Attribute missing native-train crown coverage to the exact admission
stages of the existing .001-inset Poisson experiment. This is a geometry diagnostic,
not a face-quality score. The fixed manual train hair polygon selects diagnostic
pixels only; its top quarter defines the crown query. It contains some true
background and gaps between curls, so missing-pixel counts are not anatomical errors.

Raycast the original mesh, full unchecked raw inset prior, and guarded admitted
shell at the same native train pose. Map the raw triangle visible at each selected
missing pixel to local geometry, silhouette, measured-guard or retained membership.
Then replay the individual local conditions and inspect three spatially separated
mask-rejected triangles per frame in four real train cameras each.

Roots: `/mnt/data/dec5_crown_completion_gap_attribution` and
`/mnt/data/dec5_consensus_head_normals`. No production artifacts or source data change.

Follow-up hypothesis: a single nearest ragged facet supplies an unreliable normal.
Replace that normal test, only for otherwise eligible rejected proposals, with
an oriented local consensus: 32 nearest original head facet centroids within .003,
at least eight neighbors, area weights capped at their neighborhood median,
Gaussian distance scale .0015, normal coherence >= .5, weighted agreeing fraction
>= .7, mean-direction dot >= .25. No normal sign flipping. All three proposal
vertices must pass. Retain the existing .006 distance/boundary limits, .0015 edge
limit, head coordinate, >= 2 silhouette supports and zero outside-camera votes.

The same rule runs on both frames, across the whole eligible head, without using
the diagnostic target or polygon to select geometry. Append rescued triangles to
the already guarded shell without moving or deleting any original/guarded vertex
or triangle. These additions are **unchecked inferred geometry**, not measured
depth; the screen must justify a subsequent free-space guard and RGB rerender.

## Results

Remaining crown pixels covered by the full prior but absent from the guarded mesh:

| Frame | Local geometry rejection | Silhouette rejection | Measured guard rejection | Total |
|---|---:|---:|---:|---:|
| 001083 | 2,751 | 1,535 | 0 | 4,286 |
| 001123 | 1,223 | 1,454 | 0 | 2,677 |

This zero refers to the diagnosed visible raw-prior triangles, not every surface
or every effect of the measured-depth guard. The full prior also misses 1,039
originally uncovered crown-query pixels at 001083; at 001123 it covers all 4,038.

Overlapping local rejection reasons, weighted by remaining crown-query pixels:

| Frame | Normal | Surface distance | Boundary distance | Long edge |
|---|---:|---:|---:|---:|
| 001083 | 2,751 | 1,158 | 1,158 | 683 |
| 001123 | 1,160 | 0 | 56 | 63 |

These columns must not be summed. In particular, many normal failures also fail
another geometric condition. Initial inspection accidentally recomputed triangle
normals instead of vertex normals on the raw mesh; the membership assertion caught
the discrepancy. That diagnostic was corrected and replayed. Failed directories
and logs remain under `constraints_failed_normals`; this was not a production bug.

The six native mask-veto sheets show plausible exclusions at real background and
curl-edge gaps. Their selected triangles have outside votes in 6–19 physical
cameras. Some weak vetoes are ambiguous near strands; this is not evidence for
blanket dilation of all masks or unconditional acceptance of the raw prior.

Consensus-normal screen, **before** any new measured-depth safety guard:

| Frame | Otherwise eligible normal failures | Consensus rescued | Pass silhouettes | New native depth pixels | New moving depth pixels |
|---|---:|---:|---:|---:|---:|
| 001083 | 81,979 | 16,616 | 9,985 | 3 | 0 |
| 001123 | 44,774 | 10,967 | 5,941 | 74 | 2 |

All native gains are inside the coarse hair polygon. No depth pixels are lost.
Only 220/240 native pixels and 160/70 moving pixels actually see rescued triangles;
most extra triangles are hidden. The four clay views still show the main crown
openings and fragmented top fringe. **Reject as a material crown repair.** No
RGB rerender, free-space approval, full-video replacement or quality-metric gain
is claimed. An unguarded mesh is deliberately named `unchecked_mesh.ply`.

The main LLM inspected both stage overlays, all six native four-camera veto sheets,
and all four native/moving clay images. Hash-bound notes are in
`dec5_consensus_head_normals/visual_review.json`; this terminal review supersedes
the initial producer `visual_status=pending` fields without altering frozen results.

- [001083 stage overlay](/mnt/data/dec5_crown_completion_gap_attribution/001083/crown_stages.png)
- [001123 stage overlay](/mnt/data/dec5_crown_completion_gap_attribution/001123/crown_stages.png)
- [001123 real-view veto example](/mnt/data/dec5_crown_completion_gap_attribution/001123/constraints/mask_veto_02.png)
- [001123 unchecked native clay](/mnt/data/dec5_consensus_head_normals/001123/native_train.png)
- [Replay audit](/mnt/data/dec5_consensus_head_normals/audit.json)

Three tests pass. Independent replay checks stage membership/counts, source hashes,
the exact original/guarded mesh prefix, all-camera mask votes, 64 brute-force
normal neighborhoods per frame (independent of the producer KD-tree), and eight
fresh raycasts against saved geometry buffers. Both frame workers exited normally.
The sealed artifact manifest includes source EXR hashes, masks, geometry, code,
review and diagnostic outputs. No full-frame PSNR/SSIM/LPIPS or loss is computed.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/audit_crown_gap_study.py --check
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/pytest -q -o addopts='' tests/test_crown_completion_gap.py tests/test_consensus_head_normals.py
```

## Insights

1. The normal gate is a large local rejection category, but improving its local
   estimate alone barely changes visible coverage. Counting admitted triangles
   would have misleadingly suggested success.
2. Silhouette rejection is sometimes physically justified: an unconstrained
   filling surface can cover a frontal hole while protruding into another camera's
   background. A useful surface must move to a compatible 3D location, not merely
   bypass the veto.
3. Next test a different spatially fitted completion surface inside multiview
   silhouettes, with the original geometry as a locality constraint and measured
   free space as a separate safety gate. Do not keep relaxing admission of this
   same shell. The dynamic camera-path videos remain separate presentation work;
   these negative geometry results do not establish the overall goal as achieved.
