# Earlier-time transfer of narrow-crack completion

## What was tested

Transfer the [minimum-gap-free Poisson completion](dec5_close_boundary_completion.md)
that closed a tiny late-frame jaw crack to measured times `001083` and `001123`.
Use each time's original published mesh and 62 native PatchMatch depth maps.
Keep all maximum-distance, boundary, normal, semantic, measured-support and
observed-neighborhood certificate gates unchanged. Original triangles/vertices
are preserved exactly; only inferred local triangles can be added.

These earlier frames use **original source masks**, without the late-frame D/D
measured mask override. The adapter makes that override optional, with a test
proving the original override body is otherwise unchanged. This is an explicit
input difference, not a claim of identical annotations across times. No target
camera/RGB selects geometry; moving and native F/E poses are review controls.

Six new RGB renders use the current production texture recipe: incidence 2,
frozen exposure/profiles, zero registration, hard single-camera RGB. Two unchanged
moving baselines are reused only with verified receipts. No defaults change.

## Results

**No meaningful visible improvement in these controls; not promoted.**
Passing evidence gates and adding triangles do not establish hole repair.

| Time | Raw local proposals | Semantic admitted | Direct-support faces | Final added faces |
|---|---:|---:|---:|---:|
| 001083 | 40,249 | 1,465 | 378 | 540 |
| 001123 | 54,887 | 2,283 | 417 | 472 |

Observed-neighborhood interpolation admits additional faces beyond direct sample
support. `001083` retains all 540 initially accepted faces; `001123` removes
2 then 1 via native free-space pruning, retaining 472. These are inferred
surfaces, not newly measured anatomy or certified watertight reconstructions.

Independent replays verify 2,438 / 4,015 seed queries, 1,318 / 1,897 local
certificates, and **124 native ray checks per time**. Final qualified free-space
violations are zero. A second audit recomputes semantic votes and verifies the
62+62 depth-file hashes and original geometry prefixes. Because additions are
not welded, connected-component counts increase from 95→248 and 78→204;
nonmanifold edge counts (allowing boundaries) are zero. This is another reason
not to equate extra triangles with a better portable surface.

| Time / view | New depth pixels | Lost depth pixels | RGB pixels changed | Black pixels removed |
|---|---:|---:|---:|---:|
| 001083 moving | 0 | 0 | 0 | 0 |
| 001083 real F/E | 0 | 0 | 0 | 0 |
| 001123 moving | 0 | 0 | 5 | 0 |
| 001123 real F/E | 0 | 0 | 0 | 0 |

These are integrity/support differences across each image, **not full-frame
quality metrics or anatomical hole counts**. All four controls introduce zero
new black pixels. Three RGB pairs are pixel-identical; the remaining five
changes occur at existing mesh hits. Two near the jaw silhouette become
considerably brighter (roughly RGB 114/79/55→193/143/100 and
111/78/56→186/139/91), not newly recovered skin pixels.

Main-agent review covered all four native head comparisons and four separate
jaw crops, with real train RGB in both F/E comparisons. The existing crown
notches/fringe and tiny under-chin boundary defects remain. The real photographs
also contain cast shadows and genuine background next to the neck; their dark
appearance is not sufficient evidence of a missing skin surface.

- [001083 moving head](/mnt/data/dec5_close_boundary_transfer/review/001083/moving_head.png)
- [001123 moving head](/mnt/data/dec5_close_boundary_transfer/review/001123/moving_head.png)
- [001123 real train jaw comparison](/mnt/data/dec5_close_boundary_transfer/review/001123/F004_E005_1210FP_jaw.png)
- [Manual diagnostic review](/mnt/data/dec5_close_boundary_transfer/visual_review.json)

All two geometry/audit workers and six sequential RGB jobs completed normally.
Eight focused tests pass; the final read-only audit rechecks 278 hashes.
No new PSNR/SSIM/LPIPS were computed or claimed:
these diagnostic controls have not established a held-out improvement. Source
data, published meshes/video and the parallel trajectory variants are unchanged.

Reproduction:

```bash
python scripts/transfer_close_boundary_completion.py --frame 001083
python scripts/transfer_close_boundary_completion.py --frame 001123
python scripts/render_close_boundary_transfer.py render
python scripts/render_close_boundary_transfer.py review
# Author manual review only after inspecting the saved comparisons.
python scripts/audit_close_boundary_transfer.py
python scripts/audit_close_boundary_transfer.py --check
```

Geometry preparation refuses existing attempts rather than silently mixing a
new nondeterministic Poisson solve with old evidence. RGB requests are resumable
only if unchanged. Use a separate output root for any changed configuration.

## Insights

The late-frame minimum-centroid-gap fix was a specific narrow-crack repair, not
a general remedy for head-boundary defects. Its transfer to these earlier times
passes safety checks but does not close visible holes. Do not deploy hidden
extra geometry or claim progress from triangle count alone.

Larger outstanding gaps require distinguishing absent proposals from rejected
valid surface, or texture/segmentation problems, at the actual defect. Increasing
Poisson reach or relaxing masks without that evidence is not justified by this
test. The full moving-actor/moving-camera artifact-free goal remains open;
trajectory avoidance is being evaluated separately and is not mesh recovery.
