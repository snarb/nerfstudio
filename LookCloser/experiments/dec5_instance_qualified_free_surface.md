# DEC5: instance-qualified near-depth evidence

## What was tested

Frame `000995`, two frozen topology controls. The hypothesis was that erroneous
near-depth votes beside the lipstick/finger prevent removing the false bridge.
Five reviewed train-only SAM2.1 masks of the lipstick and five of the visible
hand/nails form conservative unions; enclosed mask holes are filled. This is
an explicit semantic prior, not independent measured geometry. No held-out RGB,
new exposure fitting, synthetic views, or production changes are involved.

The coarse control is `/mnt/data/dec5_measured_free_center_pruning/000995`.
The subdivision control is `/mnt/data/dec5_subface_free_space/pruned/000995`.
Each candidate uses exactly the corresponding mesh topology before pruning.
Subdivision moves no surface; it only permits more localized deletion.

`study_instance_qualified_free_surface.py` rejects a native nearest-depth vote
only if both the query projection and actual sampled pixel are at least two
native pixels outside the reviewed hand-or-lipstick union. The point must be
inside all five crops and have at least two negative masks. Other 57 cameras
are unchanged. All original measured far-depth and corroboration thresholds
remain unchanged: all four triangle samples require zero surviving near votes,
six stable far votes and six far observations corroborated by three other views.
`study_subface_instance_qualification.py` applies exactly that rule to the
previously verified refined mesh, with explicit hash-bound input adaptation.

Hand masks and prompts: `/mnt/data/dec5_lipstick_hand_mask_000995`.
Lipstick masks and model provenance: `/mnt/data/dec5_lipstick_instance_mask_000995`.
All five selected hand masks are candidate 2, reviewed on native train crops;
the union includes nails and deliberately protects enclosed uncertain gaps.
These manually prompted masks are a single-time diagnostic, not yet a temporal
segmentation workflow. Earlier lipstick-only attribution is documented in
[instance-mask study](dec5_lipstick_instance_masks.md).

## Results

**Negative as a complete repair; neither candidate is promoted.** The offending
cloth-textured wedge behind the lipstick remains clearly visible in K/B.
The semantic rule removes a small additional rim but does not recover the
correct cylindrical surface or repair existing cheek/chin/hair defects.

| Topology | Depth-only removed | With masks removed | Additional |
|---|---:|---:|---:|
| Coarse | 196 | 200 | 4 |
| Two-round subdivision | 1844 | 1891 | 47 |

Matched RGB differences below isolate **only** the semantic rule, not earlier
pruning or subdivision. These are diagnostic pixel counts, not image-quality
metrics, and do not measure missing anatomy.

| Topology | View | Changed RGB pixels | New black pixels |
|---|---|---:|---:|
| Coarse | Moving camera | 63 | 0 |
| Coarse | H/C | 427 | 44 |
| Coarse | K/B | 103 | 74 |
| Subdivision | Moving camera | 21 | 1 |
| Subdivision | H/C | 60 | 20 |
| Subdivision | K/B | 83 | 34 |

The main agent inspected all 12 head/lipstick comparison panels and all 32
localized new-black components in six native, unscaled sheets. Most changes
trim the existing tube/finger fringe or false bridge; they do not establish
a complete correct contour. The subdivided moving view gains a tiny black
pixel next to the false skin bridge. Existing broad head defects remain.
Both variants fail the requested artifact-free result, despite modest removal.

Artifacts:

- [Coarse comparisons](/mnt/data/dec5_instance_qualified_free_surface/000995/semantic_review)
- [Subdivision comparisons](/mnt/data/dec5_subface_instance_qualified_free_surface/000995/semantic_review)

`review_instance_qualified_surface.py` checks render receipt hashes, identical
cameras, color profiles, exposure and renderer implementation, exact retained
vertices/triangle subset, and deletion-rule replay from saved evidence. It also
checks that deletion adds no ray hits or nearer depth. Original 62-camera near
counts were independently recomputed during geometry execution and exactly
matched their respective pre-existing evidence caches. Far corroboration was
recomputed, not inferred from segmentation. This audit does not independently
validate the physical truth of the masks or every far-depth measurement.

Three subdivision RGB workers finished successfully in parallel, about 26 s
per worker. All six variant renders are complete; no production/video artifact
was replaced. The first coarse geometry attempt exceeded OpenCV remap's
32767-query limit; its log is preserved. The successful implementation chunks
queries at 16384 with an equality test for large batches.

## Insights

1. Close depth is not sufficient object-identity evidence at an occlusion edge.
   The mask-qualified rule can remove a specifically diagnosed false face that
   survives the original near gate. This supports using semantics as an
   evidence qualifier, but the measured visual benefit here is too small.
2. Finer triangles do not resolve the underlying contradictory measurements.
   Forty-seven additional removed faces alter only 21 moving-view RGB pixels.
   Increasing subdivision again is not the next useful experiment.
3. Blindly deleting every point outside the five local object masks is unsafe:
   real neck/clothing behind the hand also lies outside those instance masks.
   A useful next step needs explicit object-layer ownership and occlusion-aware
   support, not a relaxed global mask threshold. Segmentation can distinguish
   a finger observation from a tube observation, but is not itself depth.
4. The delivered native-6K video is unchanged. These failed local controls must
   not be represented as a successful repair of the active video/mesh goal.
