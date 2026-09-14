# DEC5 jaw-notch proposals: independent native-depth check

## What was tested

Follow-up to [the topology pilot](dec5_residual_jaw_topology.md), limited to
001193 and 001195. The delivered dynamic phase+30 movie is unchanged. This is
not a claim that the overall artifact-removal goal is complete.

Root: `/mnt/data/dec5_jaw_measured_depth`.

The exact CUDA COLMAP 3.13.0.dev0 / 5509fffe build re-estimates both times with
the existing supervised `run_temporal_full_block_control.py`: 62 fixed train
cameras, 1920 native image width, 12 sources, photometric then geometric passes,
three iterations each. No held-out RGB participates. Historical per-image
JPEG ingest is reproduced **only for the geometry control**; production texture
exposure and frozen physical-camera profiles are untouched.

Each observed-depth set is fused with both original block activation and full
bounded-block activation. `review_jaw_depth_controls.py` compares these with the
published original and boundary-repaired mesh in the same old and phase+30
cameras. Native clay crops, source hashes and camera normalization checks are
retained. Miss counts are local geometry diagnostics, not full-frame quality
metrics.

`diagnose_jaw_measured_depth.py` checks the already-frozen camera-independent
notch triangles, without changing their construction. Ten barycentric samples
per triangle use native measured camera-z, normalized exactly like the original
mesh. Depth agreement tolerance is .001 normalized units, reference roundtrip
1.5 pixels, and viewing separation greater than one degree. A free-space veto
requires a query-camera observation farther by .003 and agreement of that
observed 3D point with at least three **other** real train depth maps. Missing
depth is unknown. These settings reuse the previous confidence pilot; they are
not adjusted per frame.

Two distinct old-surface checks are recorded: agreement in other cameras, and
agreement also in the actual query camera. A farther surface can genuinely
exist but be occluded by the proposed foreground in this view; its existence
alone is not proof that foreground geometry is incorrect. Finite triangle
sampling is diagnostic, not a complete visibility or surface certificate.

The opt-in `guard_jaw_measured_depth.py` uses at least two triangle vertices with
two measured votes, median sample votes at least two, no sampled trusted-free
contradiction, and the existing train-mask evidence. It then raycasts every
added surface pixel in all 62 cameras on integer and half-pixel lattices. A
contradiction requires a farther query-camera observation corroborated by three
other observed depths; iterative removal is supported. This gate was established
after inspecting 001193, then applied unchanged to 001195. No per-time exception.
Proposal construction is camera-independent; the **sample-support diagnostic**
still uses the old virtual camera for its roundtrip reference. A fully train-only
reference for this confidence calculation remains a generalization check.

## Results

Both controls and all 16 matched RGB renders finished. No repair promoted and no
production mesh replaced. Per-stage receipts bind commands, input scripts,
source hashes and the pinned executable; the worker recorded PID, map inventory,
GPU use, available disk and log tails every 30 seconds.

| Time | Native geometric maps | Mean / min coverage | Added triangles after guard | Local depth misses before → after | Local black RGB before → after |
|---|---:|---:|---:|---:|---:|
| 001193 | 62 | .394120 / .274125 | 28 | 45 → 4 | 45 → 4 |
| 001195 | 62 | .392743 / .268044 | 23 | 73 → 51 | 74 → 53 |

These are the original selected spot boxes in the old problem camera, not face
PSNR/SSIM/LPIPS. No new held-out quality metrics were computed; this is not a
held-out generalization or production-acceptance result.

The repeated original fusion and full-block fusion both retain the defect:
44 misses at 001193 in the original mesh (45 after existing boundary/carving
processing), and 73 at 001195. Thus full-block activation alone does not explain
or repair these particular defects. At 001195, repeated fusion has four fewer
triangles overall than the published original; exact global reproducibility is
not claimed from matching local misses.

The 001193 spot samples have median 3.5 measured votes, range 0–16; 39/50 have
at least two and 9/50 have none. No sample has a corroborated far-depth veto.
For 001195, proposals 409 and 412 have a corroborated far observation in F/E;
other spot triangles also lack sufficient vertex anchors. Only proposal 413 of
the five visible spot-closing triangles survives the uniform gate. Therefore
neither “there are no two-view observations anywhere here” nor “the old-mesh
veto is always false” fits both times.

All 124 camera/lattice checks per time report zero surviving corroborated
free-space conflicts, without needing iterative pruning in these two cases.
Original vertex/triangle prefixes are exact after saving. Component counts stay
51→51 and 63→63; no new triangle island or nonmanifold edge was introduced.
These existing many small components belong to the already carved/repaired
baseline, not the original one-component TSDF mesh.

![001193 native old-camera RGB](/mnt/data/dec5_jaw_measured_depth/rgb_review/001193/spot_native.png)
![001195 native old-camera RGB](/mnt/data/dec5_jaw_measured_depth/rgb_review/001195/spot_native.png)

The main agent inspected both native spot pairs, four full-resolution moving
head pairs, four real-train triplets (F/E and M/B at both times), and selected
clay controls. The primary 001193 black speck largely disappears without a new
obvious local seam; tiny residual flecks remain. The 001195 defect remains
visible and fragmented. The lower F/E real view still exposes under-chin tears
in both variants, and old neck-color seams/hair contour artifacts persist.
Verdict: **partial local improvement; reject as a complete two-time fix**.

![001195 lower real camera / baseline / guarded](/mnt/data/dec5_jaw_measured_depth/train_review/001195/F004_E005_1210FP.png)

The paired renderer installs both production wrappers (near-view texture prior
and source masks), with frozen exposure/profiles and hard source selection.
It uses the same camera and actor time in each pair. Only 124/49 RGB pixels
change in the old-camera pairs and 14/20 in phase+30; these localization counts
are not full-frame image-quality metrics. The phase workaround remains the
better delivered shot, not proof of repaired geometry.

Eleven tests pass. The audit checks 282 stage-bound file references per time,
124 final ray checks, mesh prefixes/components, matched render hashes, source
split, frozen identical gate rules, and absence of full-frame quality metrics.
Two audit issues were handled explicitly, not by relaxing checks:

- 001195 normalization translations differ by 1.19e-7 before scene scaling.
  Clay controls now use the exact homogeneous mapping into the published
  coordinate system; stored meshes are not edited.
- Undistortion's auto `patch-match.cfg` is deliberately superseded by the
  following `patch-config` stage. The audit accepts only this exact path/stage
  transition, with the current hash matched to the later bound receipt. The
  existing control runner's generic per-stage resume checker still needs this
  same supersession handling if rerunning an already advanced workspace.

Failure logs are retained alongside successful normalized/final audit logs.
No source, reference artifact or scratch was deleted. All workers are terminal.

## Insights

Observed query-camera depth is a more defensible free-space test than blanket
old-mesh occlusion. It admits a useful tiny repair at 001193 but does not establish
one uniform complete solution at 001195. The inferred cap remains a local prior,
not newly measured anatomy. Do not promote only the favorable time or lower the
threshold just for the unfavorable one.

Next discriminate why 001195's rejected **existing boundary vertices** sometimes
have a farther query observation: pixel-footprint/occlusion-boundary sampling,
stereo disagreement, or a genuinely incorrect proposed contour. Check continuous
projection neighborhoods and the actual train image/depth together before
changing admission. Also replace the virtual sample roundtrip reference with a
deterministic train-based reference and test both times unchanged. Remaining
under-chin tears require a broader shape-support investigation, not automatically
filling the large external boundary. The main goal remains incomplete.
