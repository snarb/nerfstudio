# DEC5 jaw repair: train anchors and fractional depth footprints

## What was tested

Follow-up to [the measured-depth pilot](dec5_jaw_measured_depth.md). Existing
native 62-camera COLMAP depths for 001193/001195 are reused after checksum and
normalization checks. No new stereo reconstruction, learned inference, color
fitting, or production-video replacement. The original geometry, bounded 3D
notch proposals, train masks, numerical tolerances and final ray veto are frozen.

`study_jaw_depth_footprint.py` removes the virtual output camera from sample
confidence: the physical camera whose observed depth most closely agrees with
the sample becomes its reference. Ties use camera name. This agreeing anchor
counts once; other cameras must still pass the previous 1.5-pixel roundtrip,
.001 normalized-depth agreement and greater-than-one-degree parallax test.
Missing depth cannot establish an anchor. This changes the reference protocol,
not just the count offset; old and new counts are not numerically identical.

`study_jaw_train_confidence.py` runs two arms, identically at both times:

1. **train_anchor:** change only the sample-confidence reference.
2. **footprint:** additionally, reject a fractional sample as observed free
   space only if all four enclosing native depth samples are farther by .003,
   and each is corroborated by at least three other observed train depth maps.
   Mixed/missing neighborhoods are unknown, not positive surface evidence.

Both arms use the **unchanged** `guard_jaw_measured_depth.py` final veto: every
added surface pixel in all 62 train cameras is checked on integer and half-pixel
ray lattices against nearest native measured depth and corroboration. Thus the
fractional-sample change cannot bypass an actual final rasterized contradiction.
No per-time tolerance, patch shape, mask or camera-path exception is made.

Roots: `/mnt/data/dec5_jaw_depth_footprint` and
`/mnt/data/dec5_jaw_train_confidence`. These remain experiment artifacts, not
replacement production meshes or a serialized raw TSDF volume.

## Results

### The rejected point

At 001195, both sample-level vetoes (proposals 409 and 412) refer to the **same
existing boundary vertex**, not two independent bad regions. In F/E it projects
to `(718.92694, 713.52216)`. Nearest rounding chooses `(719,714)`: its normalized
depth residual is `.00303699`, just over `.003`, with four other supporting
views. Adjacent samples have materially different depths, including near-depth
and missing observations. It is therefore not a uniformly corroborated far
surface over the fractional footprint. This does not prove every neighbor or
the proposed anatomy is correct; it diagnoses a fragile veto at this location.

![Native RGB and 9x9 measured-depth neighborhood; delta x1000 / other-view votes](/mnt/data/dec5_jaw_depth_footprint/001195/veto_00.png)

### Geometry and RGB

| Time | Variant | Added triangles | Local depth misses |
|---|---|---:|---:|
| 001193 | Published boundary mesh | 0 | 45 |
| 001193 | Previous virtual-reference gate | 28 | 4 |
| 001193 | Train reference only | 37 | 4 |
| 001193 | Train reference + footprint | 39 | 4 |
| 001195 | Published boundary mesh | 0 | 73 |
| 001195 | Previous virtual-reference gate | 23 | 51 |
| 001195 | Train reference only | 33 | 23 |
| 001195 | Train reference + footprint | 44 | **0** |

In the combined arm the unchanged final rasterized veto removes one triangle at
001195; its next pass is clean. All four candidates pass 124 final camera/lattice
checks. Original saved vertex/triangle prefixes are exact; no new disconnected
islands or nonmanifold edges. Existing component counts stay 51 and 63.

Sixteen RGB renders compare baseline and the combined arm at both times, in the
old moving camera, current phase+30 camera, lower real F/E and upper real M/B.
Both production renderer wrappers are installed. Exposure, physical-camera
profiles, source masks and hard-source texture selection are unchanged. All RGB
comes from the same time's 62 real train images; held-out RGB is not used.

In the old problem camera, local black RGB counts improve **45→4** and **74→0**.
The second point's depth/RGB improvement is not just a geometry-only rendering.
Only 140/147 RGB pixels change in those full native images; phase+30 changes
14/33. These are edit-localization diagnostics, **not full-frame quality metrics**.

![001193 exact native crop](/mnt/data/dec5_jaw_train_confidence/footprint/rgb_review/001193/spot_native.png)
![001195 exact native crop](/mnt/data/dec5_jaw_train_confidence/footprint/rgb_review/001195/spot_native.png)

The main agent directly inspected these two native pairs, four moving head
pairs, four real-train triplets, both red-addition clay crops, and the rejected
point's RGB/depth panel. The previously conspicuous 001195 spot is gone in the
old camera without a new obvious local seam; 001193 retains tiny flecks. No
obvious new defect attributable to these additions was seen in the inspected
real views. Lower F/E under-chin tears, other M/B silhouette holes, old neck
source seams and hair-contour defects remain. **Positive local canary, not an
artifact-free full-frame or full-video pass.**

No new held-out PSNR/SSIM/LPIPS were computed and no held-out generalization is
claimed. Three new focused tests cover deterministic train anchors, missing
depth and a mixed four-pixel footprint. They pass alongside the eleven prior
geometry/confidence tests. Audits verify measured-depth receipts, all four mesh
prefixes/topology/ray gates, identical per-time rules, and all sixteen render
receipts and matched camera/exposure/source protocols. Source/reference data and
published videos remain unchanged; no scratch was deleted. All jobs finished.

## Insights

The older failed transfer combined two avoidable restrictions: confidence tied
to a virtual output view, and a whole-triangle veto based on rounding a
fractional boundary point to one unstable depth pixel. Separating these factors
improves both evidence interpretation and actual surface/render coverage while
retaining the stricter measured-depth final visibility check.

This explains rejection of the **repair**, not the entire original TSDF hole
formation process. The cap is still an inferred bounded local prior. It is not a
learned anatomical reconstruction or proof that depth observations are correct.

Next prioritize the larger hand/forearm failures in the actual dynamic movie:
the existing three-time forearm v3 pilot has not yet been transferred into the
production renderer with source masks and view prior. Preserve the published
head/carving geometry when appending its local delta, then test the same real
depth visibility gates and native RGB. Further jaw generalization also needs
additional times and held-out evaluation before any all-150 geometry promotion.
The camera/actor motion requirement remains satisfied by the phase+30 movie;
the full no-visible-artifacts objective remains open.
