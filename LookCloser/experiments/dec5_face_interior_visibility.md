# Train-only face-interior texture visibility

## What was tested

Follow-up to [temporal source retention](dec5_temporal_source_retention.md).
The thin nose-side dark seam persists in the dynamic cinematic sequence.
At time001123, trace its156 saved diagnostic samples back to their actual mesh
intersections, source-camera rays and calibrated train RGB. These points were
selected posthoc by a dark-ridge diagnostic; they are **not** a metric ROI or
an anatomical mask and include a few eye pixels.

Then test a full-image, train-only source-admission rule: bypass the scalar
bilinear/four-tap depth-consistency test only for an exactly mesh-visible point
inside an eroded face-skin source footprint, with at least three originally
valid face-skin witnesses and strictly better existing source quality. Geometry,
target camera, exposure and camera profiles stay unchanged. RGB is sampled from
one real train camera, never averaged or synthesized. Missing old sources are
not filled. Held-out RGB is unused.

The pinned existing MediaPipe multiclass model produces face-skin confidence
for all62 calibrated train crops. A global.95 threshold admitted **zero pixels
in all62 cameras**; that preflight is preserved. The.90 threshold (uint8>=230),
four-pixel interior margin and inherited foreground intersection are reviewed
before rendering. Probabilities are not calibrated geometric confidence.

Two controls use identical masks/rays/quality:

1. Require the old selected source also to have face-skin support.
2. Remove only that old-source semantic veto. Keep its original visibility,
   three other valid skin witnesses, new-source skin, exact visibility and
   strictly higher quality. This addresses the circular veto when the bad old
   source is itself mapped onto an excluded dark contour.

All outputs are isolated HD diagnostic stills, **not** the6K final delivery.
The actual6K video remains unchanged at3456×6144; see
[native6K report](dec5_cinematic_6k_output.md).

## Results

### Source attribution

All156 old-source RGB values reproduce **exactly** from real train RGB and
frozen color profiles. All156 chosen camera-to-point rays hit that point first
on the current mesh (t min/median/max=.999999821/1/1.000000238).
Thus simple continuous-ray rejection of the old source is not a fix.

The closest H/C source is also exactly visible for156/156 points, but the
original raster footprint admits only24/156: the pixel quad spans the nose/cheek
depth break. I/C and J/C admit140/156 and156/156. Native source witnesses show
the dark contour already in I/C/J/C photos, whereas H/C is cleaner. This is
hard-source reprojection, **not two-camera color averaging**. It does not prove
the mesh positions or subpixel registration are anatomically correct.

- [H/C real train witness](/mnt/data/dec5_nose_source_visibility_001123/witnesses/H004_C005_1210SZ.png)
- [I/C real train witness](/mnt/data/dec5_nose_source_visibility_001123/witnesses/I004_C005_1210BA.png)
- [J/C real train witness](/mnt/data/dec5_nose_source_visibility_001123/witnesses/J004_C005_1210I4.png)

### Matched rendered controls

| Control | Changed RGB pixels | Changed diagnostic points /156 | New black pixels | Visual result |
|---|---:|---:|---:|---|
| Old-source skin required |57|10|0|Seam remains conspicuous|
| Three-witness consensus, no old-source skin veto |259|74|0|Seam weaker/interrupted, still visible|

All259 replacements in the second control use H/C. The H/C face mask covers
all156 diagnostic samples, and all156 have at least three originally valid
skin witnesses, but only29 old selected sources pass their own skin mask.
Removing that veto therefore has a direct, measured effect.

The remaining65 diagnostic points not already owned by H/C and not changed
have H/C-to-old quality ratios min/median/max=.02979/.89881/1.17179.
58 are still H/C-raster-invalid and lose on quality; the other7 already admit
H/C and are outside this deliberately new-source-only recovery. The quality
includes per-triangle incidence squared; graph ownership still governs already
valid sources. This localizes residual causes, not permission to blindly relax
all gates. Counters here are diagnostics, not PSNR/SSIM/LPIPS measurements.

- [Native face comparison](/mnt/data/dec5_face_consensus_visibility_001123/review/face.png)
- [Native nose comparison](/mnt/data/dec5_face_consensus_visibility_001123/review/nose.png)
- [Full overview](/mnt/data/dec5_face_consensus_visibility_001123/review/overview.png)
- [Explicit visual verdict](/mnt/data/dec5_face_visibility_recovery_review.json)

The main agent viewed19 images:62-camera mask overview, three native mask
contexts, three real-source witnesses, and all12 comparison panels across both
controls, including every changed tile. The line is still visible. Hair/neck
contour errors are untouched. **Neither control is promoted** as an artifact-free
video or mesh fix. No temporal stability claim and no full150-frame rerender.

Independent audits reconstruct all57 and259 changed mesh intersections and
source UVs, replay original visibility, skin votes, source quality, continuous
first hits and actual calibrated RGB. All replays pass exactly (ray tests use
the stated1e-5 tolerance). Every unchanged-source pixel is bit-identical;
all target depths match the original raycast. The mesh is hash-identical.
Five focused tests cover visibility misses/occluders, three-view consensus,
old-source invalidity/missing IDs, strict quality and the semantic-veto ablation.

One preparation failed because the mounted output filesystem rejects copying
timestamps with shutil.copy2. The failed directory and log are preserved at
`/mnt/data/dec5_face_consensus_visibility_001123_prepare_permission_failure`.
Using content-only copyfile fixed preparation; no failed output was published.

## Insights

The seam has a reproducible renderer contribution: the closest clean source is
excluded by a scalar pixel-footprint rule at a facial depth discontinuity.
Train semantic consensus can safely limit a visibility experiment to interior
skin, but requiring the **bad old** sample to satisfy that same semantic mask
blocks most of the useful corrections. The ablation improves the line without
painting over missing geometry or averaging hair detail.

It is nevertheless incomplete. Examine the remaining triangle normals and
graph ownership before testing a combined angular/visibility policy across
actual dynamic times. Same-mesh ray agreement and segmentation cannot certify
true shape. The separate cheek-hole and lipstick/neck membrane reconstruction
work remains necessary; this experiment does not improve or repair the mesh.

Reproduce in fresh, no-overwrite roots using the repository venv (two CPU
threads); semantic inference uses the existing isolated MediaPipe environment:

```text
scripts/diagnose_nose_source_visibility.py
scripts/infer_train_face_support.py
scripts/study_face_interior_visibility.py prepare
scripts/run_face_interior_visibility_90.py prepare
scripts/run_face_interior_visibility_90.py render
scripts/review_face_interior_visibility.py /mnt/data/dec5_face_interior_visibility90_001123
scripts/run_face_consensus_visibility.py prepare
scripts/run_face_consensus_visibility.py render
scripts/review_face_consensus_visibility.py
scripts/audit_face_visibility_recovery.py /mnt/data/dec5_face_interior_visibility90_001123 /mnt/data/dec5_face_consensus_visibility_001123
```

No existing runner/model defaults changed; previously executed helpers remain
hash-pinned. The report records a partial improvement, not goal completion.
