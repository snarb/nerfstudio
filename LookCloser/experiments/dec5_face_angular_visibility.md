# Face-only angular visibility: removing the cinematic nose seam

## What was tested

The previous [skin-consensus recovery](dec5_face_interior_visibility.md) only
weakened the line. Here diagnose its remaining normals and source ownership,
then test two matched full-image controls at001123:

1. **Raster:** prefer the closer optical-axis camera inside three-view-supported
   train face skin, but retain the old depth-footprint visibility test.
2. **Consensus:** same preference, additionally admitting exact first-hit source
   rays whose eroded RGB footprint lies in train face skin even when the scalar
   depth-footprint test rejects them.

Both require an originally visible old source, at least three originally valid
skin witnesses and strictly better angular weight. Both test exact source rays.
Already raster-valid sources may now replace old graph choices. No image
averaging, target RGB, geometry edit, profile refit or exposure change. Hair and
other regions are not globally switched to angular-only selection; the prior
global angular-only quality regression is not ignored.

The identical consensus rule is transferred to actual video times001083,
001119 and001127 with independently inferred62-camera train masks. Camera pose
and moving actor geometry come from each original cinematic frame. These four
**noncontiguous HD diagnostic times are not a continuous video acceptance test**
and do not replace the delivered3456×6144 video.

## Results

### Why normal smoothing is not the remedy

At the156 saved nose diagnostic samples, the closest H/C camera has median
absolute incidence .14556, versus .25337 for I/C. Face normals and the normalized
mean of vertex normals differ by median3.17°, maximum20.58°. The median sampled
first triangle edge is .000548 normalized units. Vertex-normal smoothing would
favor H/C over the old source at75 points rather than81, not recover the seam.
This is largely a coherent grazing surface, not randomly flipped tiny normals.
It is still **estimated mesh geometry**, not proof of correct anatomy.

With raster-only angular selection, H/C owns24/156 points (7 changes); the
line remains. With skin-consensus admission, H/C owns156/156 (139 changes),
and the black line disappears in the native comparison. The real nose shading
remains. Neither change alone was sufficient in the tested controls.

- [Three-way native nose](/mnt/data/dec5_face_angular_visibility/001123/review/nose.png)
- [Three-way native face](/mnt/data/dec5_face_angular_visibility/001123/review/face.png)
- [Saved normal/source evidence](/mnt/data/dec5_face_angular_visibility/normal_diagnosis.json)

### Transfer to moving actor/camera times

| Time | Mode | Source-ID changes | RGB changes | Newly zero RGB pixels |
|---|---|---:|---:|---:|
|001123|Raster|266|264|0|
|001123|Consensus|619|615|0|
|001119|Consensus|657|651|0|
|001127|Consensus|470|463|0|
|001083|Consensus|133160|132304|5|

All replacements choose physical H/C for these particular camera poses; it is
not hard-coded as the source. The same nose-line improvement transfers to001119
and001127 without parameter changes. At001083, however, a large part of the face
switches source, changing skin detail/shading and eye reflection. This reveals
an important temporal-risk region around competing cameras; four stills cannot
prove absence of popping.

The five newly black pixels at001083 are portrait(892,835),(896,837..840),
inside the pupil/highlight area. Old RGB was not near zero. Each new sample
exactly replays from real H/C RGB, and has existing geometry and a valid source:
they are **not newly missing mesh or source holes**. That does not establish the
physically correct view-dependent reflection. All five marked crops were viewed.

- [001119 native changes](/mnt/data/dec5_face_angular_visibility/transfer_review/001119_native_00.png)
- [001127 native changes](/mnt/data/dec5_face_angular_visibility/transfer_review/001127_native_00.png)
- [001083 broad transfer, page1](/mnt/data/dec5_face_angular_visibility/transfer_review/001083_native_00.png)
- [001083 broad transfer, page2](/mnt/data/dec5_face_angular_visibility/transfer_review/001083_native_01.png)
- [001083 pupil witness](/mnt/data/dec5_face_angular_visibility/transfer_review/001083_black_0.png)

The main agent inspected27 images: all eight pilot panels, three train-mask
overviews, six transfer overview/face panels, five native contact sheets covering
all28 changed transfer tiles, and five marked black-pixel crops. No temporal
playback claim. The late nose seam is removed; existing hair/neck contours,
cheek-hole geometry and lipstick/neck membrane are not repaired by this work.
Production and the current6K movie are unchanged.

Independent audits replay every changed point in all five renders: exact target
mesh intersections and depth, source projection, original visibility, semantic
votes, angular weight and calibrated hard-source RGB. The old independent audit
is adapted only at its quality expression and mode-specific visibility assertion;
both original and generated code hashes are retained. Same-mesh rays are not
independent depth truth. Pixels outside source replacements are bit-identical.

Three new unit tests cover the raster/consensus difference, missing/invalid old
sources, three-view skin quorum and the single-statement renderer adapter.
All original runner/model defaults and previously executed helpers are unchanged.

## Insights

A local rendering cause is now reproducibly corrected on three actual moving
views: the depth-footprint rule rejects the closest clean source at a facial
depth break, while incidence weighting and graph ownership retain a side-view
contour. Train skin consensus plus exact visibility and angular selection
addresses that combination without averaging texture.

Before whole-video adoption, test a contiguous dynamic clip, especially near
camera ties where001083 changes a broad region. If texture popping appears,
test a geometry-driven locality guard near depth discontinuities, not per-frame
camera exceptions. The mask includes facial features such as eyes and lips;
it is not an exact skin-only material classifier. Source agreement does not
solve view-dependent eye highlights or certify physical shape.

The separate mesh objective remains open. Do not call this a reconstruction
improvement: all meshes and target depths are unchanged. Cheek-hole completion
and removal of the false lipstick/neck membrane still need temporal integration.

Entry points (fresh no-overwrite roots; repository venv except MediaPipe infer):

```text
study_face_angular_visibility.py prepare --mode raster
study_face_angular_visibility.py render --mode raster
study_face_angular_visibility.py prepare --mode consensus
study_face_angular_visibility.py render --mode consensus
stage_face_visibility_transfer.py stage --frame FRAME
stage_face_visibility_transfer.py infer --frame FRAME
transfer_face_angular_visibility.py prepare --frame FRAME
transfer_face_angular_visibility.py render --frame FRAME
review_face_angular_visibility.py FRAME_MODE_ROOT [OTHER_MATCHED_MODE_ROOT]
audit_face_angular_visibility.py FRAME_MODE_ROOT ...
diagnose_nose_face_normals.py
pack_face_angular_review.py
```

Inference uses the existing isolated MediaPipe environment and pinned local
model; no download or environment upgrade. Frame-varying masks do not alter
fixed camera color/exposure profiles. See the explicit
[visual notes](/mnt/data/dec5_face_angular_visibility/visual_notes.json).
