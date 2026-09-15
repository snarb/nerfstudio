# Cinematic shots with an explicitly real train-video ending

## What was tested

The user requested several dynamic cinematic push-ins, then explicitly chose
actual train RGB for the final second instead of fixing the mesh now. This is
an opt-in **presentation edit**, not an improved reconstruction or an
artifact-free mesh claim. The detailed path/render campaign is documented in
[cinematic push-in choices](dec5_cinematic_pushin_choices.md).

Four shots preserve all 150 actual times `000899..001197`, at 24 fps, with no
slow motion, frozen actor frame or generated intermediate actor image. Physical
camera paths remain inside the calibrated train hull. A short smooth
acceleration precedes the main move and a longer deceleration reaches exactly
train camera `H004_C005_1210SZ` at index 118. Pose and virtual lens then remain
constant; the woman continues moving.

The presentation consists of 118 original 3D-rendered frames, eight explicit
display-domain dissolve frames (indices 118–125), and **24 frames wholly from
the real H/C video** (126–149). The dissolve begins only after actual camera
extrinsics and intrinsics coincide with the ending. No held-out image is used.
The original room background is retained in the real footage and appears
during the dissolve; it is not synthesized or segmented away.

The real RGB uses the existing fixed camera profile and fixed global exposure,
followed by the same Reinhard/sRGB response. A pinhole intrinsics-only mapping
supplies the virtual focal length/sensor window; it needs neither a mesh nor
depth. No new gain fitting, warp, mask or color correction is added. This is
not a claim of unchanged native pixel size: the close-up magnifies the source
using bilinear sampling in linear RGB before the frozen display response.

## Results

Independent audit of the saved matrices, inverted into the original
calibration gauge, gives:

| Variant | Physical radial approach | Path travelled in first second | Final focal multiplier | Sensor x shift |
|---|---:|---:|---:|---:|
| locked_arc | 20.88% | 18.62% | 1.65× | 0 px |
| free_arc | 20.88% | 18.62% | 1.90× | 400 px |
| rising_arc | 18.31% | 15.38% | 1.65× | 0 px |
| soft_diagonal | 5.36% | 26.99% | 1.90× | 400 px |

These are camera/path diagnostics, not image-quality metrics. The wider fourth
arc deliberately trades radial approach for more angular travel. Optical zoom
and sensor shift are disclosed independently, never counted as camera motion.
The endpoint pose error versus the actual calibration is at most `6.67e-16`.
Every source-time and geometry record remains identical to the existing
production inventory. The compositor additionally rejects any pose or lens
change during indices 118–149.

At preparation time, all 128 actual ending RGB frames (32 × four variants) are
saved with source EXR, camera, display-configuration and output hashes. A
separately implemented intrinsics-only comparison at `001151` matches the new
compositor **exactly, maximum uint8 error 0**, for both final framings. Five
focused tests cover half-pixel ray coordinates, exact identity including image
borders, out-of-source requests, immutable pose/lens hold, and the exact
118/8/24 split with byte-preserved endpoints.

A further independent float64, separable-resampling audit verifies **all 128
prepared ending images** against the 32 distinct original EXRs. It does not
call the compositor's sampler or display function: 64 two-framing replays
match within one uint8 level, and the corresponding ending images in the
other two variants match pixel-for-pixel. Two additional tests check this
independent sampler against an analytic affine image and the display response.
The combined main-agent endpoint/audit suite passes seven tests.

The main agent inspected the locked/free transition contact sheets and a native
final beauty frame. Face, eyes and ears align through the dissolve; the old mesh
silhouette briefly remains visible during blending. The original background
and support stand are visible in the whole-head framing. The beauty framing
excludes that stand and the top of the hair by design. These are editorial
tradeoffs, not repaired geometry. Full-video render/publication status and
progressive visual review belong to the linked campaign report.

- [Independent v4 path audit](/mnt/data/dec5_cinematic_pushin_v4/independent_path_audit.json)
- [Independent replay of all ending RGBs](/mnt/data/dec5_cinematic_pushin_v4/independent_real_ending_audit.json)
- [Whole-head transition](/mnt/data/dec5_cinematic_pushin_v4/locked_arc/train_transition_review/contact.png)
- [Beauty transition](/mnt/data/dec5_cinematic_pushin_v4/free_arc/train_transition_review/contact.png)
- [Actual final beauty RGB](/mnt/data/dec5_cinematic_pushin_v4/free_arc/train_ending/frames/001197/frame.png)
- [Native-image endpoint comparison and source categories](/mnt/data/dec5_cinematic_endpoint_comparisons_v3/locked_arc/001151/comparison.png)

## Insights

An exact train-camera position does not make the existing mesh renderer
reproduce its real RGB automatically. Saved source-ID diagnostics show other
camera selections along narrow nose/jaw bands and the erroneous crown fringe,
while most of the face uses H/C. Missing or extraneous geometry can still alter
the silhouette. A real RGB ending bypasses these errors for the final second,
but provides **no evidence that the underlying mesh was repaired**.

`scripts/compose_cinematic_train_ending.py` is the opt-in helper:

```bash
../.venv/bin/python scripts/compose_cinematic_train_ending.py --root /mnt/data/dec5_cinematic_pushin_v4/free_arc --prepare-ending
../.venv/bin/python scripts/compose_cinematic_train_ending.py --root /mnt/data/dec5_cinematic_pushin_v4/free_arc --compose
```

The second command requires verified raw renders only for indices 0–125. Raw
3D output remains under `frames/`; the separately labeled final cut is under
`presentation/frames/`. Atomic per-frame receipts and the final presentation
manifest record real-source versus 3D provenance. Pure 3D and pure train pixels
are checked against their respective inputs after composition. No quality
metric is computed on the hybrid presentation, and existing model/runner
defaults are unchanged.
