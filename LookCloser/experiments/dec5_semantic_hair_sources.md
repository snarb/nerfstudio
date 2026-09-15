# DEC5: train-consensus semantic hair source selection

## What was tested

2026-09-15. The preceding [interior-source study](dec5_interior_texture_sources.md)
reduced background-colored hair fringe but changed shading at the neck/shoulder.
This experiment keeps each original source unless several visible train
observations classify the surface as hair and a cleaner hair source exists.

Two official MediaPipe models provide fallible train-only semantic priors:
[Hair Segmenter and Selfie Multiclass](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter).
The former separates hair/background; the latter separates hair, face/body
skin, clothing, accessories and background. These are semantic probabilities,
**not alpha/opacity, depth, or ground truth**. The
[multiclass model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf)
documents the model and its Apache-2.0 license. No generative image model is used.

Model SHA-256:

```text
hair_segmenter.tflite
2628cf3ce5f695f604cbea2841e00befcaa3624bf80caf3664bef2656d59bf84
selfie_multiclass_256x256.tflite
c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0
```

`build_train_hair_semantics.py` stages calibrated display RGB from all 62 train
cameras. It rotates to portrait and uses the same fixed input crop
`[0,350,1080,1600]` in every source camera/time. Areas outside that crop are
unknown, not background evidence. The model input may clip the top of the head
in some rig views; those unobserved regions cannot vote. Inference runs on CPU
in the existing isolated MediaPipe 0.10.21/Python 3.12 environment; the
reconstruction environment is unchanged. Three crop-sized confidence channels
are retained as `round(probability*255)` with input/model hashes.

The policy in `study_semantic_hair_sources.py`:

- Reproject each target-ray surface point into the original 62 train views.
- Keep all existing masked mesh-depth and four-tap visibility checks.
- Require both models to give hair probability at least 0.75 in at least three
  visible, known cameras, representing at least half of visible known cameras.
- Veto a change when at least three visible cameras confidently classify
  skin/clothing/accessories (probability at least 0.75).
- For eligible hair, consider only visible sources with both hair predictions
  above threshold and at least 16 HD pixels inside the original person mask.
- Retain an already-clean original source. Otherwise choose one eligible source
  by the frozen angular/incidence quality. If none exists, retain the original.
- Reuse original graph labels. Preserve original sources everywhere else and
  never create a source for a previously source-less pixel.

There is no camera averaging, texture blur, output-image mask, geometric edit,
pose change, exposure change, per-frame manual override, or temporal RGB blend.
The native-6K movie being rendered concurrently remains completely unchanged.

## Results

Initial two actual wide-spiral camera poses, before the real-train ending:

| Time | Changed source pixels | Changed RGB pixels | Protected pixels changed | Newly zero RGB pixels |
|---|---:|---:|---:|---:|
| 001083 | 66,292 | 66,285 | 0 | 1 |
| 001123 | 34,795 | 34,792 | 0 | 0 |

At both times, target depth, missing-source masks and graph labels are exactly
equal to the published baseline. Every pixel retaining its original source is
RGB-byte-identical. Counts above are diagnostics, not face metrics. No
PSNR/SSIM/LPIPS is invented for interpolated cameras without exact RGB GT.

The newly zero RGB sample at `001083` is portrait `(254,1017)`, not a ray miss:
its depth is 0.7156238 and source is `B004_B005_1210Z3`. Its source coordinate
is approximately `(844.3942,349.0452)`. Bilinear interpolation of the existing
linear EXR samples is negative in all three channels and the existing display
pipeline clamps it to black. This is not new missing geometry. It is retained
and disclosed rather than hidden by a post-hoc pixel fill.

The main LLM inspected eight native model-input/confidence sheets and six
paired crown/face/body crops. Large portions of the brown fringe improve at
both times. The policy is less aggressive than the global interior-source
variant and leaves some brown edge portions. The dark neck transition induced
by that global variant is avoided; the original face/body appearance remains.
Jagged triangles, stretched hair texture, crown opening and existing face
lines still remain. This is not an artifact-free approval or a mesh repair.

- [001083 crown](/mnt/data/dec5_semantic_hair_sources/review/001083_crown.png)
- [001123 face/neck preservation](/mnt/data/dec5_semantic_hair_sources/review/001123_face.png)
- [Train semantic evidence](/mnt/data/dec5_train_hair_semantics/review/001083_H004_C005_1210SZ.png)

### Consecutive temporal control

The same frozen policy was evaluated on twelve consecutive source times
`001073..001095`, step two, with their twelve original moving wide-spiral poses.
`study_semantic_hair_temporal.py` records this configuration explicitly and
uses a separate output root. The fixed rate is 24 fps: a half-second diagnostic,
not a replacement for full-video temporal validation.

All twelve frames completed. Changed source counts (min/median/max) were
59,404 / 69,223 / 73,761 pixels. Protected-surface changes were zero throughout;
depth, graph labels, missing-source masks and RGB at unchanged sources remained
exact. Seven newly zero RGB pixels appeared across the sequence (0–2 per frame),
without new missing sources. Only the `001083` pixel was individually traced as
described above; the other six have not been attributed individually. Repeated
`001083` outputs are byte-identical to the independent two-frame pilot.

The LLM inspected all three overview sheets, all three unscaled hair-crop sheets,
and three additional full-resolution crown/face panels: 23 reviewed images total
including the pilot and semantic evidence. Reduced tan fringe was consistent
across the twelve times, without a new broad face/neck color jump in these
images. Ragged crown geometry, stretched hair and residual brown patches remain.
This was consecutive-frame inspection, not continuous video playback; fine
temporal flicker and full-video quality are not approved.

- [Four consecutive native hair comparisons](/mnt/data/dec5_semantic_hair_temporal_v2/review/crown_04.png)
- [Moving-camera diagnostic, 12 frames / 0.5 s](/mnt/data/dec5_semantic_hair_temporal_v2/review/diagnostic_12f.mp4)
- [Final visual verdict and inspected image hashes](/mnt/data/dec5_semantic_hair_sources/visual_review.json)

An initial staged-only controller imported reconstruction dependencies eagerly,
which were unavailable in the isolated inference environment. Its input stage
and original controller source are preserved at
`/mnt/data/dec5_semantic_hair_temporal`. The import-only portability correction
uses a clean configuration at `/mnt/data/dec5_semantic_hair_temporal_v2`; no
model, input crop, source policy, or old artifact was overwritten.

### Reproduction and audit

The isolated inference environment is required only for `infer`. Staging and
rendering use the existing reconstruction environment. Existing roots are frozen;
changed configurations must use new roots.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/study_semantic_hair_temporal.py stage
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/brans/lookcloser_temp/mediapipe_hand_env/bin/python scripts/study_semantic_hair_temporal.py infer
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/study_semantic_hair_temporal.py render
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/study_semantic_hair_temporal.py review
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/study_semantic_hair_temporal.py package
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python scripts/seal_semantic_hair_study.py check
```

The independent audit verifies 868 train-image records across the two-frame and
twelve-frame runs, their model/input/source hashes, confidence shapes, exact
camera inventory, and fourteen render receipts. It independently repeats the
geometry/source/RGB preservation checks and verifies twelve distinct actual
times and camera positions. All twelve MP4 frames decode at 1080×1920, 24 fps,
duration 0.5 s. Decode validity is not visual approval. Earlier producer receipts
retain `visual_status=pending`; the separate final visual verdict is authoritative
for this finite study and explicitly does not approve production.

Six focused semantic/interior-policy unit tests passed. The artifact manifest
binds 2,841 retained outputs and implementation/report files. No full-frame or
invented face-quality metrics were computed for these interpolated views.

## Insights

1. Train-derived semantic evidence can localize a source-selection improvement
   without changing skin shading everywhere. It must be combined with visibility
   and uncertainty; a model label alone is not proof of correspondence.
2. Preserving original source IDs outside the qualified domain is a stronger
   control than merely keeping the same graph cost or mask settings: pixel
   equality is directly testable.
3. Semantic confidence cannot remove the background component already mixed
   with fine strands in a source pixel. Interior-source selection reduces that
   contamination but does not estimate alpha or reconstruct individual strands.
4. Geometric holes and source contamination remain distinct problems. Neither
   this policy nor sharper 6K RGB establishes that the cheek/crown mesh is fixed.
