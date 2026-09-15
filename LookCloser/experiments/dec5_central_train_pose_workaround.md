# Central train-pose workaround: three-time diagnostic

## What was tested

The user permitted temporarily routing the moving camera through real train
poses to avoid visible cheek holes. Test that premise before changing the movie:
seven exact central poses, columns E..K in row C, at actual actor instants
`001037`, `001123`, `001193`. Row C is two rows from both vertical edges.

Each pose was rendered with native intrinsics and, independently, the unchanged
movie intrinsics. A fresh current-moving-pose control brings the total to
15 views per instant, **45 verified renders**. Each instant uses its own original
published production mesh and its own pose normalization; no geometry repair,
time substitution, crop, image stabilization, new exposure, or RGB averaging.
All use the same corrected native texture footprint. Only train RGB is consumed.
Native predictions are compared to fixed-profile, same-time real train GT.
These are diagnostic snapshots, **not a new dynamic video** or a mesh improvement.

Virtual target IDs deliberately differ from physical source IDs: this prevents
the source-mask wrapper indexing a native mask in a target with different
intrinsics. Source masks themselves are unchanged. The pose/intrinsics audit
independently checks all 45 requests and completed-output hashes.

## Results

| Actual instant | Observed result | Decision |
|---|---|---|
| 001037 | Torn wrist/hand remains in exact train poses. Native G..K partly or mostly clip the hand; widening to the unchanged movie intrinsics exposes the defects again. Hair has a rough brown rim. | No acceptable hand workaround; clipping is not a repair. |
| 001123 | Left central views expose a jagged under-chin boundary. More frontal/right views retain a thin dark jaw/neck seam. Crown gaps and raised fragments remain across the tested columns. | No clean combined cheek/crown waypoint established. |
| 001193 | Left views avoid the small isolated under-jaw fleck seen on the right, but show a false sharp nose contour and retain rough hair/neck boundaries. Right views retain the under-jaw black fleck. | Local visibility trade-off, not a uniformly improved replacement. |

The main agent viewed all seven native head/hand pairs and all seven wide-FOV
overviews for 001037, plus three native-resolution wide-FOV hand crops. For each
late instant, all 15 native-resolution head panels were viewed. GT overviews
were inspected separately. The explicit inspected-file hashes are in each
`visual_review.json`; uninspected panels are not presented as inspected.

This diagnostic does **not** compute new image-quality metrics. Different poses
have different GT/framing; output coverage is a geometric diagnostic, not a face
quality score. No full-frame PSNR/SSIM/LPIPS or loss was introduced.

Artifacts:

- [001037 train-pose controls](/mnt/data/dec5_central_train_pose_probe/visual_review.json),
  [E/C hand vs GT](/mnt/data/dec5_central_train_pose_probe/review/native_E004_C005_1210YM_hand.png).
- [001123 visual review](/mnt/data/dec5_central_train_pose_transfer/001123/visual_review.json),
  [E/C head vs GT](/mnt/data/dec5_central_train_pose_transfer/001123/head_review/native_E004_C005_1210YM.png),
  [K/C head vs GT](/mnt/data/dec5_central_train_pose_transfer/001123/head_review/native_K004_C005_1210BC.png).
- [001193 visual review](/mnt/data/dec5_central_train_pose_transfer/001193/visual_review.json),
  [G/C head vs GT](/mnt/data/dec5_central_train_pose_transfer/001193/head_review/native_G004_C005_121037.png),
  [K/C head vs GT](/mnt/data/dec5_central_train_pose_transfer/001193/head_review/native_K004_C005_1210BC.png).
- [Hash manifest](/mnt/data/dec5_central_train_pose_transfer/artifact_manifest.json).

Three disjoint render workers ran on clever-shadow; sampled VRAM was about
15 GB. The original multi-time launcher stopped before the second instant because
the verified renderer patch cannot be installed twice in one interpreter.
`run_central_train_probe_isolated.py` resumes the **same frozen requests** in one
child process per instant. All 001123 receipts were reused; nothing was deleted
or regenerated to hide this launcher error. Original failure logs are retained.
The initial 001037 request has a `not_a_dynamic_video=False` metadata typo;
its audit explicitly corrects the interpretation without rewriting the request.
Late-time requests correctly record `not_a_dynamic_video=True`.

Ten focused tests passed, covering pose/intrinsics identity, native-mask gauge,
actual-time isolation, per-process wrapper installation and texture footprint.
All workers terminated; source inputs, production meshes and published movie
remain unchanged. The 150-frame rollout was not launched from these snapshots.

## Insights

Exact train-pose positioning does not guarantee a clean result when the render
still uses an incomplete mesh and hard source selection: it is not simply a
copy of that train photograph. Here, central native views retain defects, so
interpolation between cameras cannot be their sole cause.

Camera avoidance remains a valid temporary option, but this particular central
row sweep does not establish a better full-shot path. It also does not prove
that every possible trajectory is bad. Do not trade the late tiny jaw defect
for larger early hand damage, false nose edges, or reduced subject visibility.
A subsequent path candidate must retain dynamic actor times and substantial
camera travel, and be tested through its smooth transitions—not just endpoints.
The separate geometry objective remains open.

Reproducible resume (one worker shown; use disjoint indices 0, 1, 2):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python \
  scripts/run_central_train_probe_isolated.py --worker 0 --workers 3
../.venv/bin/python scripts/freeze_central_train_pose_probe.py --check
```
