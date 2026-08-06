# Temporal dolly-in videos

## What was tested

Rendered two additional 45-frame moving-camera videos from the chronological
LookCloser snapshots in `/home/brans/temporal_perframe_stride7_45f`.

- Base view: central training camera `H004_C016` / `frame_train_00032.jpg`.
- Camera motion: physical translation along the camera's forward optical axis while
  keeping its orientation fixed.
- Moderate variant: offset `0.08 -> 0.28 -> 0.08` scene units.
- Close variant: offset `0.14 -> 0.42 -> 0.14` scene units.
- Easing profile: `start + (peak - start) * sin(pi * normalized_time)^2`.
- Native render resolution: 1920x1080 RGB.
- Timeline: 45 chronological frames at exactly `60/7` fps, giving 5.25 seconds.
- Lossless master: FFV1 level 3 in Matroska; compatibility copy: H.264 CRF 10 in MP4.

The reusable renderer is
[`scripts/render_temporal_dolly_videos.py`](../scripts/render_temporal_dolly_videos.py).
It builds on [`scripts/render_temporal_snapshot_videos.py`](../scripts/render_temporal_snapshot_videos.py)
for snapshot discovery, rendering, encoding, and validation. Both utilities refuse to
write inside the dataset root.

## Results

| Camera move | Codec | Resolution | Frames / fps | Duration | Size | Validation |
|---|---:|---:|---:|---:|---:|---|
| [Moderate FFV1](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly/moving_camera_dolly_moderate_lossless_ffv1.mkv) | FFV1 `bgr0` | 1920x1080 | 45 @ `60/7` | 5.25 s | 106,695,666 B | decoded RGB hashes equal PNGs |
| [Moderate H.264](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly/moving_camera_dolly_moderate_hq_h264.mp4) | H.264 `yuv420p`, CRF 10 | 1920x1080 | 45 @ `60/7` | 5.25 s | 33,495,611 B | complete decode passed |
| [Close FFV1](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly/moving_camera_dolly_close_lossless_ffv1.mkv) | FFV1 `bgr0` | 1920x1080 | 45 @ `60/7` | 5.25 s | 105,177,698 B | decoded RGB hashes equal PNGs |
| [Close H.264](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly/moving_camera_dolly_close_hq_h264.mp4) | H.264 `yuv420p`, CRF 10 | 1920x1080 | 45 @ `60/7` | 5.25 s | 33,069,448 B | complete decode passed |

Review and reproduction artifacts:

- [two-row contact sheet](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly/dolly_variants_contact_sheet.png)
  (moderate on top, close on bottom; frames 0, 11, 22, 33, and 44)
- [render manifest](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly/manifest.json)
- [render log](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly_render.log)
- [input tree before](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly/input_tree_before.json)
- [input tree after](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly/input_tree_after.json)

The manifest reports `dataset_unchanged: true`, and the before/after dataset tree
manifests are byte-identical. Each rendered sequence contains exactly 45 valid PNGs.
All four video streams decode without FFmpeg errors.

## Insights

- Using the central camera keeps the people centered throughout the scene action. The
  moderate move retains more of the set, while the close move makes the people and faces
  visibly larger without an excessive crop.
- Both camera paths have identical first and last transforms. The spatial loop therefore
  closes without a camera-position jump; scene time still advances chronologically.
- Rendering both variants from each loaded temporal snapshot avoids loading every model
  twice and keeps the two sequences exactly synchronized.
