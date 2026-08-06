# Temporal face dolly videos at 2 m and 1 m

## What was tested

Rendered two additional moving-camera videos from the 45 chronological
LookCloser per-frame snapshots in
`/home/brans/temporal_perframe_stride7_45f`.

- Base view: central training camera `H004_C016` / `frame_train_00032.jpg`.
- Target: midpoint of the actors' face points, reconstructed from model depth at
  pixels `(755, 395)` and `(895, 465)` in temporal frame 11 (`007817`).
- Metric conversion: `dataparser_scale = 0.1366135226` scene units per meter.
- Measured initial camera-to-target distance: `8.414882 m`.
- Variant 2 m: closest measured distance `1.99999993 m`.
- Variant 1 m: closest measured distance `0.99999975 m`.
- Closest approach: temporal index 11, selected because both actors' faces are
  visible there. A piecewise sine-squared/cosine-squared profile gives zero
  camera velocity at the start, closest point, and end.
- The first and last camera transforms are identical. Scene time still advances
  through all 45 snapshots.
- Native resolution: 1920x1080 RGB at exactly `60/7` fps, 5.25 seconds.
- Lossless master: FFV1 level 3 in Matroska. Compatibility copy: H.264 CRF 10,
  `yuv420p`, in MP4.

Reusable renderers:

- [`scripts/render_temporal_snapshot_videos.py`](../scripts/render_temporal_snapshot_videos.py)
  contains snapshot discovery, camera interpolation, rendering, FFmpeg encoding,
  and validation helpers.
- [`scripts/render_temporal_dolly_videos.py`](../scripts/render_temporal_dolly_videos.py)
  preserves the earlier moderate/close dolly variants.
- [`scripts/render_temporal_face_dolly_videos.py`](../scripts/render_temporal_face_dolly_videos.py)
  performs metric face-target calibration and renders the 2 m / 1 m variants.

Reproduction command used:

```bash
export CUDA_HOME=/home/brans/repos/nerfstudio/.cuda128-toolchain
export TORCH_CUDA_ARCH_LIST=12.0
source /home/brans/repos/nerfstudio/.venv/bin/activate
python scripts/render_temporal_face_dolly_videos.py
```

The renderer rejects output paths inside the dataset, records the dataset tree
before and after rendering, refuses to overwrite invalid existing PNGs, and
supports preview/resume operation through `--indices`, `--resolution-scale`,
`--skip-encode`, and `--resume`.

## Results

| Closest distance | Codec | Resolution | Frames / fps | Duration | Size | Validation |
|---|---|---:|---:|---:|---:|---|
| 2 m | [FFV1 lossless](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m/moving_camera_dolly_to_2m_lossless_ffv1.mkv) | 1920x1080 | 45 @ `60/7` | 5.25 s | 98,776,852 B | decoded RGB hashes equal source PNGs |
| 2 m | [H.264 HQ](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m/moving_camera_dolly_to_2m_hq_h264.mp4) | 1920x1080 | 45 @ `60/7` | 5.25 s | 31,197,641 B | complete 45-frame decode passed |
| 1 m | [FFV1 lossless](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m/moving_camera_dolly_to_1m_lossless_ffv1.mkv) | 1920x1080 | 45 @ `60/7` | 5.25 s | 95,259,267 B | decoded RGB hashes equal source PNGs |
| 1 m | [H.264 HQ](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m/moving_camera_dolly_to_1m_hq_h264.mp4) | 1920x1080 | 45 @ `60/7` | 5.25 s | 30,033,461 B | complete 45-frame decode passed |

Review and audit artifacts:

- [Contact sheet](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m/face_dolly_1m_2m_contact_sheet.png)
- [Manifest](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m/manifest.json)
- [Render log](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m_render.log)
- [Dataset tree before](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m/input_tree_before.json)
- [Dataset tree after](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m/input_tree_after.json)

Both dataset-tree manifests have the same SHA-256:
`26d5eab034c35ca0ee471f51b96e058a960415b1e44e1dde242db3be6c7ff9e8`.
The final manifest reports `dataset_unchanged: true`.

## Insights

- A symmetric dolly peaking at temporal index 22 was visually unsuitable: at
  that time the actors had fallen and their faces were outside the close crop.
  Moving closest approach to index 11 makes the camera motion serve the requested
  face close-up while retaining a closed spatial path.
- The 2 m version keeps both actors readable and gives a strong face close-up.
  The 1 m version is intentionally extreme and fills most of the frame with the
  actors' heads and upper bodies.
- Calibrating the target against the model for frame 11 prevents the very close
  path from aiming at the actors' earlier positions.
- The spatial camera path is loopable because its endpoint transforms match.
  The acting timeline is chronological, so the scene content itself does not
  reset between the last and first frames.
