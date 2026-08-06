# Temporal interpolated-camera dolly to 2 m

## What was tested

Rendered a combined moving-camera video from all 45 chronological LookCloser
per-frame snapshots in `/home/brans/temporal_perframe_stride7_45f`.

The camera motion combines two independent components:

1. The previously validated closed interpolation through medium training views:
   `H004_B014 -> H004_D014 -> J004_E014 -> L004_E014 -> L004_B014 -> J004_A014 -> H004_B014`.
2. A gradual radial dolly toward the reconstructed midpoint between the actors'
   faces. The distance changes from `8.283362 m` to exactly `2.0 m` at temporal
   index 11 and returns to `8.283362 m` at index 44.

The face target is calibrated from the model depth for frame `007817` at central
camera pixels `(755, 395)` and `(895, 465)`. The dolly uses a sine-squared
approach followed by a cosine-squared return. The original interpolated camera
rotations and intrinsics are retained.

The reusable renderer is
[`scripts/render_temporal_interpolated_dolly_2m_video.py`](../scripts/render_temporal_interpolated_dolly_2m_video.py).
It uses the camera interpolation, encoding, and validation helpers in
[`scripts/render_temporal_snapshot_videos.py`](../scripts/render_temporal_snapshot_videos.py)
and the metric target calibration from
[`scripts/render_temporal_face_dolly_videos.py`](../scripts/render_temporal_face_dolly_videos.py).

Reproduction command:

```bash
export CUDA_HOME=/home/brans/repos/nerfstudio/.cuda128-toolchain
export TORCH_CUDA_ARCH_LIST=12.0
source /home/brans/repos/nerfstudio/.venv/bin/activate
python scripts/render_temporal_interpolated_dolly_2m_video.py
```

## Results

| Codec | Resolution | Frames / fps | Duration | Size | Validation |
|---|---:|---:|---:|---:|---|
| [FFV1 lossless](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_interpolated_dolly_2m/moving_camera_interpolated_dolly_to_2m_lossless_ffv1.mkv) | 1920x1080 | 45 @ `60/7` | 5.25 s | 98,126,906 B | decoded RGB hashes equal source PNGs |
| [H.264 HQ](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_interpolated_dolly_2m/moving_camera_interpolated_dolly_to_2m_hq_h264.mp4) | 1920x1080 | 45 @ `60/7` | 5.25 s | 31,988,295 B | complete 45-frame decode passed |

Review and audit artifacts:

- [Contact sheet](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_interpolated_dolly_2m/interpolated_dolly_2m_contact_sheet.png)
- [Manifest](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_interpolated_dolly_2m/manifest.json)
- [Render log](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_interpolated_dolly_2m_render.log)
- [Dataset tree before](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_interpolated_dolly_2m/input_tree_before.json)
- [Dataset tree after](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_interpolated_dolly_2m/input_tree_after.json)

The first and last combined camera transforms match exactly: maximum position
and rotation error are both `0.0`. The dataset-tree manifests are byte-identical
and have SHA-256
`26d5eab034c35ca0ee471f51b96e058a960415b1e44e1dde242db3be6c7ff9e8`.
The manifest reports `dataset_unchanged: true`.

## Insights

- Radially blending each interpolated camera position toward one fixed 3D target
  preserves the lateral and diagonal train-camera movement while adding a
  separately controlled physical dolly.
- Keeping the original interpolated rotations gives the same viewing behavior as
  the earlier moving-camera video instead of forcing every frame into a synthetic
  look-at orientation.
- Temporal index 11 remains the best closest-approach point: both people and a
  clear face are visible in the 2 m crop.
- The spatial camera path is loopable. Scene time remains chronological, so the
  actors' state does not reset at the video boundary.
