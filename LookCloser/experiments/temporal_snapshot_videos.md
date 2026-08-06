# Temporal snapshot videos

## What was tested

Rendered the 45 selected per-frame LookCloser snapshots from
`/home/brans/temporal_perframe_stride7_45f` in chronological order (`007740` through
`008048`, source stride 7 at 60 fps).

- Comparison video: central training camera `H004_C016` / `frame_train_00032.jpg`,
  with an unscaled 50/50 vertical split between GT and model output.
- Moving-camera video: closed path
  `H004_B014 -> H004_D014 -> J004_E014 -> L004_E014 -> L004_B014 -> J004_A014 -> H004_B014`.
- Native render resolution: 1920x1080 RGB.
- Timeline: 45 frames at exactly `60/7` fps, giving 5.25 seconds.
- Lossless master: FFV1 level 3 in Matroska; compatibility copy: H.264 CRF 10 in MP4.

The render utility is `scripts/render_temporal_snapshot_videos.py`. It refuses an output
directory inside the dataset, records input file size/mtime manifests before and after the
run, and does not create or alter files under the dataset root.

## Results

| Video | Codec | Resolution | Frames / fps | Duration | Size | Validation |
|---|---:|---:|---:|---:|---:|---|
| [GT vs rendered FFV1](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final/gt_vs_rendered_lossless_ffv1.mkv) | FFV1 `bgr0` | 1920x1080 | 45 @ `60/7` | 5.25 s | 114,713,101 B | decoded RGB hashes equal PNGs |
| [GT vs rendered H.264](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final/gt_vs_rendered_hq_h264.mp4) | H.264 `yuv420p`, CRF 10 | 1920x1080 | 45 @ `60/7` | 5.25 s | 35,612,142 B | ffprobe + visual pass |
| [Moving camera FFV1](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final/moving_camera_temporal_lossless_ffv1.mkv) | FFV1 `bgr0` | 1920x1080 | 45 @ `60/7` | 5.25 s | 107,842,648 B | decoded RGB hashes equal PNGs |
| [Moving camera H.264](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final/moving_camera_temporal_hq_h264.mp4) | H.264 `yuv420p`, CRF 10 | 1920x1080 | 45 @ `60/7` | 5.25 s | 35,202,872 B | ffprobe + visual pass |

Visual review artifacts:

- [comparison contact sheet](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final/comparison_contact_sheet.png)
- [moving-camera contact sheet](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final/moving_contact_sheet.png)
- [render manifest](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final/manifest.json)
- [render log](/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final_render.log)

The final manifest reports `dataset_unchanged: true`; the before/after input tree manifests
are identical. All three PNG sequences contain exactly 45 valid 1920x1080 RGB frames.

## Insights

- A ray chunk of `131072` reached roughly 83 GiB reserved memory and caused one CUDA illegal
  memory access after repeated model swaps. The interrupted run preserved 20 complete frames.
- Re-rendering the failing snapshot in a fresh process with `65536` succeeded. Resuming the
  campaign at `65536` completed all remaining frames with roughly 37 GiB active GPU memory;
  this is now the utility default.
- The camera pose is spatially closed: frames 0 and 44 use the same camera transform. Scene
  time remains chronological, so looping wraps from the final temporal snapshot to the first.
