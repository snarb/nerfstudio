# Rendering

Generated: 2026-07-08 UTC

## What was tested

This file records the reproducible rendering commands and render hyperparameters used for:

- high-quality camera-path rendering from a single static checkpoint (`007740`);
- high-quality temporal video from one eval view across all 45 trained frames, one checkpoint per output frame, without interpolation.

The important environment detail is that renders must use the same code tree as the temporal training jobs:

```bash
export PYTHONPATH=/home/brans/repos/nerfstudio_time_run:/home/brans/repos/nerfstudio/LookCloser:$PYTHONPATH
export PATH=/home/brans/repos/nerfstudio/.venv/bin:$PATH
export TORCH_CUDA_ARCH_LIST='9.0+PTX'
export TORCH_EXTENSIONS_DIR=/home/brans/.cache/torch_extensions_lookcloser
export PYTHON=/home/brans/repos/nerfstudio/.venv/bin/python
export FFMPEG="$($PYTHON - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
```

Using `/home/brans/repos/nerfstudio_time_run` first on `PYTHONPATH` is required. The earlier black camera-path outputs were produced by using the wrong code tree.

## Static frame camera-path render

### Script

Static camera paths are rendered by:

`scripts/render_central_camera_path_videos.py`

The script loads the full train/eval datamanager cameras via `eval_setup`, selects central camera names from `/home/brans/temporal_perframe_stride7_45f/perframe_manifest.json`, builds a Nerfstudio interpolated camera path, and renders image sequences through `_render_trajectory_video`.

### High-quality 007740 command

The high-quality static leader render used an eval config copy:

`/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/config/eval_config_step_000106316_hq.yml`

That config points to the same leader weights as the original checkpoint. Only inference memory/quality settings were changed:

| Parameter | Value |
|---|---:|
| `max_steps_per_ray` | `2048` |
| `eval_num_rays_per_chunk` | `1024` |
| checkpoint weights | unchanged, `step-000106316.ckpt` |

Render command:

```bash
$PYTHON scripts/render_central_camera_path_videos.py \
  --config /home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/config/eval_config_step_000106316_hq.yml \
  --output-root /home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png \
  --interpolation-steps 8 \
  --frame-rate 24 \
  --downscale-factor 1 \
  --image-format png \
  --skip-video \
  > /home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/render.log 2>&1
```

Render hyperparameters:

| Parameter | Value |
|---|---:|
| source checkpoint | `007740` static leader |
| path source cameras | full train/eval datamanager cameras |
| coordinate handling | no subset dataparser recentering |
| interpolation steps | `8` |
| frame rate | `24` |
| downscale factor | `1.0` |
| image format | PNG |
| JPEG quality | not used |
| master frame compression | PNG sequence, no video compression artifacts |

Camera paths:

| Path | Cameras | Frames |
|---|---|---:|
| `vertical_col_c_D_to_L` | `D004_C014`, `E004_C014`, `F004_C014`, `G004_C014`, `H004_C016`, `I004_C014`, `J004_C014`, `K004_C014`, `L004_C014` | 64 |
| `horizontal_row_h_B_to_D` | `H004_B014`, `H004_C016`, `H004_D014` | 16 |
| `horizontal_row_i_B_to_D` | `I004_B014`, `I004_C014`, `I004_D014` | 16 |
| `diagonal_center_D_B_to_L_D` | `D004_B014`, `F004_C014`, `H004_C016`, `J004_C014`, `L004_D014` | 32 |

### Static video encoding

PNG frames are the master output. Lossless videos were encoded with FFV1:

```bash
for path in vertical_col_c_D_to_L horizontal_row_h_B_to_D horizontal_row_i_B_to_D diagonal_center_D_B_to_L_D; do
  "$FFMPEG" -y -framerate 24 \
    -i "/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/${path}/%05d.png" \
    -c:v ffv1 -level 3 -pix_fmt bgr0 \
    "/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/${path}_hq_lossless_ffv1.mkv"
done
```

High-quality MP4 preview copies were encoded from the same PNG masters:

```bash
for path in vertical_col_c_D_to_L horizontal_row_h_B_to_D horizontal_row_i_B_to_D diagonal_center_D_B_to_L_D; do
  "$FFMPEG" -y -framerate 24 \
    -i "/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/${path}/%05d.png" \
    -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -preset slow -crf 10 -pix_fmt yuv420p \
    "/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/${path}_hq_crf10.mp4"
done
```

The `crop=trunc(iw/2)*2:trunc(ih/2)*2` filter is only for H.264 even-dimension compatibility. It does not affect the PNG masters or FFV1 lossless MKV files.

### Static outputs

Output root:

`/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png`

| Path | PNG frames | Lossless video | Preview video | Resolution |
|---|---:|---|---|---|
| `vertical_col_c_D_to_L` | 64 | `vertical_col_c_D_to_L_hq_lossless_ffv1.mkv` | `vertical_col_c_D_to_L_hq_crf10.mp4` | `1900x1098` |
| `horizontal_row_h_B_to_D` | 16 | `horizontal_row_h_B_to_D_hq_lossless_ffv1.mkv` | `horizontal_row_h_B_to_D_hq_crf10.mp4` | `1915x1103` |
| `horizontal_row_i_B_to_D` | 16 | `horizontal_row_i_B_to_D_hq_lossless_ffv1.mkv` | `horizontal_row_i_B_to_D_hq_crf10.mp4` | `1912x1098` |
| `diagonal_center_D_B_to_L_D` | 32 | `diagonal_center_D_B_to_L_D_hq_lossless_ffv1.mkv` | `diagonal_center_D_B_to_L_D_hq_crf10.mp4` | `1907x1102` |

Verification sheet:

`/home/brans/lookcloser_temporal_runs/videos/camera_paths_007740_hq_fullres_png/camera_path_007740_hq_lossless_verification_sheet.jpg`

## Eval-view temporal render

### Input

The temporal eval-view video uses `eval_img_0000.png` from each selected frame render directory in `experiments/training_on_video.md`.

Nerfstudio eval render files are side-by-side `GT | render` images at `3840x1080`. The video uses the right render half only, producing `1920x1080` output frames.

This path does not rerender the model. It packages already generated full-resolution eval renders from each checkpoint into a temporal video, preserving one checkpoint per output frame and no interpolation.

### Script used

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

FRAMES = [
    ("007740", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_sanity/lookcloser/007740_leader_local_20260705_183929/renders_local_data_step-000106316/eval_img_0000.png"),
    ("007747", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser/007747_const5e-4_20260705_184726/renders_selected_step-000151880/eval_img_0000.png"),
    ("007754", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007754_from_000151880_const5e-4/renders_selected_step-000197444/eval_img_0000.png"),
    ("007761", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007761_from_000197444_const5e-4/renders_selected_step-000243008/eval_img_0000.png"),
    ("007768", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007768_from_000243008_const5e-4/renders_selected_step-000288572/eval_img_0000.png"),
    ("007775", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007775_from_000288572_const5e-4/renders_selected_step-000334136/eval_img_0000.png"),
    ("007782", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007782_from_000334136_const5e-4/renders_selected_step-000379700/eval_img_0000.png"),
    ("007789", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007789_from_000379700_const5e-4/renders_selected_step-000425264/eval_img_0000.png"),
    ("007796", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007796_from_000425264_const5e-4/renders_selected_step-000470828/eval_img_0000.png"),
    ("007803", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007803_from_000470828_constant0p0005/renders_selected_step-000516392/eval_img_0000.png"),
    ("007810", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007810_from_000516392_constant0p0005/renders_selected_step-000561956/eval_img_0000.png"),
    ("007817", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007817_from_000561956_constant0p0005/renders_selected_step-000607520/eval_img_0000.png"),
    ("007824", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007824_from_000607520_constant0p0005/renders_selected_step-000653084/eval_img_0000.png"),
    ("007831", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007831_from_000653084_constant0p0005/renders_selected_step-000698648/eval_img_0000.png"),
    ("007838", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007838_from_000698648_constant0p0005/renders_selected_step-000744212/eval_img_0000.png"),
    ("007845", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007845_from_000744212_constant0p0005/renders_selected_step-000789776/eval_img_0000.png"),
    ("007852", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007852_from_000789776_constant0p0005/renders_selected_step-000835340/eval_img_0000.png"),
    ("007859", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007859_from_000835340_constant0p0005/renders_selected_step-000880904/eval_img_0000.png"),
    ("007866", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007866_from_000880904_constant0p0005/renders_selected_step-000926468/eval_img_0000.png"),
    ("007873", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007873_from_000926468_constant0p0005/renders_selected_step-000972032/eval_img_0000.png"),
    ("007880", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007880_from_000972032_constant0p0005/renders_selected_step-001017596/eval_img_0000.png"),
    ("007887", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007887_from_001017596_constant0p0005/renders_selected_step-001063160/eval_img_0000.png"),
    ("007894", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007894_from_001063160_constant0p0005/renders_selected_step-001108724/eval_img_0000.png"),
    ("007901", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007901_from_001108724_constant0p0005/renders_selected_step-001154288/eval_img_0000.png"),
    ("007908", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007908_from_001154288_constant0p0005/renders_selected_step-001199852/eval_img_0000.png"),
    ("007915", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007915_from_001199852_constant0p0005/renders_selected_step-001245416/eval_img_0000.png"),
    ("007922", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007922_from_001245416_constant0p0005/renders_selected_step-001290980/eval_img_0000.png"),
    ("007929", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007929_from_001290980_constant0p0005/renders_selected_step-001336544/eval_img_0000.png"),
    ("007936", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007936_from_001336544_constant0p0005/renders_selected_step-001382108/eval_img_0000.png"),
    ("007943", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007943_from_001382108_constant0p0005/renders_selected_step-001427672/eval_img_0000.png"),
    ("007950", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007950_from_001427672_constant0p0005/renders_selected_step-001473236/eval_img_0000.png"),
    ("007957", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007957_from_001473236_constant0p0005/renders_selected_step-001518800/eval_img_0000.png"),
    ("007964", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007964_from_001518800_constant0p0005/renders_selected_step-001564364/eval_img_0000.png"),
    ("007971", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007971_from_001564364_constant0p0005/renders_selected_step-001609928/eval_img_0000.png"),
    ("007978", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007978_from_001609928_constant0p0005/renders_selected_step-001655492/eval_img_0000.png"),
    ("007985", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007985_from_001655492_constant0p0005/renders_selected_step-001701056/eval_img_0000.png"),
    ("007992", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007992_from_001701056_constant0p0005/renders_selected_step-001746620/eval_img_0000.png"),
    ("007999", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/007999_from_001746620_constant0p0005/renders_selected_step-001792184/eval_img_0000.png"),
    ("008006", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008006_from_001792184_constant0p0005/renders_selected_step-001837748/eval_img_0000.png"),
    ("008013", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008013_from_001837748_constant0p0005/renders_selected_step-001883312/eval_img_0000.png"),
    ("008020", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008020_from_001883312_constant0p0005/renders_selected_step-001928876/eval_img_0000.png"),
    ("008027", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008027_from_001928876_constant0p0005/renders_selected_step-001959252/eval_img_0000.png"),
    ("008034", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008034_from_001959252_constant0p0005/renders_selected_step-001989628/eval_img_0000.png"),
    ("008041", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008041_from_001989628_constant0p0005/renders_selected_step-002020004/eval_img_0000.png"),
    ("008048", "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/008048_from_002020004_constant0p0005/renders_selected_step-002050380/eval_img_0000.png"),
]

OUT_ROOT = Path("/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png")
FRAMES_DIR = OUT_ROOT / "frames_png"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

manifest = {
    "eval_index": 0,
    "source_format": "Nerfstudio eval_img_0000.png side-by-side GT|render",
    "output": "right render half only",
    "frame_count": len(FRAMES),
    "frame_rate": {"preview": 30.0, "source60_stride7_timeline": 60.0 / 7.0},
    "frames_dir": str(FRAMES_DIR),
    "frames": [],
}

for idx, (frame, source_path) in enumerate(FRAMES):
    source = Path(source_path)
    image = Image.open(source).convert("RGB")
    width, height = image.size
    render_half = image.crop((width // 2, 0, width, height))
    output = FRAMES_DIR / f"{idx:06d}_{frame}.png"
    render_half.save(output, compress_level=0)
    manifest["frames"].append({
        "temporal_index": idx,
        "frame": frame,
        "source": str(source),
        "frame_png": str(output),
        "source_resolution": f"{width}x{height}",
        "render_resolution": f"{render_half.size[0]}x{render_half.size[1]}",
    })

(OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
```

### Eval-view hyperparameters

| Parameter | Value |
|---|---:|
| eval view | `eval_img_0000` |
| frame count | `45` |
| temporal interpolation | none |
| camera interpolation | none |
| source image format | Nerfstudio eval `GT | render` PNG |
| selected half | right render half |
| master frame resolution | `1920x1080` |
| master frame format | PNG |
| PNG compression level | `0` |
| preview FPS | `30.0` |
| source timeline FPS assumption | `60 / 7 = 8.571428571` |

### Eval-view video encoding

```bash
ROOT=/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png

"$FFMPEG" -y -framerate 30 \
  -pattern_type glob -i "$ROOT/frames_png/*.png" \
  -c:v ffv1 -level 3 -pix_fmt bgr0 \
  "$ROOT/eval_view_0000_hq_fps30_lossless_ffv1.mkv"

"$FFMPEG" -y -framerate 30 \
  -pattern_type glob -i "$ROOT/frames_png/*.png" \
  -c:v libx264 -preset slow -crf 10 -pix_fmt yuv420p \
  "$ROOT/eval_view_0000_hq_fps30_crf10.mp4"

"$FFMPEG" -y -framerate 8.571428571 \
  -pattern_type glob -i "$ROOT/frames_png/*.png" \
  -c:v ffv1 -level 3 -pix_fmt bgr0 \
  "$ROOT/eval_view_0000_hq_fps8p571_lossless_ffv1.mkv"

"$FFMPEG" -y -framerate 8.571428571 \
  -pattern_type glob -i "$ROOT/frames_png/*.png" \
  -c:v libx264 -preset slow -crf 10 -pix_fmt yuv420p \
  "$ROOT/eval_view_0000_hq_fps8p571_crf10.mp4"
```

### Eval-view outputs

Output root:

`/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png`

| Output | FPS | Frames | Resolution | Notes |
|---|---:|---:|---|---|
| `frames_png/*.png` | n/a | 45 | `1920x1080` | PNG masters, no temporal interpolation |
| `eval_view_0000_hq_fps30_lossless_ffv1.mkv` | `30.000` | 45 | `1920x1080` | lossless FFV1 |
| `eval_view_0000_hq_fps30_crf10.mp4` | `30.000` | 45 | `1920x1080` | high-quality preview |
| `eval_view_0000_hq_fps8p571_lossless_ffv1.mkv` | `8.571` | 45 | `1920x1080` | source 60 FPS / stride 7 timeline assumption |
| `eval_view_0000_hq_fps8p571_crf10.mp4` | `8.571` | 45 | `1920x1080` | high-quality timeline preview |

Verification sheet:

`/home/brans/lookcloser_temporal_runs/videos/eval_view_0000_hq_fullres_png/eval_view_0000_hq_lossless_verification_sheet.jpg`

## Insights

- Static camera-path rendering must use full train/eval camera objects from `eval_setup`; building a subset dataset separately can change the coordinate frame and produce invalid renders.
- The static HQ camera-path masters are PNG sequences. Use FFV1 MKV for archival/lossless review and CRF10 H.264 only as a convenient preview format.
- The temporal eval-view video is a packaging step from already rendered eval images. It is the safest way to preserve the exact selected per-frame checkpoints and avoid accidental camera or model-parameter changes during video generation.
