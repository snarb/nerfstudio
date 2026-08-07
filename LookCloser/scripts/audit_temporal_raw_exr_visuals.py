#!/usr/bin/env python3
"""Create low-resolution visual correspondence sheets for the raw EXR rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from convert_temporal_raw_exr_dataset import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    FINAL_MANIFEST_NAME,
    HD_SIZE,
    QHD_SIZE,
    center_crop_box,
    graded_display_u8,
    read_exr_rgb_and_header,
)
from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFAULT_TARGETS = (
    "frame_eval_00001",
    "frame_eval_00002",
    "frame_eval_00003",
    "frame_train_00001",
    "frame_train_00032",
    "frame_train_00066",
)
TILE_WIDTH = 480
TILE_HEIGHT = 270
LABEL_HEIGHT = 34


def font() -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), 18) if path.is_file() else ImageFont.load_default()


def full_and_overlap(path: Path) -> tuple[Image.Image, Image.Image]:
    pixels, _ = read_exr_rgb_and_header(path)
    if pixels.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
        raise RuntimeError(f"Unexpected EXR shape: {path}: {pixels.shape}")
    display, _visual_preview_exposure_gain = graded_display_u8(pixels)
    full = Image.fromarray(display, "RGB").resize((TILE_WIDTH, 240), Image.Resampling.LANCZOS)
    full_tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), "black")
    full_tile.paste(full, (0, (TILE_HEIGHT - full.height) // 2))
    overlap = Image.fromarray(display, "RGB")
    overlap = overlap.crop(center_crop_box(EXPECTED_WIDTH, EXPECTED_HEIGHT, QHD_SIZE))
    overlap = overlap.resize(QHD_SIZE, Image.Resampling.LANCZOS)
    overlap = overlap.resize(HD_SIZE, Image.Resampling.LANCZOS)
    overlap = overlap.resize((TILE_WIDTH, TILE_HEIGHT), Image.Resampling.LANCZOS)
    return full_tile, overlap


def labelled(tile: Image.Image, label: str, typeface: ImageFont.ImageFont) -> Image.Image:
    result = Image.new("RGB", (tile.width, tile.height + LABEL_HEIGHT), (24, 24, 24))
    result.paste(tile, (0, LABEL_HEIGHT))
    ImageDraw.Draw(result).text((8, 7), label, fill="white", font=typeface)
    return result


def small_display(path: Path) -> Image.Image:
    pixels, _ = read_exr_rgb_and_header(path)
    if pixels.shape != (HD_SIZE[1], HD_SIZE[0], 3):
        raise RuntimeError(f"Unexpected small EXR shape: {path}: {pixels.shape}")
    display, _visual_preview_exposure_gain = graded_display_u8(pixels)
    return Image.fromarray(display, "RGB").resize((TILE_WIDTH, TILE_HEIGHT), Image.Resampling.LANCZOS)


def build_small_sheet(small_root: Path, jpeg_root: Path, rows: list[tuple[str, str]], output: Path) -> None:
    typeface = font()
    rendered_rows = []
    for frame, stem in rows:
        exr = small_root / frame / "images" / f"{stem}.exr"
        jpeg = jpeg_root / frame / "images" / f"{stem}.jpg"
        reference = Image.open(jpeg).convert("RGB").resize((TILE_WIDTH, TILE_HEIGHT), Image.Resampling.LANCZOS)
        preview = small_display(exr)
        reference_gray = ImageOps.autocontrast(reference.convert("L")).convert("RGB")
        preview_gray = ImageOps.autocontrast(preview.convert("L")).convert("RGB")
        tiles = (
            labelled(reference, f"{frame}/{stem} old JPEG", typeface),
            labelled(preview, "1920 EXR display preview", typeface),
            labelled(reference_gray, "old JPEG normalized grayscale", typeface),
            labelled(preview_gray, "1920 EXR normalized grayscale", typeface),
        )
        row_image = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "black")
        x = 0
        for tile in tiles:
            row_image.paste(tile, (x, 0))
            x += tile.width
        rendered_rows.append(row_image)
    sheet = Image.new("RGB", (rendered_rows[0].width, sum(row.height for row in rendered_rows)), "black")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)
    print(f"wrote rows={len(rows)} size={sheet.size} output={output}")


def build_sheet(exr_root: Path, jpeg_root: Path, rows: list[tuple[str, str]], output: Path) -> None:
    typeface = font()
    rendered_rows = []
    for frame, stem in rows:
        exr = exr_root / frame / "images" / f"{stem}.exr"
        jpeg = jpeg_root / frame / "images" / f"{stem}.jpg"
        full, overlap = full_and_overlap(exr)
        reference = Image.open(jpeg).convert("RGB").resize((TILE_WIDTH, TILE_HEIGHT), Image.Resampling.LANCZOS)
        ref_array = np.asarray(reference, dtype=np.int16)
        overlap_array = np.asarray(overlap, dtype=np.int16)
        difference = np.clip(np.abs(overlap_array - ref_array) * 4, 0, 255).astype(np.uint8)
        diff_tile = Image.fromarray(difference, "RGB")
        tiles = (
            labelled(reference, f"{frame}/{stem} current JPEG", typeface),
            labelled(overlap, "new EXR central overlap", typeface),
            labelled(diff_tile, "absolute difference ×4", typeface),
            labelled(full, "new EXR full 6144×3072", typeface),
        )
        row_image = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "black")
        x = 0
        for tile in tiles:
            row_image.paste(tile, (x, 0))
            x += tile.width
        rendered_rows.append(row_image)
    sheet = Image.new(
        "RGB",
        (rendered_rows[0].width, sum(row.height for row in rendered_rows)),
        "black",
    )
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)
    print(f"wrote rows={len(rows)} size={sheet.size} output={output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exr-root", type=Path)
    parser.add_argument("--small-exr-root", type=Path)
    parser.add_argument("--jpeg-root", type=Path, required=True)
    parser.add_argument("--frame", action="append", required=True)
    parser.add_argument("--target", action="append")
    parser.add_argument("--include-worst", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.exr_root is None) == (args.small_exr_root is None):
        raise RuntimeError("Provide exactly one of --exr-root or --small-exr-root")
    rows = [(frame, target) for frame in args.frame for target in (args.target or DEFAULT_TARGETS)]
    if args.include_worst:
        if args.exr_root is None:
            raise RuntimeError("--include-worst currently requires --exr-root")
        manifest = json.loads((args.exr_root / FINAL_MANIFEST_NAME).read_text(encoding="utf-8"))
        candidates = []
        for frame in manifest["frames"]:
            for record in frame["outputs"]:
                candidates.append(
                    (
                        float(record["preview_vs_jpeg_psnr"]),
                        frame["frame_name"],
                        Path(record["output_relative"]).stem,
                    )
                )
        for _, frame, target in sorted(candidates)[: args.include_worst]:
            if (frame, target) not in rows:
                rows.append((frame, target))
    if args.small_exr_root is not None:
        build_small_sheet(args.small_exr_root, args.jpeg_root, rows, args.output)
    else:
        build_sheet(args.exr_root, args.jpeg_root, rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
