"""Build downloadable, camera-verified AABB comparisons from actual renders."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from nerfstudio.cameras.cameras import Cameras
from marshawn_aabb_overlay import overlay_box, project_box


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    manifest = json.loads(args.manifest.read_text())
    out = args.output
    out.mkdir(exist_ok=False)
    cameras = {}
    for row in manifest["cameras_actual_runtime"]:
        pose = np.eye(4)
        pose[:3] = row["camera_to_world_nerfstudio"]
        cameras[row["camera"]] = dict(w=row["width"], h=row["height"],
            fl_x=row["fx"], fl_y=row["fy"], cx=row["cx"], cy=row["cy"],
            transform_matrix=pose.tolist())
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    records, panels = [], {}
    for variant in ["tight", "medium", "loose"]:
        entry = manifest["variants"][variant]
        bounds = entry["exact_AABB_prepared_world"]
        folder = out / variant
        folder.mkdir()
        for name, camera in cameras.items():
            source = Path(entry["files"][name]["premult_black"])
            annotated, projection = overlay_box(Image.open(source), bounds, camera,
                f"{name} / {variant.upper()} / same model 20k")
            corners, uv, _ = project_box(bounds, camera)
            # Independent installed camera implementation verifies conventions.
            actual = Cameras(camera_to_worlds=torch.tensor(camera["transform_matrix"],
                dtype=torch.float32)[:3], fx=camera["fl_x"], fy=camera["fl_y"],
                cx=camera["cx"], cy=camera["cy"], width=camera["w"], height=camera["h"])
            coords = torch.tensor(np.stack([uv[:, 1] + .5, uv[:, 0] + .5], -1), dtype=torch.float32)
            rays = actual.generate_rays(camera_indices=0, coords=coords)
            expected = corners - np.asarray(camera["transform_matrix"])[:3, 3]
            expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
            ray_error = float(np.max(np.linalg.norm(rays.directions.numpy() - expected, axis=-1)))
            assert ray_error < 2e-6, (name, variant, ray_error)
            destination = folder / f"{name}_red_dashed.png"
            annotated.save(destination)
            shutil.copy2(source, folder / f"{name}_raw_black.png")
            shutil.copy2(entry["files"][name]["straight_RGBA"], folder / f"{name}_raw_RGBA.png")
            records.append(dict(camera=name, variant=variant, exact_bounds=bounds,
                source=str(source), source_sha256=sha(source), annotated=str(destination),
                annotated_sha256=sha(destination), independent_Cameras_ray_error=ray_error,
                projection=projection))
            # The same crop for all sizes within a camera preserves comparisons.
            _, crop_uv, _ = project_box(manifest["variants"]["loose"]["exact_AABB_prepared_world"], camera)
            x0, y0 = np.maximum(np.floor(crop_uv.min(0) - 36), 0).astype(int)
            x1, y1 = np.minimum(np.ceil(crop_uv.max(0) + 36), [camera["w"], camera["h"]]).astype(int)
            patch = annotated.crop((x0, y0, x1, y1))
            patch.thumbnail((630, 670), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (640, 725), (15, 15, 15))
            tile.paste(patch, ((640 - patch.width) // 2, 50 + (670 - patch.height) // 2))
            draw = ImageDraw.Draw(tile)
            dims = np.diff(np.asarray(bounds), axis=0)[0]
            draw.text((10, 5), f"{name} / {variant.upper()} / 20k", font=font, fill="white")
            draw.text((10, 28), "box: " + " x ".join(f"{v:.3f}" for v in dims), fill=(210, 210, 210))
            panels[name, variant] = tile
    contact = Image.new("RGB", (1920, 2175))
    for row, name in enumerate(["DC", "HC", "KA"]):
        per_camera = Image.new("RGB", (1920, 725))
        for col, variant in enumerate(["tight", "medium", "loose"]):
            tile = panels[name, variant]
            contact.paste(tile, (640 * col, 725 * row))
            per_camera.paste(tile, (640 * col, 0))
        per_camera.save(out / f"comparison_{name}.png")
    contact.save(out / "comparison_all.png")
    (out / "README.txt").write_text(
        "Три размера AABB: tight, medium, loose; виды DC, HC, KA.\n"
        "Одна и та же модель R1 на 20000 шагах. Меняются только границы интеграции лучей.\n"
        "Красные пунктирные линии — все 12 геометрически спроецированных рёбер бокса.\n"
        "raw_black и raw_RGBA сохранены без линий. Размеры боксов указаны в координатах сцены.\n"
        "Внутри боксов ещё присутствуют пол, мат и артефакты исходной модели.\n"
        "Это сравнение размеров AABB; варианты обучения с мягкими масками готовятся отдельно.\n",
        encoding="utf8")
    receipt = dict(source_manifest=str(args.manifest), source_manifest_sha256=sha(args.manifest),
        checkpoint_sha256=manifest["checkpoint_sha256"], same_model=True, new_training=False,
        renderer_pixels_unchanged_except_declared_overlay=True, records=records,
        generator_sha256=sha(__file__), overlay_source_sha256=sha(Path(__file__).with_name("marshawn_aabb_overlay.py")))
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, Path(out.name) / path.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(dict(output=str(out), archive=str(archive), bytes=archive.stat().st_size,
        archive_sha256=sha(archive), images=len(records), maximum_ray_error=max(r["independent_Cameras_ray_error"] for r in records))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
