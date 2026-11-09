#!/usr/bin/env python3
"""Audit a completed signed AP1 master; export a separate PQ-code EXR control.

No training, source image changes, exposure fitting, or accepted-package edits.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
from PIL import Image, ImageDraw
from prepare_marshawn_canonical_pq import vendor, sha, write, export_exr, srgb_preview


def oetf(value):
    value = np.clip(value, 0, 1)
    return np.rint(np.where(value <= .0031308, value * 12.92,
                            1.055 * value ** (1 / 2.4) - .055) * 255).astype(np.uint8)


def audit(root, old_data):
    completion = json.loads((root / "complete.json").read_text())
    pq = vendor()
    gain = completion["canonical_contract"]["gain_to_nits"]
    data = root / "linear_ap1_exr"
    meta = json.loads((data / "transforms.json").read_text())
    old = json.loads((old_data / "transforms.json").read_text())
    if len(meta["frames"]) != 66:
        raise ValueError("Incomplete canonical data")
    for actual, source in zip(meta["frames"], old["frames"]):
        if {k:v for k,v in actual.items() if k!="file_path"} != {k:v for k,v in source.items() if k!="file_path"}:
            raise ValueError("Camera geometry changed")
    all_hashes = {}
    for row in completion["records"]:
        for output in row["outputs"].values():
            path = Path(output["path"])
            if sha(path) != output["sha256"]:
                raise ValueError("Output hash mismatch")
            all_hashes[str(path.relative_to(root))] = output["sha256"]
    luminances, pixel_count = [], 0
    for frame in meta["frames"]:
        if frame["file_path"] not in meta["train_filenames"]:
            continue
        rgb = cv2.imread(str(data / frame["file_path"]), cv2.IMREAD_UNCHANGED)[..., ::-1]
        if rgb.dtype != np.float32 or not np.isfinite(rgb).all() or (rgb < 0).any():
            raise ValueError("Expected finite positive relative AP1 float32")
        mask_path = frame.get("mask_path")
        if not mask_path:
            raise ValueError("Explicit valid-border mask required")
        mask = cv2.imread(str(data / mask_path), cv2.IMREAD_GRAYSCALE) > 0
        y = (rgb.astype(np.float64) @ pq.AP1_LUMA)[mask].astype(np.float32)
        luminances.append(y)
        pixel_count += len(y)
    values = np.concatenate(luminances)
    del luminances
    quantiles = np.quantile(values, [.5, .999])
    del values
    preview = root / "physical_and_matched_display_preview"
    preview.mkdir(exist_ok=False)
    old_contract = json.loads((old_data / "color_contract.json").read_text())
    ev = {r["camera"]:r["exposure_correction_ev"] for r in old_contract["rows"]}
    sheet = Image.new("RGB", (5 * 480, 3 * 285), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(["DC", "HC", "KA"]):
        frame = next(f for f in meta["frames"] if f["physical_camera"] == name)
        original = next(f for f in old["frames"] if f["physical_camera"] == name)
        rgb = cv2.imread(str(data / frame["file_path"]), cv2.IMREAD_UNCHANGED)[..., ::-1]
        linear709 = np.maximum(rgb.astype(float) @ pq.rgb_to_rgb_matrix("ap1", "rec709").T, 0)
        panels = [("Current scalar SDR", Image.open(old_data / original["file_path"]).convert("RGB")),
            ("Signed nits/203 CLIP + sRGB", Image.fromarray(oetf(linear709 * gain / 203))),
            ("Signed nits/203 Reinhard+sRGB", Image.fromarray(srgb_preview(linear709 * gain / 203))),
            ("Display ONLY x4, no scalar", Image.fromarray(srgb_preview(linear709 * 4))),
            ("Display ONLY x4 + old scalar", Image.fromarray(srgb_preview(linear709 * 4 * 2 ** ev[name])))]
        for j,(label,image) in enumerate(panels):
            image.save(preview / f"{name}_panel{j}.png")
            image.thumbnail((480,253));sheet.paste(image,(480*j,285*i+27))
            draw.text((480*j+3,285*i+3), name+" "+label,fill="white")
    sheet.save(preview / "comparison.jpg", quality=94)
    result = dict(passed=True, source_complete_sha256=sha(root / "complete.json"),
        source_signed_manifest_sha256=completion["canonical_contract"]["signed_manifest_sha256"],
        source_script_sha256=sha(__file__), numeric_camera_geometry_equal=True,
        source_RGB_no_extra_gain_or_scalar=True, files_sha256=all_hashes,
        train_only_statistics=dict(cameras=63,valid_border_pixels=pixel_count,
            luminance_weights_vendor_AP1_Y=pq.AP1_LUMA.tolist(),
            relative_AP1_Y_q50=float(quantiles[0]),relative_AP1_Y_q999=float(quantiles[1]),
            signed_nits_Y_q50=float(quantiles[0]*gain),signed_nits_Y_q999=float(quantiles[1]*gain),
            eval_RGB_used=False),
        preview_sha256=sha(preview / "comparison.jpg"),
        preview_limits="Display-only x4/scalar columns do not alter canonical EXR/PQ. Physical203 reference previews use AP1->Rec709 before clip or Reinhard; no exposure fitting.",
        training_gate="Requires root visual and independent actual-tier matrix review; QA alone is not train authorization")
    write(root / "QA.json", result)
    print(json.dumps(dict(QA=str(root/"QA.json"),statistics=result["train_only_statistics"])),flush=True)


def codes(root, output):
    if not (root / "QA.json").exists() or json.loads((root / "QA.json").read_text())["passed"] is not True:
        raise ValueError("Canonical master must seal and pass QA first")
    source = root / "pq_ap1_png16"
    meta = json.loads((source / "transforms.json").read_text())
    contract = json.loads((source / "color_contract.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "masks").symlink_to((source / "masks").resolve(), target_is_directory=True)
    records = []
    remap = {}
    for frame in meta["frames"]:
        path = source / frame["file_path"]
        png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., ::-1]
        if png.dtype != np.uint16:
            raise ValueError("Actual AP1 PQ16 archive required")
        rgb = png.astype(np.float32) / np.float32(65535)
        if not np.array_equal(np.rint(rgb.astype(float)*65535).astype(np.uint16), png):
            raise ValueError("PQ16 code identity failed")
        relative = "images/" + Path(frame["file_path"]).stem + ".exr"
        target = output / relative
        export_exr(target, rgb)
        records.append(dict(camera=frame["physical_camera"], source_PQ16_sha256=sha(path),
                            relative=relative, sha256=sha(target), uint16_roundtrip_exact=True))
        remap[frame["file_path"]] = relative
        print("PQcode EXR", frame["physical_camera"], flush=True)
    for frame in meta["frames"]:
        frame["file_path"] = remap[frame["file_path"]]
    for split in ["train_filenames","val_filenames","test_filenames"]:
        meta[split] = [remap[p] for p in meta[split]]
    write(output / "transforms.json", meta)
    pq = vendor()
    definition = dict(contract,representation="NONLINEAR_AP1_ST2084_PQ16_codes_in_float32_EXR",
        encoding="EXR contains actual archival uint16 PQ RGB divided by65535; NOT scene-linear and NOT nits",
        source_canonical_QA_sha256=sha(root / "QA.json"),
        background_PQ_code=pq.bg_pq_code_quantized(.005),
        numerical_training="Ordinary encoded-code RGB objective; native sigmoid output and float32 image cache; do not apply linear-HDR PQ encoder a second time",
        common_display="PQ EOTF nits -> divide52.13360318135631 -> AP1D60toRec709D65 Bradford -> displaygain4 -> Reinhard+sRGB; no camera scalar/no fitted display exposure")
    write(output / "color_contract.json", definition)
    write(output / "complete.json", dict(images=66,train=63,eval=3,records=records,
        source_script_sha256=sha(__file__),source_canonical_QA_sha256=sha(root / "QA.json"),
        no_training_launched=True,not_a_physical_linear_master=True))
    print(json.dumps(dict(complete=str(output/"complete.json"),images=66)),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["qa","codes"])
    p.add_argument("--canonical",type=Path,required=True)
    p.add_argument("--old-data",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args();cv2.setNumThreads(1)
    if a.mode=="qa":audit(a.canonical,a.old_data)
    else:codes(a.canonical,a.output)
