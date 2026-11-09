#!/usr/bin/env python3
"""Signed Marshawn AP1/PQ export; no fitted exposure or extra global gain.

The EXR stores relative linear AP1, with the actual vendor whitepoint declared
from source constants. Signed nits gain/floor/content peak are applied only to
the PQ export and explicitly labelled previews. Existing artifacts are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tifffile

TASK = Path("/fsx/tmp/marshawn_001660_hd")
PREP = Path("/fsx/oregon/gta_marshawn_2_1_prep_20261028")
PQ_SOURCE = Path("/fsx/oregon/gta_marshawn_2_1/six_sec_20261019/code/engine_gta_pv_export/ca4d/ca4d/pq.py")
RAW = Path("/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local")
RIG = PREP / "root/Marshawn_2_1/golden/rig"
SIGNED = PREP / "manifests/footage_range_signed.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def vendor():
    spec = importlib.util.spec_from_file_location("marshawn_canonical_vendor_pq", PQ_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def srgb_preview(relative709):
    value = np.maximum(relative709, 0)
    value = value / (1 + value)
    encoded = np.where(value <= .0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - .055)
    return np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)


def linear_ap1_hd(row, pq):
    from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text, read_images_text
    name = row["physical_camera"]
    source = RAW / name / (name + ".000408.tif")
    raw = tifffile.memmap(source, mode="r")
    if raw.shape != (4320, 8192, 3) or raw.dtype != np.uint16:
        raise ValueError("Expected original lens-distorted uint16 RGB TIFF")
    linear = np.empty(raw.shape, np.float32)
    matrix = pq.rgb_to_rgb_matrix("rwg", "ap1")
    negative = 0
    for y in range(0, 4320, 128):
        # Exact vendor order: do not clamp RWG before its AP1 conversion.
        value = pq.log3g10_decode(raw[y:y + 128].astype(np.float64) / 65535) @ matrix.T
        negative += int((value < 0).sum())
        linear[y:y + len(value)] = np.maximum(value, 0).astype(np.float32)
    image = next(v for v in read_images_text(RIG / "images.txt").values() if v.name.startswith(name + "/"))
    camera = read_cameras_text(RIG / "cameras.txt")[image.camera_id]
    fx, fy, cx, cy, *distortion = camera.params
    original_k = np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1.]])
    width, height = row["w"], row["h"]
    target_k = np.array([[row["fl_x"], 0, row["cx"] - .5], [0, row["fl_y"], row["cy"] - .5], [0, 0, 1.]])
    map_x, map_y = cv2.initUndistortRectifyMap(original_k, np.array(distortion), None, target_k,
                                            (width, height), cv2.CV_32FC1)
    small = cv2.resize(linear, (width, height), interpolation=cv2.INTER_AREA)
    del linear
    result = cv2.remap(small, (map_x + .5) * width / 8192 - .5,
                       (map_y + .5) * height / 4320 - .5, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Nonfinite or negative linear master")
    return result, dict(raw=str(source), raw_sha256=sha(source), global_frame=1660,
                        local_segment_frame=408, mapping="local = global - 1252",
                        negative_ap1_channel_fraction_before_clip=negative / raw.size)


def export_exr(path, rgb):
    value = np.asarray(rgb, np.float32)
    if not cv2.imwrite(str(path), np.ascontiguousarray(value[..., ::-1]),
                       [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
                        cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]):
        raise OSError(path)
    if not np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., ::-1], value):
        raise ValueError("EXR float32 lossless roundtrip failed")


def contract(pq, signed):
    chosen = signed["chosen"]
    if signed["signed_off"] is not True or chosen["name"] != "p999-half-peak":
        raise ValueError("Expected signed take-specific scale")
    gain, floor, peak = [float(chosen[k]) for k in ("gain_to_nits", "min_nits", "peak_nits")]
    if (gain, floor, peak) != (52.13360318135631, .005, 4000.):
        raise ValueError("Signed scale changed; explicit review required")
    return dict(kind="vendor_signed_Marshawn_AP1_relative_linear_master", global_frame=1660,
        signed_manifest=str(SIGNED), signed_manifest_sha256=sha(SIGNED),
        vendor_pq_source=str(PQ_SOURCE), vendor_pq_sha256=sha(PQ_SOURCE),
        export_script_sha256=sha(__file__), linear_exr_primaries="ACES AP1",
        actual_ap1_white_xy=list(pq.CHROMA["ap1"][3]),
        whitepoint_note="Actual vendor constants are ACES D60; historical exporter tag ap1-d65 conflicts with these constants. Numeric vendor matrix is authoritative here.",
        raw_space="REDWideGamutRGB/Log3G10 uint16 TIFF", raw_white_xy=list(pq.CHROMA["rwg"][3]),
        rwg_to_ap1_matrix=pq.rgb_to_rgb_matrix("rwg", "ap1").tolist(),
        ap1_to_rec709_preview_matrix=pq.rgb_to_rgb_matrix("ap1", "rec709").tolist(),
        master_units="Relative camera-linear AP1 RGB; signed gain NOT baked into EXR",
        gain_to_nits=gain, min_nits=floor, content_peak_nits=peak, pq_normalization_nits=10000.,
        physical_pq="AP1 channels * signed gain, clamp each channel to [.005,4000] nits, exact ST2084, uint16 round",
        extra_global_gain=1., camera_scalar_corrections=False, white_balance="as-shot unchanged",
        source_mapping="global 1660 -> segment-local 408, offset1252; DATA_LOCATIONS_f1300-2020_20261105.md",
        sampling="AP1 negative clip after matrix; linear area-prefilter then one calibrated lens remap at prepared HD K",
        vendor_native_pixel_identity_claim=False,
        native_pixel_identity_limit="This HD derivative preserves prepared K/poses but uses linear prefilter before remap; vendor native tier remaps full-resolution linear AP1 before any downstream reduction.",
        physical_preview="AP1 -> Rec709 Bradford; *52.133603/203 then Reinhard+sRGB. 203 nits per linear RGB display unit is a display reference, not the signed scene gain.",
        matched_display_only="AP1 -> Rec709; *4 then Reinhard+sRGB; optional old scalar is explicitly display-only, never baked into canonical master or PQ.",
        eval_rgb_used_to_fit_gains=False, hdr_loss_rgb_space="AP1 relative linear -> SAME AP1 PQ channels; no implicit Rec2020 conversion")


def previews(output, data, meta, pq, scale, rows, cached=None):
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((data / "color_contract.json").read_text())
    ev = {r["camera"]:float(r["exposure_correction_ev"]) for r in old["rows"]}
    sheet = Image.new("RGB", (4 * 480, len(rows) * 285), "#202020")
    draw = ImageDraw.Draw(sheet)
    receipts = []
    for i, row in enumerate(rows):
        name = row["physical_camera"]
        if cached is None:
            ap1, source = linear_ap1_hd(row, pq)
        else:
            ap1 = cv2.imread(str(cached / (Path(row["file_path"]).stem + ".exr")), cv2.IMREAD_UNCHANGED)[..., ::-1]
            source = {}
        linear709 = np.maximum(ap1.astype(float) @ pq.rgb_to_rgb_matrix("ap1", "rec709").T, 0)
        panels = [("Current scalar SDR PNG", Image.open(data / row["file_path"]).convert("RGB")),
            ("Canonical: signed nits /203, Reinhard+sRGB", Image.fromarray(srgb_preview(linear709 * scale / 203))),
            ("Display ONLY: x4, no camera scalar", Image.fromarray(srgb_preview(linear709 * 4))),
            ("Display ONLY: x4 + old declared scalar", Image.fromarray(srgb_preview(linear709 * 4 * 2 ** ev[name])))]
        for j, (label, image) in enumerate(panels):
            image.save(output / f"{name}_panel{j}.png")
            image.thumbnail((480, 253))
            sheet.paste(image, (480 * j, 285 * i + 27))
            draw.text((480 * j + 3, 285 * i + 3), name + " " + label, fill="white")
        receipts.append(dict(camera=name, old_scalar_ev=ev[name], raw=source))
        print("preview", name, flush=True)
    sheet.save(output / "comparison.jpg", quality=94)
    write(output / "receipt.json", dict(display_only=True, master_unmodified=True,
        no_new_fit=True, rows=receipts, vendor_pq_sha256=sha(PQ_SOURCE),
        source_contract_sha256=sha(data / "color_contract.json"),
        comparison_sha256=sha(output / "comparison.jpg")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preview", "build"])
    parser.add_argument("--data", type=Path, default=TASK / "color_variants/gain4_scalar_trainfit_poseeval/sdr_png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    meta = json.loads((args.data / "transforms.json").read_text())
    if len(meta["frames"]) != 66 or len(meta["train_filenames"]) != 63 or len(meta["val_filenames"]) != 3:
        raise ValueError("Expected the frozen 66-camera, 63/3 prepared split")
    if sha(args.data / "transforms.json") != "79893ade1ef2fbab92b9f1e0891bcb925be9ae4abc8d79a21238c209c5fb4406":
        raise ValueError("Prepared geometry/input manifest changed")
    pq = vendor()
    definition = contract(pq, json.loads(SIGNED.read_text()))
    scale = definition["gain_to_nits"]
    eval_rows = [next(r for r in meta["frames"] if r["physical_camera"] == n) for n in ["DC", "HC", "KA"]]
    if args.mode == "preview":
        previews(args.output, args.data, meta, pq, scale, eval_rows)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    kinds = {"linear_ap1_exr":".exr", "pq_ap1_png16":".png", "canonical_sdr203_png":".png"}
    for kind in kinds:
        (args.output / kind / "images").mkdir(parents=True)
        (args.output / kind / "masks").symlink_to((args.data / "masks").resolve(), target_is_directory=True)
    def one(row):
        ap1, source = linear_ap1_hd(row, pq)
        stem = Path(row["file_path"]).stem
        exr = args.output / "linear_ap1_exr/images" / (stem + ".exr")
        export_exr(exr, ap1)
        codes = np.rint(pq.encode_tier(ap1.astype(float) * scale, .005, 4000.) * 65535).astype(np.uint16)
        archive = args.output / "pq_ap1_png16/images" / (stem + ".png")
        if not cv2.imwrite(str(archive), np.ascontiguousarray(codes[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(archive)
        if not np.array_equal(cv2.imread(str(archive), cv2.IMREAD_UNCHANGED)[..., ::-1], codes):
            raise ValueError("PQ uint16 roundtrip mismatch")
        linear709 = np.maximum(ap1.astype(float) @ pq.rgb_to_rgb_matrix("ap1", "rec709").T, 0)
        preview = args.output / "canonical_sdr203_png/images" / (stem + ".png")
        Image.fromarray(srgb_preview(linear709 * scale / 203)).save(preview)
        nits = pq.pq_decode(codes.astype(float) / 65535)
        expected = np.clip(ap1.astype(float) * scale, .005, 4000.)
        receipt = dict(camera=row["physical_camera"], raw=source,
            finite=True, relative_linear_max=float(ap1.max()),
            pq_nits_quantization_mean_abs=float(np.abs(nits - expected).mean()),
            pq_nits_quantization_max_abs=float(np.abs(nits - expected).max()),
            floor_channel_fraction=float((ap1 * scale < .005).mean()),
            peak_channel_clip_fraction=float((ap1 * scale > 4000).mean()),
            outputs={kind:dict(path=str(path), sha256=sha(path)) for kind,path in
                [("linear_ap1_exr",exr),("pq_ap1_png16",archive),("canonical_sdr203_png",preview)]})
        print("converted", row["physical_camera"], flush=True)
        return receipt
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(one, meta["frames"]))
    for kind, ext in kinds.items():
        copied = json.loads(json.dumps(meta))
        remap = {r["file_path"]:"images/" + Path(r["file_path"]).stem + ext for r in copied["frames"]}
        for row in copied["frames"]:
            row["file_path"] = remap[row["file_path"]]
        for split in ["train_filenames", "val_filenames", "test_filenames"]:
            copied[split] = [remap[p] for p in copied[split]]
        write(args.output / kind / "transforms.json", copied)
        write(args.output / kind / "color_contract.json", dict(definition,
            representation=kind, source_transforms_sha256=sha(args.data / "transforms.json"),
            encoding=("Relative AP1 linear float32" if kind=="linear_ap1_exr" else
                "Signed AP1 ST2084 uint16" if kind=="pq_ap1_png16" else "Display-only canonical signed-nits/203 Reinhard+sRGB uint8")))
    write(args.output / "signed_range_excerpt.json", dict(chosen=definition,
        signed_by=json.loads(SIGNED.read_text())["signed_by"]))
    previews(args.output / "three_view_preview", args.data, meta, pq, scale, eval_rows,
             args.output / "linear_ap1_exr/images")
    write(args.output / "complete.json", dict(images=66, train=63, eval=3,
        canonical_contract=definition, records=records, time=time.time(),
        no_training_launched=True, accepted_SDR_package_unchanged=True))


if __name__ == "__main__":
    main()
