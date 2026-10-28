"""Recompose ONE archived Luster hybrid frame without editing the campaign.

The generic fix is in layered_dof.py; this adapter reconstructs the original
floating-point foreground contribution and reads the archived alpha/plate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import cv2
import numpy as np
from layered_dof import compose_in_focus, extend_background_coc, linear_to_srgb


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archived_coc(campaign, index, files):
    """Reproduce the archived temporal depth filter, before changing layers."""
    params = json.loads(files["dof_params"].read_text())
    disparity = np.load(files["disparity"]).astype(np.float32)
    q = (disparity - params["focus"]) / max(params["span"], 1e-5)
    image = cv2.imread(str(files["raw"]))
    gray = cv2.cvtColor(cv2.resize(image, (375, 512)), cv2.COLOR_BGR2GRAY)
    accum, norm = q.copy(), np.ones_like(q)
    yy, xx = np.indices(q.shape, dtype=np.float32)
    total = len(json.loads((campaign / "plan.json").read_text())["records"])
    for neighbor in (index - 1, index + 1):
        if not 0 <= neighbor < total:
            continue
        paths = dict(raw=campaign / f"raw/{neighbor:05d}.png",
                     disparity=campaign / f"B/depth/{neighbor:05d}.npy",
                     params=campaign / f"B/frames/{neighbor:05d}.json")
        files.update({f"neighbor_{neighbor}_{k}": v for k, v in paths.items()})
        other = cv2.imread(str(paths["raw"]))
        other_gray = cv2.cvtColor(cv2.resize(other, (375, 512)), cv2.COLOR_BGR2GRAY)
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(
            gray, other_gray, None)
        flow = cv2.resize(flow, (q.shape[1], q.shape[0])) * 8
        p = json.loads(paths["params"].read_text())
        d = np.load(paths["disparity"]).astype(np.float32)
        neighbor_q = (d - p["focus"]) / max(p["span"], 1e-5)
        warped = cv2.remap(neighbor_q, xx + flow[..., 0], yy + flow[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        weight = .3 * np.clip(1 - np.abs(warped - q) / .10, 0, 1)
        accum += weight * warped
        norm += weight
    return np.clip((-(accum / norm) - .12) / .45, 0, 1) * 48


def run(campaign, index, out, background_coc_guard_pixels=0.):
    out.mkdir(parents=True, exist_ok=False)
    receipt_path = campaign / f"raw/{index:05d}.json"
    receipt = json.loads(receipt_path.read_text())
    job = campaign / f"jobs/{index:05d}_{receipt['frame']:06d}"
    camera = 1000 + index
    geometry = job / f"geometry/cam{camera}"
    sys.path.insert(0, str(campaign / "code"))
    from compose_arc import detail_guard

    files = dict(alpha=geometry / "alpha.npy", base=geometry / "rgb.npy",
                 candidate=job / f"candidates/contour_s2_d1/cam{camera}.npy",
                 confidence=job / f"candidates/contour_s2_d1/cam{camera}_confidence.npy",
                 old_background=job / "background.npy",
                 background=job / "background_filled.npy",
                 disparity=campaign / f"B/depth/{index:05d}.npy",
                 old_dof=campaign / f"B/frames/{index:05d}.png",
                 raw=campaign / f"raw/{index:05d}.png",
                 dof_params=campaign / f"B/frames/{index:05d}.json")
    original_alpha = np.load(files["alpha"])
    alpha = np.clip(original_alpha, 0, 1)
    base = np.load(files["base"])
    candidate = np.load(files["candidate"])
    confidence = np.load(files["confidence"])
    weight = detail_guard(base, candidate)
    weight[confidence == 0] = 0
    background = np.load(files["background"])
    old_background = np.load(files["old_background"])
    delta = (candidate - base) * weight[..., None]
    legacy = base + delta
    legacy[confidence == 0] = base[confidence == 0]
    legacy += (1 - original_alpha[..., None]) * (background - old_background)
    foreground = base - (1 - alpha[..., None]) * old_background + delta
    raw = cv2.imread(str(files["raw"]))[:, :, ::-1]
    restored = np.uint8(np.rint(np.clip(
        foreground + (1 - alpha[..., None]) * background, 0, 1) * 255))
    np.testing.assert_array_equal(np.uint8(np.rint(np.clip(legacy, 0, 1) * 255)), raw)
    # Algebraically equivalent layer separation changes float32 association.
    # At quantization ties this can produce a single u8 code of difference.
    reconstruction_delta = np.abs(restored.astype(int) - raw.astype(int))
    assert reconstruction_delta.max() <= 1
    # Same focus/radius convention as the archived far-only DoF. Extend only
    # background CoC; never use the subject depth to defocus its backdrop.
    coc = archived_coc(campaign, index, files)
    background_coc = extend_background_coc(
        coc, alpha, background_guard_pixels=background_coc_guard_pixels)
    old = cv2.imread(str(files["old_dof"]))[:, :, ::-1]
    opaque = alpha >= 1 - 1e-6
    boundary = (alpha > .001) & (alpha < .999)
    metrics = {}
    for space in ("srgb", "linear"):
        result = compose_in_focus(foreground, alpha, background, background_coc,
                                  source_space="srgb", composite_space=space)
        if space == "linear":
            result = linear_to_srgb(result)
        result = np.uint8(np.rint(np.clip(result, 0, 1) * 255))
        cv2.imwrite(str(out / f"layers_{space}.png"), result[:, :, ::-1],
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        changed = np.any(result != old, axis=-1)
        cv2.imwrite(str(out / f"changed_{space}.png"), changed.astype(np.uint8) * 255)
        np.testing.assert_array_equal(result[~changed], old[~changed])
        difference = np.max(np.abs(result.astype(int) - raw.astype(int)), axis=-1)
        old_difference = np.max(np.abs(old.astype(int) - raw.astype(int)), axis=-1)
        metrics[space] = dict(
            opaque_max_u8_delta_from_raw=int(difference[opaque].max()),
            old_opaque_max_u8_delta_from_raw=int(old_difference[opaque].max()),
            boundary_pixels=int(boundary.sum()), changed_pixels=int(changed.sum()),
            unchanged_outside_saved_change_mask=True,
            alpha_unchanged=True, source_geometry_unchanged=True)
    inputs = {name: dict(path=str(path), sha256=sha(path)) for name, path in files.items()}
    metadata = dict(index=index, source_frame=receipt["frame"], inputs=inputs,
                    background_coc_guard_pixels=background_coc_guard_pixels,
                    old_composite_reconstruction_exact=True, metrics=metrics,
                    separated_layers_roundoff_max_u8=int(reconstruction_delta.max()),
                    separated_layers_roundoff_channels=int(np.count_nonzero(reconstruction_delta)),
                    code_sha256={p.name: sha(p) for p in [Path(__file__),
                        Path(__file__).with_name("layered_dof.py"),
                        campaign / "code/compose_arc.py"]},
                    caveats=["Requires in-focus foreground and a valid clean plate/alpha",
                             "Does not repair incorrect geometry or opaque floaters",
                             "Background CoC extrapolation is not metric depth recovery",
                             "No GT at virtual camera; no image quality metric is claimed"])
    (out / "receipt.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--background-coc-guard-pixels", type=float, default=0.,
                        help="Exclude uncertain depth observations near alpha support; leaves alpha unchanged")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    run(args.campaign, args.index, args.out, args.background_coc_guard_pixels)
