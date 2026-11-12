"""Isolated native-BC MatAnyone2 test; never modifies source frames or training."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion


def panel(images, labels, size=(500, 470)):
    canvas = Image.new("RGB", (size[0] * len(images), size[1] + 24))
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        canvas.paste(Image.fromarray(im).convert("RGB").resize(size), (i * size[0], 24))
        draw.text((i * size[0] + 8, 5), label, fill="white")
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--first", type=int, default=1251)
    p.add_argument("--last", type=int, default=1267)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--prompt", choices=["official_close10", "soft", "corrected_close10"], default="official_close10")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for directory in ["alpha", "black", "baseline_black", "comparison"]:
        (args.out / directory).mkdir(exist_ok=True)
    torch.set_num_threads(8)
    started = time.perf_counter()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model = get_matanyone2_model(str(args.checkpoint), device)
    processor = InferenceCore(model, cfg=model.cfg, device=device)
    first_mask = np.asarray(Image.open(args.source / f"{args.first:06}_alpha.png").convert("L"))
    Image.fromarray(first_mask).save(args.out / "original_sam_prompt.png")
    if args.prompt in ["official_close10", "corrected_close10"]:
        first_mask = gen_erosion(gen_dilate(first_mask, 10, 10), 10, 10)
    if args.prompt == "corrected_close10":
        unedited = first_mask.copy()
        # Tiny grey core in the first-frame gap, specified in native BC crop pixels.
        # Do not trace the entire thin diagonal structure: it may include real hair.
        polygon = [[1317, 546], [1320, 548], [1323, 553], [1319, 551]]
        correction = np.zeros(first_mask.shape, np.uint8)
        cv2.fillPoly(correction, [np.array(polygon, np.int32)], 255)
        first_mask = first_mask.copy()
        first_mask[correction > 0] = 0
        Image.fromarray(unedited.astype(np.uint8)).save(args.out / "unedited_close10_prompt.png")
        difference = np.abs(first_mask.astype(np.int16) - unedited.astype(np.int16))
        Image.fromarray(difference.astype(np.uint8)).save(args.out / "prompt_diff.png")
        (args.out / "correction.json").write_text(json.dumps({"polygon_xy": polygon,
            "changed_pixels": int(np.count_nonzero(first_mask != unedited)),
            "coordinate_space": "native BC crop 2120x1450, frame1251", "value": 0}, indent=2))
    Image.fromarray(first_mask.astype(np.uint8)).save(args.out / "first_prompt.png")
    mask = torch.from_numpy(first_mask.copy()).float().to(device)
    rows = []
    roi = (1030, 380, 1530, 850)
    x0, y0, x1, y1 = roi
    video = imageio.get_writer(args.out / "hair_comparison_120fps.mp4", fps=120,
                               codec="libx264", quality=9, macro_block_size=2)
    slow = imageio.get_writer(args.out / "hair_comparison_12fps_review.mp4", fps=12,
                              codec="libx264", quality=9, macro_block_size=2)
    contact = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for ti, frame in enumerate([args.first] * args.warmup + list(range(args.first, args.last + 1))):
            tick = time.perf_counter()
            rgb = np.asarray(Image.open(args.source / f"{frame:06}_rgb.png").convert("RGB"))
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(device).float() / 255
            if ti == 0:
                processor.step(tensor, mask, objects=[1])
            prob = processor.step(tensor, first_frame_pred=ti <= args.warmup)
            alpha = processor.output_prob_to_mask(prob).float().cpu().numpy().clip(0, 1)
            torch.cuda.synchronize()
            duration = time.perf_counter() - tick
            if ti < args.warmup:
                print(json.dumps({"warmup": ti, "seconds": round(duration, 3)}), flush=True)
                continue
            base = np.asarray(Image.open(args.source / f"{frame:06}_alpha.png").convert("L"), dtype=np.float32) / 255
            black = np.rint(rgb * alpha[..., None]).astype(np.uint8)
            baseline = np.rint(rgb * base[..., None]).astype(np.uint8)
            alpha8 = np.rint(alpha * 255).astype(np.uint8)
            Image.fromarray(alpha8).save(args.out / "alpha" / f"{frame:06}.png")
            Image.fromarray(black).save(args.out / "black" / f"{frame:06}.png")
            Image.fromarray(baseline).save(args.out / "baseline_black" / f"{frame:06}.png")
            images = [rgb, baseline, black, np.repeat(alpha8[..., None], 3, axis=2)]
            comparison = panel([im[y0:y1, x0:x1] for im in images],
                               [f"Source BC {frame}", "Original SAM soft alpha", "MatAnyone2 on black", "MatAnyone2 alpha"])
            comparison.save(args.out / "comparison" / f"{frame:06}.jpg", quality=95)
            video.append_data(np.asarray(comparison)); slow.append_data(np.asarray(comparison))
            if frame in [args.first, (args.first + args.last)//2, args.last]:
                contact.append(comparison.resize((1200, 296)))
            row = {"frame": frame, "inference_seconds": duration,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
            rows.append(row)
            print(json.dumps(row), flush=True)
    video.close(); slow.close()
    contact_image = Image.new("RGB", (1200, 296 * len(contact)))
    for i, im in enumerate(contact):
        contact_image.paste(im, (0, i * 296))
    contact_image.save(args.out / "temporal_contact.jpg", quality=95)
    receipt = {"model": "MatAnyone2", "source": str(args.source), "prompt": args.prompt,
               "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "first": args.first, "last": args.last, "fps": 120, "warmup": args.warmup,
               "native_size": list(rgb.shape[:2]), "roi_xyxy": roi,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
               "elapsed_seconds": time.perf_counter()-started, "frames": rows,
               "note": "SAM is a comparison baseline, not ground truth. No PSNR/SSIM/LPIPS quality claim."}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
