"""CPU visual comparison of immutable source and two native resolution zero gates."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    root = args.technical
    campaign = root.parent
    source = json.loads((root / "selected_parent.json").read_text())
    parity = json.loads((root / "parity.json").read_text())
    phases = ["HD_identity_zero", "TRAIN4K_zero"]
    assert all(phase + "_FHD" in parity for phase in phases)
    ready = campaign / "data/ready.json"
    records = {row["camera"]: row for row in json.loads(ready.read_text())["records"]}
    region_path = campaign / "native_semantic_grid_20000/corrected_v2/review/region_crops.json"
    regions = json.loads(region_path.read_text())
    output = root / "zero_update_visual_review"
    output.mkdir(exist_ok=False)
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
    panels, files = [], []
    for camera in ["DC", "HC", "KA"]:
        paths = [campaign / "data" / records[camera]["image"],
                 Path(source["renders"]) / camera / (camera + "_inside.png")]
        paths += [root / (phase + "_FHD_evaluation") / "renders" / camera /
                  (camera + "_inside.png") for phase in phases]
        images = [Image.open(path).convert("RGB") for path in paths]
        assert all(image.size == (1920, 1012) for image in images)
        files.extend({"path": str(path), "sha256": sha(path)} for path in paths)
        labels = ["SOURCE", "selected native " + str(source["step"]),
                  "HD adapter zero", "4K TRAIN adapter zero"]
        for name, crop in regions[camera].items():
            crops = [image.crop(crop) for image in images]
            width, height = crops[0].size
            panel = Image.new("RGB", (4 * width, height + 37), (20, 20, 20))
            draw = ImageDraw.Draw(panel)
            for index, (image, label) in enumerate(zip(crops, labels)):
                panel.paste(image, (index * width, 37))
                draw.text((index * width + 3, 2), camera + " " + name, font=font, fill="white")
                draw.text((index * width + 3, 18), label, font=font, fill="white")
            panel.save(output / (camera + "_" + name + ".png"))
            panel.thumbnail((1800, 450))
            panels.append(panel)
    sheet = Image.new("RGB", (max(panel.width for panel in panels),
                              sum(panel.height for panel in panels)), (20, 20, 20))
    offset = 0
    for panel in panels:
        sheet.paste(panel, (0, offset))
        offset += panel.height
    sheet.save(output / "all_details.jpg", quality=95)
    (output / "provenance.json").write_text(json.dumps({
        "source_step": source["step"], "inputs": files,
        "region_sha256": sha(region_path), "parity_sha256": sha(root / "parity.json"),
        "helper_sha256": sha(__file__), "no_GPU_or_model_loaded": True,
        "visual_inspection_pending": True,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technical", type=Path, required=True)
    main(parser.parse_args())
