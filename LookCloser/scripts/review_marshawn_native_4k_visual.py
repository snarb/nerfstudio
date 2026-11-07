"""CPU-only, fixed-coordinate actual4K visual comparisons; no scoring or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    descriptor = json.loads(args.diagnostic.read_text())
    assert descriptor['training_eligible'] is False
    row = next(r for r in descriptor['frames'] if r['physical_camera'] == args.camera)
    source = args.diagnostic.parent / row['file_path']
    paths = [source] + args.prediction
    labels = ['Actual4K SOURCE'] + args.label
    assert len(labels) == len(paths), 'One label per prediction is required'
    images = [Image.open(path).convert('RGB') for path in paths]
    assert all(im.size == (3840, 2024) for im in images), 'Predictions must be actual4K renders'
    regions = json.loads(args.regions.read_text())[args.camera]
    args.output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    panels = []
    crops = {}
    for name, original in regions.items():
        crop = tuple(2 * int(v) for v in original)
        crops[name] = crop
        width, height = crop[2] - crop[0], crop[3] - crop[1]
        panel = Image.new('RGB', (width * len(images), height + 48), '#181818')
        draw = ImageDraw.Draw(panel)
        for index, (im, label) in enumerate(zip(images, labels)):
            panel.paste(im.crop(crop), (index * width, 48))
            draw.text((index * width + 5, 3), args.camera + ' ' + name, fill='white', font=font)
            draw.text((index * width + 5, 25), label, fill='white', font=font)
        panel.save(args.output / (args.camera + '_' + name + '.png'))
        panel.thumbnail((1800, 600))
        panels.append(panel)
    overview = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), '#181818')
    top = 0
    for panel in panels:
        overview.paste(panel, (0, top))
        top += panel.height
    overview.save(args.output / 'all_details.jpg', quality=95)
    receipt = dict(camera=args.camera, supplementary_visual_only=True,
                   primary_metric_and_selection_unchanged=True, no_training_or_model_loaded=True,
                   source_and_predictions=[dict(path=str(path), sha256=digest(path), label=label)
                                           for path, label in zip(paths, labels)],
                   actual_image_size=[3840, 2024], crops=crops,
                   crop_rule='Exactly twice the existing fixed FHD review coordinates; no refitting',
                   descriptor_sha256=digest(args.diagnostic), region_file_sha256=digest(args.regions),
                   helper_sha256=digest(Path(__file__)))
    (args.output / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--regions', type=Path, required=True)
    parser.add_argument('--camera', choices=['HC', 'DC', 'KA'], required=True)
    parser.add_argument('--prediction', type=Path, action='append', required=True)
    parser.add_argument('--label', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
