"""CPU source/100k/matched-HD/4K detail sheets; no model or GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    campaign = a.production.parent
    state = json.loads((a.production / 'state.json').read_text())
    ready = campaign / 'data/ready.json'
    rows = {r['camera']: r for r in json.loads(ready.read_text())['records']}
    regions_path = campaign / 'native_semantic_grid_20000/corrected_v2/review/region_crops.json'
    regions = json.loads(regions_path.read_text())
    arms = ['HD', 'TRAIN4K'] if a.arm == 'both' else [a.arm]
    baseline = state['arms']['HD']['evaluations'][0]
    assert baseline['step'] == 100000
    gates = [(name, next(r for r in state['arms'][name]['evaluations'] if r['step'] == a.step)) for name in arms]
    a.output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
    panels, sources = [], []
    for camera in ['DC', 'HC', 'KA']:
        paths = [campaign / 'data' / rows[camera]['image'], Path(baseline['renders']) / camera / (camera + '_inside.png')]
        paths += [Path(row['renders']) / camera / (camera + '_inside.png') for name, row in gates]
        labels = ['SOURCE', 'native100k'] + [name + str(a.step) for name, row in gates]
        images = [Image.open(path).convert('RGB') for path in paths]
        assert all(image.size == (1920, 1012) for image in images)
        sources.extend(dict(path=str(path), sha256=sha(path)) for path in paths)
        for region, crop in regions[camera].items():
            views = [image.crop(crop) for image in images]
            width, height = views[0].size
            panel = Image.new('RGB', (len(views) * width, height + 37), (20, 20, 20))
            draw = ImageDraw.Draw(panel)
            for i, (view, label) in enumerate(zip(views, labels)):
                panel.paste(view, (i * width, 37))
                draw.text((i * width + 3, 2), camera + ' ' + region, fill='white', font=font)
                draw.text((i * width + 3, 18), label, fill='white', font=font)
            panel.save(a.output / (camera + '_' + region + '.png'))
            panel.thumbnail((1800, 450))
            panels.append(panel)
    sheet = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), (20, 20, 20))
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(a.output / 'all_details.jpg', quality=95)
    receipt = dict(step=a.step, arms=arms, source_files=sources, ready_sha256=sha(ready),
                   region_crops_sha256=sha(regions_path), helper_sha256=sha(Path(__file__)),
                   evaluations={name: dict(path=row['evaluation'], sha256=sha(Path(row['evaluation']))) for name, row in gates},
                   no_GPU_or_model_loaded=True, primary_FHD=True)
    (a.output / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--arm', choices=['HD', 'TRAIN4K', 'both'], default='both')
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
