"""CPU visual review of atomically retained gates against unchanged source/parent."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sheet(panels, destination):
    out = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), (20, 20, 20))
    y = 0
    for panel in panels: out.paste(panel, (0, y)); y += panel.height
    out.save(destination, quality=95)


def main(root, step):
    campaign = root.parent; guard = root / 'external_guard_v3'; output = guard / 'reviews' / f'step-{step:09d}'
    output.mkdir(parents=True, exist_ok=False)
    ready = campaign / 'data/ready.json'; rows = {r['camera']: r for r in json.loads(ready.read_text())['records']}
    region_file = campaign / 'native_semantic_grid_20000/corrected_v2/review/region_crops.json'
    regions = json.loads(region_file.read_text()); arms = ['control', 'old7369', 'seven10481']
    labels = ['SOURCE', 'parent91128'] + arms; font = ImageFont.truetype('DejaVuSans.ttf', 14)
    metrics, sources, bindings = {}, {}, {}
    for arm in arms:
        gate = guard / 'retained_gates' / arm / f'step-{step:09d}'
        receipt = json.loads((gate / 'retention.json').read_text())
        for rel, expected in receipt['files_sha256'].items():
            if sha(gate / rel) != expected: raise ValueError('Retained artifact changed: ' + rel)
        bindings[arm] = sha(gate / 'retention.json')
        metrics[arm] = json.loads((gate / 'renders' / f'step-{step:09d}' / 'metrics.json').read_text())
    details, actors = [], []
    for camera in ['DC', 'HC', 'KA']:
        paths = [campaign / 'data' / rows[camera]['image'], root / 'anchor/renders' / (camera + '.png')]
        paths += [guard / 'retained_gates' / arm / f'step-{step:09d}' / 'renders' / f'step-{step:09d}' / (camera + '.png') for arm in arms]
        images = [Image.open(p).convert('RGB') for p in paths]
        if any(im.size != (1920, 1012) for im in images): raise ValueError('Unexpected FHD raster')
        sources.update({str(p): sha(p) for p in paths})
        crops = rows[camera]['crops']; values = list(crops.values())
        actor_crop = [max(0, min(v[0] for v in values)-64), max(0, min(v[1] for v in values)-64),
                      min(1920, max(v[2] for v in values)+64), min(1012, max(v[3] for v in values)+64)]
        for name, crop in [('actors_margin64', actor_crop)] + list(regions[camera].items()):
            w, h = crop[2]-crop[0], crop[3]-crop[1]
            panel = Image.new('RGB', (w*5, h+38), (20, 20, 20)); draw = ImageDraw.Draw(panel)
            for i, (im, label) in enumerate(zip(images, labels)):
                panel.paste(im.crop(crop), (i*w, 38)); draw.text((i*w+3, 2), camera+' '+name, font=font, fill='white'); draw.text((i*w+3, 19), label, font=font, fill='white')
            panel.save(output / (camera+'_'+name+'.png'))
            if name == 'actors_margin64': panel.thumbnail((1800, 950)); actors.append(panel)
            else: panel.thumbnail((2200, 500)); details.append(panel)
    sheet(details, output / 'all_details.jpg'); sheet(actors, output / 'all_actors.jpg')
    receipt = dict(step=step, source_files_sha256=sources, retained_manifests_sha256=bindings,
                   ready_sha256=sha(ready), fixed_detail_regions_sha256=sha(region_file), helper_sha256=sha(__file__),
                   primary_metrics={arm: {k: metrics[arm]['metrics'][k] for k in ['psnr','ssim','lpips']} for arm in arms},
                   actor_visual_crop='union of original fixed actor crops plus64px clipped to FHD viewport',
                   GPU_used=False, fitting_or_source_mutation=False, promotion=False)
    (output / 'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), metrics=receipt['primary_metrics'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True); p.add_argument('--step', type=int, required=True)
    a = p.parse_args(); main(a.root, a.step)
