"""Compare independently stopped terminals with the unchanged source/GT and ROIs."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    from PIL import Image, ImageDraw, ImageFont
    arms = ('legacy1024', 'corrected1024')
    base = root / 'external_guard_corrected_ARM_v1/visual_gates_v1'
    status = read(base / 'status.json')
    steps = {arm: status['states'][arm]['complete.json']['final_step'] for arm in arms}
    ready = root.parent / 'data/ready.json'
    rows = {row['camera']: row for row in read(ready)['records']}
    regions_path = root.parent / 'native_semantic_grid_20000/corrected_v2/review/region_crops.json'
    assert sha(regions_path) == '603991a50762581ce1f76912c2f6d8a7d768e77f3679f74787662db2e777a309'
    regions = read(regions_path)
    paths = {'source126316': root / 'anchor/renders'}
    receipts = {}
    for arm in arms:
        gate = base / 'retained_gates' / arm / f'step-{steps[arm]:09d}'
        receipt = read(gate / 'retention.json')
        for name, digest in receipt['files_sha256'].items():
            assert sha(gate / name) == digest
        receipts[arm] = sha(gate / 'retention.json')
        paths[arm] = gate / 'renders' / f'step-{steps[arm]:09d}'
    output = base / 'terminal_comparison'
    assert not output.exists(), 'Preserve previous terminal review generation'
    output.mkdir()
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
    actors, details, sources = [], [], {}
    for camera in ('DC', 'HC', 'KA'):
        inputs = [root.parent / 'data' / rows[camera]['image']] + [path / (camera + '.png') for path in paths.values()]
        images = [Image.open(path).convert('RGB') for path in inputs]
        assert all(image.size == (1920, 1012) for image in images)
        sources.update({str(path): sha(path) for path in inputs})
        crops = list(rows[camera]['crops'].values())
        bounds = [max(0, min(c[0] for c in crops) - 64), max(0, min(c[1] for c in crops) - 64),
                  min(1920, max(c[2] for c in crops) + 64), min(1012, max(c[3] for c in crops) + 64)]
        for name, crop in [('actors_margin64', bounds)] + list(regions[camera].items()):
            w, h = crop[2] - crop[0], crop[3] - crop[1]
            panel = Image.new('RGB', (4 * w, h + 38), (20, 20, 20))
            draw = ImageDraw.Draw(panel)
            labels = ['GT', 'source126316'] + [arm + ' @' + str(steps[arm]) for arm in arms]
            for i, (image, label) in enumerate(zip(images, labels)):
                panel.paste(image.crop(crop), (i * w, 38))
                draw.text((i * w + 3, 2), camera + ' ' + name, font=font, fill='white')
                draw.text((i * w + 3, 19), label, font=font, fill='white')
            panel.save(output / (camera + '_' + name + '.png'))
            panel.thumbnail((2200, 950 if name == 'actors_margin64' else 500))
            (actors if name == 'actors_margin64' else details).append(panel)
    for name, panels in (('all_actors', actors), ('all_details', details)):
        contact = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), (20, 20, 20))
        y = 0
        for panel in panels:
            contact.paste(panel, (0, y)); y += panel.height
        contact.save(output / (name + '.jpg'), quality=95)
    provenance = dict(steps=steps, metrics={name: {key: read(path / 'metrics.json')['metrics'][key]
        for key in ('psnr', 'ssim', 'lpips')} for name, path in paths.items()}, source_files_sha256=sources,
        retained_manifests_sha256=receipts, ready_sha256=sha(ready), fixed_regions_sha256=sha(regions_path),
        helper_sha256=sha(__file__), CPU_only=True, no_checkpoint_transfer=True, quality_promotion=False,
        comparison='Each arm at its own natural terminal; use common-step retained galleries for matched-step comparisons')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), steps=steps, metrics=provenance['metrics'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
