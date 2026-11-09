"""Post-gate paired-prior evidence; fixed selectors, source guards, no fitting."""
import argparse
import json
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha, write


def run(spec_path, execute):
    s = json.loads(spec_path.read_text())
    if not execute:
        print(json.dumps(dict(phase='paired_prior_CPU_review', spec_sha256=sha(spec_path), initial_probability=[.99, .01],
            optimizer_created=False, CUDA_started=False, automatic_promotion=False)))
        return
    release = json.loads(Path(s['root_release']).read_text())
    if release.get('reviewer') != 'root' or not release.get('paired_prior_comparison_authorized') or release['spec_sha256'].get(str(spec_path)) != sha(spec_path):
        raise ValueError('Exact root paired-prior post-gate comparison required')
    for p, digest in release['bound_files'].items():
        if sha(p) != digest: raise ValueError('Bound paired source differs: '+p)
    from PIL import Image, ImageDraw
    roots = [Path(s['prior99']), Path(s['prior01'])]
    render = [json.loads((p/'renders/complete.json').read_text()) for p in roots]
    metrics = [json.loads((p/'visual_gate/metrics.json').read_text()) for p in roots]
    distributions = [json.loads(Path(s[k]).read_text()) for k in ['prior99_TRAIN', 'prior01_TRAIN']]
    if render[0]['native_model_digest'] != render[1]['native_model_digest'] or metrics[0]['ready_sha256'] != metrics[1]['ready_sha256']:
        raise ValueError('Native field or fixed source targets changed across priors')
    probabilities = [r.get('initial_semantic_probability', r.get('initial_preservation_probability')) for r in render]
    if probabilities != [.99, .01]: raise ValueError('Wrong paired initial probabilities')
    if render[0]['physical_selectors'] != render[1]['physical_selectors']:
        raise ValueError('Physical selector thresholds changed')
    out = Path(s['output']); out.mkdir(parents=True, exist_ok=False)
    ready_path = Path(s['ready']); rows = {r['camera']: r for r in json.loads(ready_path.read_text())['records']}
    regions = json.loads(Path(s['regions']).read_text())
    detail_panels, body_panels, hidden_comparison = [], [], []
    for name in ['DC', 'HC', 'KA']:
        paths = [ready_path.parent/rows[name]['image']]
        paths += [root/'renders'/name/(variant+'.png') for variant in ['hard_theta05', 'soft_probability'] for root in roots]
        labels = ['SOURCE', 'p99_hard.5', 'p01_hard.5', 'p99_soft', 'p01_soft']
        images = [Image.open(p).convert('RGB') for p in paths]
        for region, crop in regions[name].items():
            tiles = [im.crop(crop) for im in images]; w, h = tiles[0].size
            panel = Image.new('RGB', (5*w, h+35), '#171717'); draw = ImageDraw.Draw(panel)
            for i, (tile, label) in enumerate(zip(tiles, labels)):
                panel.paste(tile, (i*w, 35)); draw.text((i*w+2, 2), name+' '+region, fill='white'); draw.text((i*w+2, 17), label, fill='white')
            panel.save(out/(name+'_'+region+'.png')); panel.thumbnail((1800, 430)); detail_panels.append(panel)
        crops = list(rows[name]['crops'].values()); crop = [max(0, min(v[0] for v in crops)-48), max(0, min(v[1] for v in crops)-32), min(1920, max(v[2] for v in crops)+48), min(1012, max(v[3] for v in crops)+48)]
        panel = Image.new('RGB', (5*330, 560), '#171717'); draw = ImageDraw.Draw(panel)
        for i, (im, label) in enumerate(zip(images, labels)):
            tile = im.crop(crop); tile.thumbnail((325, 520)); panel.paste(tile, (i*330+(325-tile.width)//2, 35)); draw.text((i*330+2, 3), name+' '+label, fill='white')
        body_panels.append(panel)
        source_rows = [next(r for r in doc['records'] if r['camera'] == name) for doc in render]
        hidden_comparison.append(dict(camera=name,
            prior99=source_rows[0]['hidden_native_zero_weight'], prior01=source_rows[1]['hidden_native_zero_weight'],
            newly_visible_mass_pixels_gt001={str(p): row['newly_visible_mass_pixels_gt001'] for p, row in zip(probabilities, source_rows)}))
    for filename, panels in [('all_actors_prior_pair.jpg', body_panels), ('all_details_prior_pair.jpg', detail_panels)]:
        canvas = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), '#171717'); y = 0
        for panel in panels: canvas.paste(panel, (0, y)); y += panel.height
        canvas.save(out/filename, quality=94)
    write(out/'comparison.json', dict(initial_probability=probabilities, native_field_same=True, fixed_eval_sources_same=True,
        physical_selectors_same=True, metric_comparison={str(p): {k: v['metrics'] for k, v in doc['variants'].items()} for p, doc in zip(probabilities, metrics)},
        TRAIN_class_distributions={str(p): doc['groups'] for p, doc in zip(probabilities, distributions)},
        hidden_comparison=hidden_comparison, CPU_only=True, optimizer_created=False, automatically_promoted_selector=None,
        interpretation='Initialp.01 explicitly removes unsupported regions under hardselectors; it is a modelprior, not evidence that hidden space is physically empty. Judge positive/hair/contact/feet preservation and hidden-mass changes. No threshold chosen automatically or from EVALalone.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); run(a.spec, a.execute)
