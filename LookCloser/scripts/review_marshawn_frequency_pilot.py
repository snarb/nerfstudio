"""Review completed source-bound frequency fits, retaining censoring separately.

This CPU report compares map labels on original TRAIN actor patches. Neither
the full-image censor rate nor map disagreement measures 3D reconstruction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    if args.output.exists():
        raise ValueError('Preserve prior report generation')
    variants = ('baseline', 'repeat', 'lr_decay', 'threshold093')
    for camera in ('AA', 'HA', 'MA'):
        if not all((args.fits / (camera+'_'+v) / 'complete.json').exists() for v in variants):
            raise ValueError('All four fits must finish for '+camera)
    ready = json.loads((args.data / 'ready.json').read_text())
    records = {r['camera']: r for r in ready['records']}
    rows, comparisons, inputs = [], [], {str(args.data / 'ready.json'): sha(args.data / 'ready.json')}
    args.output.mkdir(parents=True)
    # Same discrete palette for every variant/camera, never individually scaled.
    from matplotlib import colormaps
    palette = (colormaps['viridis'](np.arange(16)/15)[:, :3]*255+.5).astype(np.uint8)
    for camera in ('AA', 'HA', 'MA'):
        rec = records[camera]
        source, semantic = args.data / rec['image'], args.data / 'semantics' / (camera+'.npz')
        if sha(source) != rec['rgb_sha256'] or sha(semantic) != rec['semantic_sha256']:
            raise ValueError('Original TRAIN source changed')
        inputs[str(source)], inputs[str(semantic)] = sha(source), sha(semantic)
        maps, censors, receipts = {}, {}, {}
        for variant in variants:
            folder = args.fits / (camera+'_'+variant)
            for name in ('frequency.pt', 'censored.npy', 'complete.json'):
                inputs[str(folder/name)] = sha(folder/name)
            receipt = json.loads((folder/'complete.json').read_text())
            if receipt['source_sha256'] != sha(source) or receipt['map_sha256'] != sha(folder/'frequency.pt'):
                raise ValueError('Fit/source binding changed')
            frequency = torch.load(folder/'frequency.pt', map_location='cpu', weights_only=True).numpy()
            levels_float = np.log(frequency/16)/np.log(8192/16)*15
            if not np.isfinite(levels_float).all() or np.max(np.abs(levels_float-np.rint(levels_float))) > 1e-4:
                raise ValueError('Unexpected frequency schedule')
            maps[variant] = np.rint(levels_float).astype(np.int64)
            censors[variant] = np.load(folder/'censored.npy', allow_pickle=False)
            if censors[variant].dtype != np.bool_ or censors[variant].shape != frequency.shape:
                raise ValueError('Wrong censor mask')
            if int(censors[variant].sum()) != receipt['unresolved']:
                raise ValueError('Censor count changed')
            receipts[variant] = receipt
        height, width = maps['baseline'].shape
        with np.load(semantic, allow_pickle=False) as masks:
            for actor in ('black', 'grey'):
                patch_fraction = masks[actor][:height*8, :width*8].reshape(height, 8, width, 8).mean((1, 3))
                region = patch_fraction >= .9
                if not region.any():
                    raise ValueError('Empty actor patch region')
                for variant in variants:
                    labels = maps[variant][region]
                    rows.append(dict(camera=camera, actor=actor, variant=variant,
                        patches=int(region.sum()), censored_fraction=float(censors[variant][region].mean()),
                        mean_level=float(labels.mean()), histogram=np.bincount(labels, minlength=16).tolist()))
                    if variant != 'baseline':
                        delta = labels-maps['baseline'][region]
                        same_label = labels == maps['baseline'][region]
                        comparisons.append(dict(camera=camera, actor=actor, variant=variant,
                            map_changed_fraction=float((~same_label).mean()),
                            mean_absolute_level_change=float(np.abs(delta).mean()),
                            mean_signed_level_change=float(delta.mean()),
                            newly_resolved_fraction=float((censors['baseline'][region] & ~censors[variant][region]).mean()),
                            newly_resolved_same_label_fraction=float((censors['baseline'][region] & ~censors[variant][region] & same_label).mean())))
                gt = Image.open(source).convert('RGB')
                crop = rec['crops'][actor]
                tiles = [gt.crop(crop)]
                for variant in variants:
                    rgb = np.asarray(gt).copy()
                    colors = np.repeat(np.repeat(palette[maps[variant]], 8, axis=0), 8, axis=1)
                    rgb[:height*8, :width*8] = np.uint8(.55*rgb[:height*8, :width*8]+.45*colors+.5)
                    # Uncovered tail pixels retain the source; they are not assigned map labels.
                    tiles.append(Image.fromarray(rgb).crop(crop))
                w, h = tiles[0].size
                sheet = Image.new('RGB', (5*w, h+24), (28, 28, 28))
                draw = ImageDraw.Draw(sheet)
                for index, (tile, label) in enumerate(zip(tiles, ('GT',)+variants)):
                    sheet.paste(tile, (index*w, 24))
                    draw.text((index*w+3, 4), label, fill='white')
                sheet.thumbnail((1800, 1000))
                sheet.save(args.output/(camera+'_'+actor+'_levels.jpg'), quality=94)
        for variant in variants[1:]:
            for key in ('initial_parameters', 'first8_random_draws', 'final_rng', 'updates'):
                if receipts[variant][key] != receipts['baseline'][key]:
                    raise ValueError('Pilot execution contract differs: '+camera+' '+variant+' '+key)
    report = dict(kind='TRAIN_FREQUENCY_MAP_PILOT_REVIEW', rows=rows, comparisons=comparisons,
        inputs_sha256=inputs, helper_sha256=sha(__file__), model_loaded=False,
        outputs_sha256={p.name:sha(p) for p in args.output.glob('*.jpg')},
        three_dimensional_quality_measured=False,
        patch_policy='Original actor fraction >=.9 in complete8x8 patches; uncovered tail rows excluded.',
        limitation='Baseline repeat quantifies map variation. Resolving a formerly censored final-level patch may leave its saved label unchanged. No 3D recipe or map promotion.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(rows=len(rows), comparisons=len(comparisons), output=str(args.output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'fits', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    main(parser.parse_args())
