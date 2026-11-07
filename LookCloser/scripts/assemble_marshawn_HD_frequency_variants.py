"""CPU-only, isolated HD map assemblies and source comparisons; never train."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image, ImageDraw
import torch
from prepare_marshawn_tuning import refresh_inventory, verify_inventory


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(a):
    if a.output.exists():
        raise ValueError('Preserve previous assembly generation')
    final = json.loads((a.activation / 'final_validation.json').read_text())
    assert final['passed'] and final['jobs'] == 126 and final['cameras'] == 63
    ready = verify_inventory(a.data)
    before = sha(a.data / 'ready.json')
    parent = json.loads((a.recipe / 'spec.json').read_text())
    assert parent['coarse_step'] == .0015625 and parent['distortion_final'] == .1
    assert parent['objective'] == 'actor' and parent['width'] == 64
    transforms = json.loads((a.data / 'transforms.json').read_text())
    records = {r['camera']: r for r in ready['records']}
    train = {Path(p).stem.removeprefix('train_') for p in transforms['train_filenames']}
    assert len(train) == 63 and not train.intersection({'DC', 'HC', 'KA'})
    assert len(a.cameras) == len(set(a.cameras)) == 6 and set(a.cameras).issubset(train)
    a.output.mkdir(parents=True)
    rows = {(r['camera'], r['variant']): r for r in final['rows']}
    bindings = {}
    for row in final['rows']:
        for original, digest in row['files_sha256'].items():
            path = Path(original)
            if row['host'] == 'silver':
                relative = original.split('/host/jobs/', 1)[1]
                path = a.activation / 'results_silver' / relative
            if sha(path) != digest:
                raise ValueError('Validated fit copy changed: ' + str(path))
            bindings[str(path)] = digest
    assemblies = {}
    for variant in ('lr_decay', 'threshold093'):
        data = a.output / variant / 'data'
        data.mkdir(parents=True)
        for name in ('images', 'masks', 'semantics'):
            (data / name).symlink_to(os.path.relpath(a.data / name, data), target_is_directory=True)
        for name in ('transforms.json', 'box.json'):
            shutil.copy2(a.data / name, data / name)
        maps = data / 'lookcloser_frequencies'
        maps.mkdir()
        # Preserve any pre-existing non-TRAIN maps exactly; never fit EVAL.
        baseline_maps = a.data / 'lookcloser_frequencies'
        for path in baseline_maps.iterdir():
            if path.is_file():
                shutil.copy2(path, maps / path.name)
        derivations = []
        for camera in sorted(train):
            row = rows[camera, variant]
            name = row['name']
            base = a.activation / ('host/jobs' if row['host'] == 'local' else 'results_silver')
            folder = base / name / 'run' / (camera + '_' + variant)
            source = folder / 'frequency.pt'
            dest = maps / ('train_' + camera + '.pt')
            shutil.copy2(source, dest)
            metadata = json.loads((folder / 'frequency.json').read_text())
            metadata.update(source_image_sha256=records[camera]['rgb_sha256'],
                frequency_map_sha256=sha(dest), ssim_threshold=.95 if variant == 'lr_decay' else .93,
                train_steps_per_level=1000, eval_patch_batch_size=1024,
                fit_script_sha256='13a3d723cb5462eeadead9df5862cd57eba37c5b351a5e9d417f0a5be524d91e',
                preprocess_source_sha256='729f10f26b24b97f56adcc3c62a6722a78c4ee5d59d5b0b8dd963fde6561ffad',
                assembly_source_metadata_sha256=sha(folder / 'frequency.json'),
                assembly_source_completion_sha256=sha(folder / 'complete.json'),
                censoring='Exact separate per-patch censored.npy retained; scalar level15 alone is ambiguous')
            write(dest.with_suffix('.json'), metadata)
            shutil.copy2(folder / 'censored.npy', maps / ('train_' + camera + '_censored.npy'))
            derivations.append(dict(camera=camera, source=str(source), map_sha256=sha(dest),
                                    metadata_sha256=sha(dest.with_suffix('.json')),
                                    censor_sha256=sha(maps / ('train_' + camera + '_censored.npy'))))
        new_ready = copy.deepcopy(ready)
        new_ready.pop('inventory')
        new_ready['frequency_parent_ready_sha256'] = before
        new_ready['frequency_activation_validation_sha256'] = sha(a.activation / 'final_validation.json')
        new_ready['training_authorized'] = False
        write(data / 'ready.json', new_ready)
        sealed = refresh_inventory(data)
        changed = sorted(k for k, h in ready['inventory']['files'].items()
                         if sealed['inventory']['files'].get(k) != h)
        expected = {f'lookcloser_frequencies/train_{c}{suffix}' for c in train for suffix in ('.pt', '.json')}
        assert set(changed).issubset(expected)
        assert all(k.startswith('lookcloser_frequencies/') for k in changed)
        extra = {p.name: sha(p) for p in maps.iterdir()
                 if p.stem.removeprefix('train_') not in train and not p.name.endswith('_censored.npy')}
        for name, h in extra.items():
            assert sha(baseline_maps / name) == h
        spec = copy.deepcopy(parent)
        spec['data'] = str(data)
        spec['output'] = str(a.output / variant / 'proposed_run')
        spec['frequency_threshold'] = .95 if variant == 'lr_decay' else .93
        write(a.output / variant / 'proposed_spec.json', spec)
        assemblies[variant] = dict(data=str(data), ready_sha256=sha(data / 'ready.json'),
            map_fit_factor=dict(ssim_threshold=.95 if variant == 'lr_decay' else .93,
                                LR_schedule='exponential .01 to .001 per level, Adam retained' if variant == 'lr_decay' else 'constant .01',
                                updates_per_level=1000, levels=16),
            changed_existing_files=changed, added_censor_files=63, map_derivations=derivations,
            nonTRAIN_maps_preserved=extra, EVAL_fit_performed=False,
            source_grid_manifest_sha256=parent['semantic_admission']['manifest_sha256'],
            proposed_spec_sha256=sha(a.output / variant / 'proposed_spec.json'),
            runtime_patch_required='Exact expected frequency_threshold .95 to .93 validator only' if variant == 'threshold093' else 'None for map factor',
            staging_policy='Mirror this assembly beside the existing hash-identical campaign data; relative RGB/mask/semantic links remain portable. Remap only proposed spec data/output paths to host campaign root.',
            training_authorized=False)
    visual = a.output / 'review'
    visual.mkdir()
    from matplotlib import colormaps
    palette = np.uint8(colormaps['viridis'](np.arange(16) / 15)[:, :3] * 255 + .5)
    statistics = []
    for camera in sorted(train):
        rec = records[camera]
        maps = {}
        for variant in ('baseline', 'lr_decay', 'threshold093'):
            data = a.data if variant == 'baseline' else a.output / variant / 'data'
            values = torch.load(data / 'lookcloser_frequencies' / ('train_' + camera + '.pt'), map_location='cpu', weights_only=True).numpy()
            level = np.log(values / 16) / np.log(8192 / 16) * 15
            assert values.shape == (126, 240) and np.isfinite(level).all()
            assert np.max(abs(level - np.rint(level))) < 1e-4
            maps[variant] = np.rint(level).astype(np.int16)
        with np.load(a.data / 'semantics' / (camera + '.npz'), allow_pickle=False) as masks:
            for actor in ('black', 'grey'):
                mask = masks[actor][:1008].reshape(126, 8, 240, 8).mean((1, 3)) >= .9
                assert mask.any()
                for variant, level in maps.items():
                    selected = level[mask]
                    r = dict(camera=camera, actor=actor, variant=variant, patches=int(mask.sum()),
                             mean_level=float(selected.mean()), maxlevel_fraction=float((selected == 15).mean()),
                             changed_from_baseline=float((level[mask] != maps['baseline'][mask]).mean()))
                    if variant != 'baseline':
                        censor = np.load(a.output / variant / 'data/lookcloser_frequencies' / ('train_' + camera + '_censored.npy'))
                        r['censored_fraction'] = float(censor[mask].mean())
                    else:
                        r['censored_fraction'] = None
                    statistics.append(r)
        if camera not in a.cameras:
            continue
        rgb = np.asarray(Image.open(a.data / rec['image']).convert('RGB'))
        crops = list(rec['crops'].values())
        crop = (max(0, min(c[0] for c in crops) - 24), max(0, min(c[1] for c in crops) - 24),
                min(1920, max(c[2] for c in crops) + 24), min(1012, max(c[3] for c in crops) + 24))
        images = [Image.fromarray(rgb)]
        for level in maps.values():
            colored = np.repeat(np.repeat(palette[level], 8, 0), 8, 1)
            overlay = rgb.copy()
            overlay[:1008] = np.uint8(.55 * rgb[:1008] + .45 * colored + .5)
            images.append(Image.fromarray(overlay))
        width, height = crop[2] - crop[0], crop[3] - crop[1]
        panel = Image.new('RGB', (width * 4, height + 26), '#181818')
        draw = ImageDraw.Draw(panel)
        for i, (im, label) in enumerate(zip(images, ['SOURCE', 'frozen baseline', 'LR decay', 'SSIM .93'])):
            panel.paste(im.crop(crop), (i * width, 26));draw.text((i * width + 3, 5), camera + ' ' + label, fill='white')
        panel.save(visual / (camera + '_actors.png'))
        for name, lo, hi in [('upper', 0, .34), ('middle', .3, .7), ('lower', .66, 1)]:
            parts = [im.crop((crop[0], crop[1] + round(height * lo), crop[2], crop[1] + round(height * hi))) for im in images]
            out = Image.new('RGB', (width * 4, parts[0].height + 26), '#181818');d = ImageDraw.Draw(out)
            for i, (im, label) in enumerate(zip(parts, ['SOURCE', 'baseline', 'decay', '.93'])):
                out.paste(im, (i * width, 26));d.text((i * width + 3, 5), camera + ' ' + name + ' ' + label, fill='white')
            out.save(visual / (camera + '_' + name + '.png'))
    assert sha(a.data / 'ready.json') == before and not torch.cuda.is_initialized()
    verify_inventory(a.data)
    write(a.output / 'assembly.json', dict(schema=1, CPU_only=True, source_ready_sha256=before,
        activation_validation_sha256=sha(a.activation / 'final_validation.json'),
        source_recipe_spec_sha256=sha(a.recipe / 'spec.json'), assemblies=assemblies,
        trial_design='Two fresh cold quarter-step distortion.1 trials, same seed42, original confidence, W64pad1 admission/R0/body80/legacy projection and original LR/stopping schedule; frequency maps are the only treatment. No checkpoint continuation or fixed22k cap.',
        inputs_sha256=bindings, reviewed_cameras=a.cameras, actor_patch_statistics=statistics,
        helper_sha256=sha(__file__), visual_files_sha256={p.name:sha(p) for p in visual.glob('*.png')},
        training_authorized=False, promotion=False,
        limitations=['Baseline unresolved status unavailable; level15 is not equivalent to censored.',
                    'Same-camera new variants share fit initialization/RNG; frozen legacy baseline is not a repeat-matched fit.',
                    'Changing maps affects both FAS proposal and frequency field updates; this is one map-treatment factor with two consumers.',
                    'Source-aligned overlays are qualitative; no NeRF quality has been measured.']))
    print(json.dumps(dict(output=str(a.output), assembly_sha256=sha(a.output / 'assembly.json'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('activation', 'data', 'recipe', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--cameras', nargs='+', default=['AA', 'DA', 'GA', 'JA', 'MA', 'OA'])
    main(p.parse_args())
