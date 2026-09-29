"""Collect paired, inspectable ablation evidence without making visual decisions."""
import argparse
import json
from pathlib import Path
from statistics import mean

from supervise_blur_campaign import changed_conditions


def compact(record):
    result = {'step': record['step'], 'eval_stride': record['eval_stride'],
              'full': {k: record['eval_all_' + k] for k in ('psnr', 'ssim', 'lpips')}}
    for split in ('train', 'eval'):
        views = [v for v in record['per_view'] if v['split'] == split]
        regions = sorted({r for v in views for r in v['rois']})
        result[split + '_rois'] = {
            r: {k: mean(v['rois'][r][k] for v in views if r in v['rois'])
                for k in ('psnr', 'ssim', 'lpips')} for r in regions}
        details = [v['rois'][r] for v in views for r in ('face', 'hair', 'lipstick')
                   if r in v['rois']]
        if details:
            result[split + '_detail'] = {
                k: mean(v[k] for v in details) for k in ('psnr', 'ssim', 'lpips')}
    return result


def read_run(folder):
    history = [compact(r) for r in json.loads((folder / 'history.json').read_text())]
    selection = json.loads((folder / 'selection.json').read_text())
    step = int(Path(selection['path']).stem.split('_')[-1])
    return dict(complete=(folder / 'complete.json').exists(), history=history,
                selected=next(r for r in history if r['step'] == step),
                identity=json.loads((folder / 'identity.json').read_text()),
                first_batches=json.loads((folder / 'first_batches.json').read_text()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--requests', type=Path, default=Path(__file__).resolve().parents[1] /
                   'experiments/assets/blur_ablation_fresh')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runs = {d.name: read_run(d) for d in args.root.iterdir()
            if (d / 'history.json').exists() and (d / 'selection.json').exists()}
    pairs = []
    for path in sorted(args.requests.glob('*.json')):
        request = json.loads(path.read_text())
        if not isinstance(request, dict) or not request.get('comparison_parent'):
            continue
        parent = request['comparison_parent']
        changes = changed_conditions(json.loads(path.with_name(parent + '.json').read_text()), request)
        if len(changes) != 1:
            raise ValueError(f'{path.name}: expected one changed condition, got {changes}')
        pair = dict(child=path.stem, parent=parent, changed_condition=changes[0])
        if path.stem in runs and parent in runs:
            a, b = runs[parent], runs[path.stem]
            pair['both_complete'] = a['complete'] and b['complete']
            pair['same_initial_parameters'] = a['identity']['initial_weights'] == b['identity']['initial_weights']
            pair['same_first_32_batches'] = a['first_batches'] == b['first_batches']
            common = sorted({r['step'] for r in a['history']} & {r['step'] for r in b['history']})
            pair['matched'] = [dict(step=step,
                parent=next(r for r in a['history'] if r['step'] == step),
                child=next(r for r in b['history'] if r['step'] == step)) for step in common]
            pair['selected'] = dict(parent=a['selected'], child=b['selected'])
        pairs.append(pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(runs=runs, pairs=pairs), indent=2) + '\n')
    print(f'{len(runs)} runs, {len(pairs)} one-condition comparisons; saved {args.output}')


if __name__ == '__main__':
    main()
