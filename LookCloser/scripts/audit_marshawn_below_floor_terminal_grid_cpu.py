"""Read actual saved occupancy; EMA and admission are not physical geometry proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(root, source, cells, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    sys.path[:0] = [str(root.parent / 'distortion03_warm_continuation_candidate_v1/code/LookCloser/scripts')]
    from train_marshawn_semantic_late_distortion import tree_digest

    arms = ['production_control0', 'production_prior1e4', 'production_prior1e3']
    records = {}
    inventory = {}
    states = {}
    src = dict(np.load(source, allow_pickle=False))
    boxes = src['cascade_aabbs'].reshape(-1, 2, 3)
    assert len(boxes) == 1 and src['core'].shape == (1, 128, 128, 128)
    lo, hi = boxes[0]
    pitch = (hi - lo) / 128
    zlo = lo[2] + np.arange(128) * pitch[2]
    # Intersection includes cells crossing the cutoff; fully below is separate.
    intersect = np.broadcast_to((zlo < -.05)[None, None, None, :], src['core'].shape)
    fully = np.broadcast_to((zlo + pitch[2] <= -.05)[None, None, None, :], src['core'].shape)
    island = json.loads(Path(cells).read_text())
    if isinstance(island, dict):
        island = island['cells']
    indices = [tuple(x['cell']) for x in island]
    assert len(indices) == 7

    def describe(core, active, ema):
        result = {}
        for name, region in [('whole_grid', np.ones(core.shape, bool)),
                             ('intersects_below_minus005', intersect),
                             ('fully_below_minus005', fully),
                             ('source_active_below_minus005', intersect & src['active'])]:
            v = ema[region]
            result[name] = dict(total_cells=int(region.sum()), core=int((core & region).sum()),
                active=int((active & region).sum()),
                EMA_diagnostic_quantiles=dict(zip(['min', 'median', 'p95', 'max'],
                    [float(x) for x in np.quantile(v, [0, .5, .95, 1])])) if v.size else None)
        result['seven_island_cells'] = [dict(cell=list(i), core=bool(core[(0,) + i]),
            active=bool(active[(0,) + i]), EMA_diagnostic=float(ema[(0,) + i])) for i in indices]
        return result

    source_state = {k: src[k].copy() for k in ['core', 'active', 'ema']}
    records['source126316'] = dict(source_grid_sha256=sha(source),
        description=describe(source_state['core'], source_state['active'], source_state['ema']))
    for arm in arms:
        run = root / 'runs' / arm
        complete = json.loads((run / 'complete.json').read_text())
        assert complete['passed'], arm
        latest = json.loads((run / 'latest.json').read_text())
        # Actual terminal step, never select by quality or reinterpret source metadata.
        step = int(complete['final_step'])
        cp = run / 'checkpoints' / f'step-{step:09d}.ckpt'
        digest = sha(cp)
        p = torch.load(cp, map_location='cpu', weights_only=False)
        assert int(p['step']) == step
        capture, pipe = p['actual_occupancy_capture'], p['pipeline']
        def array(v):
            return v.detach().cpu().numpy().copy()
        core = array(capture['core'])
        active = array(pipe['_model.occupancy_grid.binaries'])
        ema = array(pipe['_model.occupancy_grid.occs']).reshape(core.shape)
        assert capture['radius'] == 0 and not capture['binary_warmup']
        assert np.array_equal(active, array(capture['active']))
        assert np.array_equal(active, array(pipe['_model.adaptive_sampler.occupancy_grid.binaries']))
        assert np.array_equal(core.reshape(-1), array(pipe['_model.semantic_learned_core']).reshape(-1))
        assert np.array_equal(array(pipe['_model.occupancy_grid.aabbs']).reshape(-1, 2, 3), boxes)
        assert np.isfinite(ema).all()
        assert np.array_equal(core & src['admission'], active), 'Original R0 admission policy'
        records[arm] = dict(step=step, attempts=int(complete['attempts']),
            complete_sha256=sha(run / 'complete.json'), checkpoint_sha256=digest,
            core_update_step=int(capture['step']), latest_metadata=latest,
            description=describe(core, active, ema),
            source_core_removed_below=int((source_state['core'] & ~core & intersect).sum()),
            source_core_added_below=int((~source_state['core'] & core & intersect).sum()))
        config = run / 'worker/lookcloser/frozen/config.yml'
        inventory[arm] = dict(checkpoint=str(cp), checkpoint_sha256=digest, step=step,
            pipeline_digest=tree_digest(p['pipeline']), config=str(config), config_sha256=sha(config),
            completion=str(run / 'complete.json'), completion_sha256=sha(run / 'complete.json'))
        states[arm] = dict(core=core, active=active, ema=ema)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / (arm + '_actual_grid.npz'), core=core, active=active,
            ema=ema, cascade_aabbs=boxes, checkpoint_step=step)
        del p, pipe, capture
    cp = root / 'anchor/checkpoint.ckpt'
    p = torch.load(cp, map_location='cpu', weights_only=False)
    config = root.parent / 'below_floor_density_warm_technical_plan_v1/runs/zero_source/worker/lookcloser/frozen/config.yml'
    inventory['source126316'] = dict(checkpoint=str(cp), checkpoint_sha256=sha(cp), step=int(p['step']),
        pipeline_digest=tree_digest(p['pipeline']), config=str(config), config_sha256=sha(config))
    assert inventory['source126316']['checkpoint_sha256'] == '340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
    del p
    control = states[arms[0]]
    for arm in arms[1:]:
        v = states[arm]
        records[arm]['terminal_vs_control'] = dict(
            same_absolute_step=records[arm]['step'] == records[arms[0]]['step'],
            core_removed_below=int((control['core'] & ~v['core'] & intersect).sum()),
            core_added_below=int((~control['core'] & v['core'] & intersect).sum()),
            below_EMA_max_abs_diagnostic=float(np.abs(v['ema'][intersect].astype('float64') -
                control['ema'][intersect].astype('float64')).max()))
    report = dict(CPU_only=True, CUDA_initialized=torch.cuda.is_initialized(), records=records,
        helper_sha256=sha(__file__),
        source_cells_sha256=sha(cells), below_cutoff=-.05,
        geometry_claim=False, quality_promotion=False,
        limitations=['Saved actual traversal masks only; no EMA threshold reconstruction.',
            'EMA, occupied or skipped cells do not certify actual density or free space.',
            'Terminals may have different steps; those comparisons are explicitly flagged.',
            'Measured numerical repeatability differences remain a competing explanation.'])
    assert not report['CUDA_initialized']
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'selected_checkpoint_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), arms={a:dict(step=records[a]['step'],
        below=records[a]['description']['intersects_below_minus005']) for a in arms})))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'source', 'cells', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    main(**vars(p.parse_args()))
