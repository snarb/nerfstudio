"""Frame-local cold training around the sealed Marshawn winner.

The worker, model, sampler and training loop live in campaign/code. This module
only binds a global frame ID to its own inputs/output and records provenance.
"""
from pathlib import Path
import argparse, hashlib, importlib, json, os, re, sys, time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def read(path): return json.loads(Path(path).read_text())


def owned(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if root != path and root not in path.parents: raise ValueError('Foreign frame path: ' + str(path))
    return path


def verify_code(campaign):
    campaign = Path(campaign).resolve(); seal = read(campaign / 'code_seal.json')
    if sha(campaign / 'recipe.json') != seal['recipe_sha256'] or sha(campaign / 'factory_proof.json') != seal['factory_proof_sha256']:
        raise ValueError('Frozen recipe or factory proof changed')
    for relative, digest in seal['files_sha256'].items():
        if sha(owned(campaign / 'code', campaign / 'code' / relative)) != digest:
            raise ValueError('Frozen campaign code changed: ' + relative)
    actual={str(p.relative_to(campaign/'code')) for p in (campaign/'code').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
    if actual != set(seal['files_sha256']):raise ValueError('Unsealed files were added to campaign code')
    return seal


def bind(campaign, frame, technical=False, technical_options=None, technical_label=None):
    campaign = Path(campaign).resolve(); frame_root = campaign / 'frames' / ('%06d' % frame)
    if technical_label and (not technical or not re.fullmatch('[A-Za-z0-9_]+',technical_label)):
        raise ValueError('Safe technical-only output label required')
    suffix=('_'+technical_label) if technical_label else ''
    data = frame_root / 'data'; output = frame_root / ('technical_train'+suffix if technical else 'train')
    owned(frame_root,output)
    seal = verify_code(campaign); ready = read(data / 'ready.json')
    if int(ready['frame']) != frame or ready['train'] != 66: raise ValueError('Wrong frame or ALL66 policy')
    frequency = read(data / 'frequency_ready.json')
    if frequency['ready_sha256'] != sha(data / 'ready.json'): raise ValueError('Unsealed frequency inventory')
    gate = data / 'admission' / 'manifest.json'
    source_spec = read(campaign / 'recipe.json')
    if technical_options:
        if not technical or set(technical_options)-{'technical_start_step','technical_pause_after','technical_memory_cap_gib'}:
            raise ValueError('Only explicit technical controls may alter a smoke spec')
        source_spec.update(technical_options)
    source_spec.update(data=str(data), output=str(output), semantic_admission=dict(
        manifest=str(gate), manifest_sha256=sha(gate), identity_control=False,
        release_receipt=str(campaign / 'release.json')))
    spec_path = frame_root / ('technical_spec'+suffix+'.json' if technical else 'spec.json')
    if spec_path.exists() and read(spec_path) != source_spec: raise ValueError('Frame specification changed')
    if not spec_path.exists(): atomic(spec_path, source_spec)
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(schema=1, frame=frame, data=str(data), output=str(output),
        data_ready_sha256=sha(data / 'ready.json'), spec_sha256=sha(spec_path),
        code_seal_sha256=sha(campaign / 'code_seal.json'), seed=source_spec['seed'],
        initialization='cold from scratch; resume only from this frame/output',
        learned_cross_frame_state=False, frequency_cross_frame_state=False,
        train_cameras=66, monitoring='DC/HC/KA in-sample')
    claim = output / 'cold_provenance.json'
    if claim.exists() and read(claim) != provenance: raise ValueError('Foreign or changed frame provenance')
    if not claim.exists():
        if (output / 'latest.json').exists() or list((output / 'checkpoints').glob('*.ckpt')):
            raise ValueError('Existing weights without frame-local cold provenance')
        atomic(claim, provenance)
    latest = output / 'latest.json'
    if latest.exists():
        checkpoint = owned(output / 'checkpoints', output / read(latest)['checkpoint'])
        if sha(checkpoint) != read(latest)['checkpoint_sha256']: raise ValueError('Same-frame resume hash differs')
    return spec_path


def verify(spec_path):
    """Production authorization is immutable campaign release, not old experiment gates."""
    spec_path = Path(spec_path).resolve(); campaign = spec_path.parents[2]
    release = read(campaign / 'release.json')
    if release.get('production_enabled') is not True or release.get('all66_training') is not True:
        raise ValueError('Campaign has not passed cold/resume qualification')
    if release['code_seal_sha256'] != sha(campaign / 'code_seal.json'): raise ValueError('Release code binding differs')
    verify_code(campaign)
    return release


def cold_setup_audit(trainer, output, resume):
    """Read-only assertion after native setup; does not draw RNG or alter tensors."""
    import torch
    grid = trainer.pipeline.model.occupancy_grid
    report = dict(cold=resume is None, resume=str(resume) if resume else None,
                  occupied_cells=int(torch.count_nonzero(grid.binaries)),
                  nonzero_ema_cells=int(torch.count_nonzero(grid.occs)),
                  optimizer_state_sizes={k:len(v.state) for k,v in trainer.optimizers.optimizers.items()},
                  trainer_start_step=int(trainer._start_step))
    if resume is None and (report['occupied_cells'] or report['nonzero_ema_cells'] or
                           any(report['optimizer_state_sizes'].values()) or report['trainer_start_step'] != 0):
        raise ValueError('Cold frame unexpectedly contains learned state')
    atomic(Path(output) / ('resume_setup_audit.json' if resume else 'cold_setup_audit.json'), report)


def selectors(evaluations):
    raw = min(evaluations, key=lambda r:(r['metrics']['lpips'], -r['metrics']['psnr'], r['step']))
    best_psnr = max(r['metrics']['psnr'] for r in evaluations)
    psnr = min((r for r in evaluations if r['metrics']['psnr'] >= best_psnr-.07),
               key=lambda r:(r['metrics']['lpips'], -r['metrics']['psnr'], r['step']))
    return {'actor_lpips':raw, 'actor_psnr07':psnr}


def completion(campaign, frame, technical=False, technical_label=None):
    frame_root = Path(campaign).resolve() / 'frames' / ('%06d' % frame)
    suffix=('_'+technical_label) if technical_label else ''
    output = frame_root / ('technical_train'+suffix if technical else 'train')
    final = read(output / 'complete.json'); result = read(output / 'result.json')
    latest = read(output / 'latest.json')
    import torch
    checkpoint = owned(output / 'checkpoints', output / latest['checkpoint'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['tuning_runtime']
    choices = selectors(state['evaluations']); choices['historical'] = final['selection']
    exported = {}
    for key, row in choices.items():
        path = output / 'checkpoints' / ('step-%09d.ckpt' % row['step'])
        exported[key] = dict(step=row['step'], checkpoint=str(path), checkpoint_sha256=sha(path),
            metrics={k:v for k,v in row['metrics'].items() if k in ('psnr','ssim','lpips')},
            render_directory=str(output / 'renders' / ('step-%09d' % row['step'])))
    receipt = dict(schema=1, frame=frame, status='complete', technical=technical,
        final_step=result['final_step'], selectors=exported,
        config=read(output / 'runtime.json')['config'],
        code=str(Path(campaign).resolve() / 'code'),
        cold_provenance_sha256=sha(output / 'cold_provenance.json'),
        monitoring='DC/HC/KA in-sample; all66 train',
        pipeline_loader=dict(module='marshawn_video_train', function='load_pipeline',
                             args=dict(frame_root=str(frame_root), checkpoint_kind='actor_lpips')))
    atomic(frame_root / ('technical_completion'+suffix+'.json' if technical else 'frame_completion.json'), receipt)
    return receipt


def load_pipeline(frame_root, checkpoint_kind='actor_lpips', device='cuda:0'):
    """Return (pipeline, binding), loading the exact selected checkpoint."""
    root = Path(frame_root).resolve(); receipt = read(root / 'frame_completion.json')
    verify_code(root.parents[1])
    code = Path(receipt['code']); sys.path[:0] = [str(code), str(code / 'LookCloser/scripts')]
    choice = receipt['selectors'][checkpoint_kind]
    if sha(choice['checkpoint']) != choice['checkpoint_sha256']: raise ValueError('Render checkpoint changed')
    from nerfstudio.utils.eval_utils import eval_setup
    def selected(config):
        config.eval_checkpoint = Path(choice['checkpoint']); return config
    result = eval_setup(Path(receipt['config']), test_mode='inference', update_config_callback=selected)
    if str(result[1].device) != str(device): result[1].to(device)
    return result[1], dict(frame=receipt['frame'], checkpoint_kind=checkpoint_kind,
        checkpoint=str(result[2]), checkpoint_sha256=choice['checkpoint_sha256'],
        step=result[3], config=receipt['config'], code=receipt['code'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True); parser.add_argument('--frame',type=int,required=True)
    parser.add_argument('--smoke-steps',type=int); parser.add_argument('--bind-only',action='store_true')
    parser.add_argument('--technical-label');parser.add_argument('--technical-start-step',type=int)
    parser.add_argument('--technical-pause-after',type=int);parser.add_argument('--technical-memory-cap-gib',type=float)
    args=parser.parse_args(); technical=args.smoke_steps is not None
    controls={key:value for key,value in vars(args).items() if key in ('technical_start_step','technical_pause_after','technical_memory_cap_gib') and value is not None}
    spec=bind(args.campaign,args.frame,technical,controls,args.technical_label)
    if args.bind_only: print(spec); return
    code=args.campaign.resolve()/'code'; sys.path[:0]=[str(code),str(code/'LookCloser/scripts')]
    worker=importlib.import_module('train_marshawn_semantic_admission')
    worker.train(spec,args.smoke_steps)
    completion(args.campaign,args.frame,technical,args.technical_label)


if __name__=='__main__':main()
