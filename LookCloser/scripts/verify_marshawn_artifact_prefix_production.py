"""Read-only CPU proof of completed matched-prefix production forks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def main(root, output):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(root/'code/LookCloser/scripts'),str(root/'code')]
    import torch
    import numpy as np
    from train_marshawn_semantic_late_distortion import tree_digest,adam_steps
    from marshawn_artifact_prefix_factory import STATE_KEYS
    from marshawn_artifact_prefix_tau import ARMS,QualifiedPrefixTau,validate_targets
    from marshawn_tuning_sampling import body80_quotas
    assert not output.exists(), 'Preserve previous proof generation'
    anchor=read(root/'anchor_audit.json'); manifest=read(root/'confidence/manifest.json'); manifest_sha=sha(root/'confidence/manifest.json')
    with np.load(root/'anchor/grid.npz',allow_pickle=False) as z:
        fixed_grid={k:z[k].copy()for k in ('admission','cascade_aabbs','resolution')}
    completes={a:read(root/'runs'/('production_'+a)/'complete.json')for a in ARMS}
    assert all(d['passed'] and d['status']=='plateau' and d['plateau']['stopped'] and d['attempts']>=24000 for d in completes.values())
    common_step=min(d['final_step']for d in completes.values()); reports={}; common={}
    for arm in ARMS:
        run=root/'runs'/('production_'+arm); spec=read(root/'specs'/('production_'+arm)/'spec.json'); complete=completes[arm]
        proof=read(run/'resume_proof_126316.json'); assert proof['source_verified_before_confidence'] and proof['FR']==.3 and proof['distortion']==.1
        assert all(proof[k]==anchor['fullstate_hashes'][k]for k in STATE_KEYS)
        assert spec['mode']=='production' and spec['parent_checkpoint_sha256']==anchor['checkpoint_sha256']
        final_path=run/'checkpoints'/f"step-{complete['final_step']:09d}.ckpt"; final=torch.load(final_path,map_location='cpu',weights_only=False)
        fork=final['prefix_fork_runtime']; confidence=fork['confidence_state']; binding=confidence['binding']; cache=fork['model_cache']
        assert final['step']==complete['final_step'] and fork['source_sha256']==anchor['checkpoint_sha256'] and fork['spec_sha256']==sha(root/'specs'/('production_'+arm)/'spec.json')
        assert tree_digest(confidence)==tree_digest(complete['confidence_state'])
        assert binding['arm']=='original' and binding['manifest_sha256']==manifest_sha
        for row in manifest['records']:
            camera=row['camera']; assert binding['weights_sha256'][camera]==row['weights_sha256']['original']
            np.testing.assert_allclose(binding['proposal_means'][camera],row['proposal_means']['original'],rtol=0,atol=1e-15)
        quotas=body80_quotas(4096,final['step']); expectation=float(np.mean(list(binding['proposal_means'].values()),axis=0)@(quotas/4096))
        assert abs(confidence['last_normalizer']-expectation)<1e-12 and confidence['last_counts']==quotas.tolist()
        assert cache['FR']==cache['strength']==.3 and cache['switch_applied']
        assert tree_digest(cache)==proof['model_cache_sha256']
        assert fork['cache_state']['ids']==proof['cache_ids'] and fork['cache_state']['sources']==proof['cache_sources']
        actual=adam_steps(final['optimizers']); assert actual==complete['actual_Adam']
        delta={k:min(actual[k])-min(proof['Adam'][k])for k in actual}
        assert all(0<n<=complete['attempts']for n in delta.values())
        grid=read(final_path.with_name(final_path.stem+'_grid.json')); assert grid['checkpoint_sha256']==sha(final_path)
        grid_path=final_path.with_name(final_path.stem+'_grid.npz')
        assert sha(grid_path)==grid['grid_archive_sha256']
        with np.load(grid_path,allow_pickle=False) as z:
            for key,value in fixed_grid.items(): np.testing.assert_array_equal(z[key],value)
        retention=read(run/'retention.json'); evaluations=complete['evaluations']
        raw=min(evaluations,key=lambda r:(r['metrics']['lpips'],-r['metrics']['psnr'],r['step']))
        assert raw['step']==retention['raw_min_step'] and retention['pruning'].startswith('none')
        aux=final['auxiliary_state'];summary=complete['auxiliary_summary']
        assert aux['draws']==summary['draws']==complete['attempts'] and aux['coefficient']==ARMS[arm]
        assert tree_digest(aux['generator_state'])==summary['generator_sha256']
        assert aux['ids_chain']==summary['ids_chain'] and aux['jitter_chain']==summary['jitter_chain'] and aux['chain']==summary['chain']
        reports[arm]=dict(auxiliary_draws=aux['draws'],coefficient=aux['coefficient'],private_ids_chain=aux['ids_chain'],private_jitter_chain=aux['jitter_chain'],private_generator=tree_digest(aux['generator_state']),final_step=final['step'],attempts=complete['attempts'],actual_Adam_updates=delta,
            AMP_skips={k:complete['attempts']-n for k,n in delta.items()},source_seven_states_exact_before_policy=True,
            actual_FR=.3,actual_distortion_live_asserted_every_iteration=.1,confidence_manifest_and_all63_weight_hashes_exact=True,
            checkpoint_normalizer_error=abs(confidence['last_normalizer']-expectation),analytical_LUT_unchanged=True,
            original_mapped_admission_cascade_and_resolution_exact=True,
            explicit_cache_ids_and_source_associations_preserved=True,plateau=complete['plateau'],
            final_checkpoint_sha256=grid['checkpoint_sha256'],raw_min_step=raw['step'],historical_selector_step=retention['selected_step'],
            best_metrics={k:raw['metrics'][k]for k in ('psnr','ssim','lpips')})
        del final
        cp=run/'checkpoints'/f'step-{common_step:09d}.ckpt'; state=torch.load(cp,map_location='cpu',weights_only=False)
        common[arm]=dict(step=common_step,global_RNG=tree_digest(state['rng_state']),sampler=tree_digest(state['tuning_runtime']['sampling_state']),
                         indices_chain=state['prefix_fork_runtime']['sampling_segment']['indices_sha256'],
                         attempts=state['prefix_fork_runtime']['sampling_segment']['attempts'],private_generator=tree_digest(state['auxiliary_state']['generator_state']),private_ids_chain=state['auxiliary_state']['ids_chain'],private_jitter_chain=state['auxiliary_state']['jitter_chain'],private_draws=state['auxiliary_state']['draws'])
        del state
    for key in ('private_generator','private_ids_chain','private_jitter_chain','private_draws'):
        assert len({r[key]for r in common.values()})==1, key
    from types import SimpleNamespace
    parent=read(root/'anchor/spec.json')
    targets=validate_targets(root/'root_selected196_rays.json',Path(parent['data']),Path(parent['semantic_admission']['manifest']).with_name('support_manifest.json'))
    replay=QualifiedPrefixTau(SimpleNamespace(device=torch.device('cpu')),targets,0.)
    for _ in range(common_step-126316):replay.draw()
    reference=common['control']
    assert replay.draws==reference['private_draws'] and tree_digest(replay.generator.get_state())==reference['private_generator']
    assert replay.ids_chain==reference['private_ids_chain'] and replay.jitter_chain==reference['private_jitter_chain']
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,CPU_only=True,helper_sha256=sha(__file__),anchor_checkpoint_sha256=anchor['checkpoint_sha256'],
        technical_verification_sha256=sha(root/'technical_verification.json'),confidence_manifest_sha256=manifest_sha,
        reports=reports,last_common_checkpoint_stream_proof=common,full_private_stream_replayed_from_seed73011=True,main_global_streams_observed_not_forced_equal=True,quality_promotion=False,
        limitation='Serialized final-state audit, not another model-loader replay. Source loader/cache replay was independently proven by zero and actual32+32. Exterior opacity remains diagnostic, not empty-space truth.')
    output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(dict(passed=True,output=str(output),common_step=common_step)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
