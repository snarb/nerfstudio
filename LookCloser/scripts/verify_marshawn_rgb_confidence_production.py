"""Read-only CPU proof of completed matched-confidence production forks."""
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
    from marshawn_rgb_confidence_factory import STATE_KEYS
    from marshawn_rgb_confidence_policy import ARMS
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
        fork=final['confidence_fork_runtime']; confidence=fork['confidence_state']; binding=confidence['binding']; cache=fork['model_cache']
        assert final['step']==complete['final_step'] and fork['source_sha256']==anchor['checkpoint_sha256'] and fork['spec_sha256']==sha(root/'specs'/('production_'+arm)/'spec.json')
        assert tree_digest(confidence)==tree_digest(complete['confidence_state'])
        assert binding['arm']==arm and binding['manifest_sha256']==manifest_sha
        for row in manifest['records']:
            camera=row['camera']; assert binding['weights_sha256'][camera]==row['weights_sha256'][arm]
            np.testing.assert_allclose(binding['proposal_means'][camera],row['proposal_means'][arm],rtol=0,atol=1e-15)
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
        reports[arm]=dict(final_step=final['step'],attempts=complete['attempts'],actual_Adam_updates=delta,
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
                         indices_chain=state['confidence_fork_runtime']['sampling_segment']['indices_sha256'],
                         attempts=state['confidence_fork_runtime']['sampling_segment']['attempts'])
        del state
    for key in ('global_RNG','sampler','indices_chain','attempts'):
        assert len({r[key]for r in common.values()})==1, key
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,CPU_only=True,helper_sha256=sha(__file__),anchor_checkpoint_sha256=anchor['checkpoint_sha256'],
        technical_verification_sha256=sha(root/'technical_verification.json'),confidence_manifest_sha256=manifest_sha,
        reports=reports,last_common_checkpoint_stream_proof=common,quality_promotion=False,
        limitation='Serialized final-state audit, not another model-loader replay. Source loader/cache replay was independently proven by zero and actual32+32. Exterior opacity remains diagnostic, not empty-space truth.')
    output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(dict(passed=True,output=str(output),common_step=common_step)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
