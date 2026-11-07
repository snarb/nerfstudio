"""CPU verification of future zero and actual32+32 matched confidence gates."""
import argparse,json,os,sys
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def main(root,phase):
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(root/'code/LookCloser/scripts'),str(root/'code')]
    import torch
    from marshawn_artifact_prefix_tau import ARMS,QualifiedPrefixTau,validate_targets
    from marshawn_artifact_prefix_factory import STATE_KEYS
    from train_marshawn_semantic_late_distortion import tree_digest,verify_state_equality,adam_steps
    reports={};anchor=read(root/'anchor_audit.json')
    for arm in ARMS:
        out=root/'runs'/f'{phase}_{arm}';complete=read(out/'complete.json');assert complete['passed'];proof=read(out/'resume_proof_126316.json');assert proof['source_verified_before_confidence']and proof['FR']==.3 and proof['distortion']==.1
        for k in STATE_KEYS:assert proof[k]==anchor['fullstate_hashes'][k]
        if phase=='zero':
            assert complete['status']=='zero_update_state_equal'and complete['render_state_unchanged'];reports[arm]=dict(source_seven_states_exact=True,render_parity=complete['render_parity'],cache_sha256=proof['cache_sha256'])
        else:
            first=torch.load(out/'checkpoints/step-000126348.ckpt',map_location='cpu',weights_only=False);final=torch.load(out/'checkpoints/step-000126380.ckpt',map_location='cpu',weights_only=False);resume=read(out/'resume_proof_126348.json')
            for k in STATE_KEYS:assert resume[k]==tree_digest(first[k])
            assert resume['confidence_state_sha256']==tree_digest(first['prefix_fork_runtime']['confidence_state']);assert resume['model_cache_sha256']==tree_digest(first['prefix_fork_runtime']['model_cache']);assert first['prefix_fork_runtime']['cache_state']['ids']==final['prefix_fork_runtime']['cache_state']['ids']
            middle=adam_steps(first['optimizers']);after=adam_steps(final['optimizers']);assert all(min(proof['Adam'][k])<min(middle[k])<min(after[k])for k in after);assert len(complete['gradient_checks'])==64 and complete['attempts']==64
            assert all(r['actual_distortion']==.1 and r['actual_FR']==.3 and(not r['Adam_updated']or r['combined_gradients_finite'])for r in complete['gradient_checks'])
            for payload in (first,final):assert all(p['normalizer_error']<1e-12 for p in payload['prefix_fork_runtime']['sampling_segment']['packets'])
            if True:assert len(complete['source_control_checks'])==2 and all(r['source_W64_exact']and r['actual_distortion']==.1 and r['actual_FR']==.3 and r['extra_optimizer_updates']==0 for r in complete['source_control_checks'])
            assert resume['auxiliary_state_sha256']==tree_digest(first['auxiliary_state'])
            from types import SimpleNamespace
            target=validate_targets(root/'root_selected196_rays.json',Path(read(root/'anchor/spec.json')['data']),Path(read(root/'anchor/spec.json')['semantic_admission']['manifest']).with_name('support_manifest.json'))
            aux=QualifiedPrefixTau(SimpleNamespace(device=torch.device('cpu')),target,ARMS[arm]);aux.load_state_dict(first['auxiliary_state'])
            for _ in range(32):aux.draw()
            assert tree_digest(aux.state_dict())==tree_digest(final['auxiliary_state'])
            assert final['auxiliary_state']['draws']==64
            assert all(r['auxiliary_draws']==i+1 for i,r in enumerate(complete['gradient_checks']))
            if arm=='control':assert all(r['auxiliary_last']['density_query_skipped']for r in complete['gradient_checks'])
            else:assert all(r['auxiliary_last']['mean_tau']>=0 for r in complete['gradient_checks'])
            reports[arm]=dict(private_generator=tree_digest(final['auxiliary_state']['generator_state']),private_ids_chain=final['auxiliary_state']['ids_chain'],private_jitter_chain=final['auxiliary_state']['jitter_chain'],private_draws=64,private_resume_exact=True,source_seven_states_exact=True,first32_indices=first['prefix_fork_runtime']['sampling_segment']['indices_sha256'],resume32_indices=final['prefix_fork_runtime']['sampling_segment']['indices_sha256'],first32_global_RNG=tree_digest(first['rng_state']),final_global_RNG=tree_digest(final['rng_state']),final_sampler=tree_digest(final['tuning_runtime']['sampling_state']),first_Adam=middle,final_Adam=after,confidence_and_cache_restored_exact=True)
            del first,final
    if phase=='technical':
        for key in ('private_generator','private_ids_chain','private_jitter_chain','private_draws'):assert len({r[key]for r in reports.values()})==1,key
    else:assert len({r['cache_sha256']for r in reports.values()})==1
    assert not torch.cuda.is_initialized();(root/(phase+'_verification.json')).write_text(json.dumps(dict(passed=True,phase=phase,arms=reports,CPU_only=True,technical_checkpoints_excluded_from_production=True,main_global_streams_observed_not_forced_equal=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['zero','technical'],required=True);a=p.parse_args();main(a.root,a.phase)
