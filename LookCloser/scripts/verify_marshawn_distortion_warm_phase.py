"""CPU verification of future zero and actual32+32 matched confidence gates."""
import argparse,json,os,sys
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def main(root,phase):
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(root/'code/LookCloser/scripts'),str(root/'code')]
    import torch
    from marshawn_distortion_warm_factory import ARMS
    from marshawn_distortion_warm_factory import STATE_KEYS
    from train_marshawn_semantic_late_distortion import tree_digest,verify_state_equality,adam_steps
    reports={};anchor=read(root/'anchor_audit.json')
    for arm in ARMS:
        out=root/'runs'/f'{phase}_{arm}';complete=read(out/'complete.json');assert complete['passed'];proof=read(out/'resume_proof_126316.json');assert proof['source_verified_before_confidence']and proof['FR']==.3 and proof['distortion']==.1
        for k in STATE_KEYS:assert proof[k]==anchor['fullstate_hashes'][k]
        if phase=='zero':
            assert complete['effective_distortion']==ARMS[arm] and proof['source_verified_before_loss_scalar']
            assert complete['status']=='zero_update_state_equal'and complete['render_state_unchanged'];reports[arm]=dict(source_seven_states_exact=True,render_parity=complete['render_parity'],cache_sha256=proof['cache_sha256'])
        else:
            first=torch.load(out/'checkpoints/step-000126348.ckpt',map_location='cpu',weights_only=False);final=torch.load(out/'checkpoints/step-000126380.ckpt',map_location='cpu',weights_only=False);resume=read(out/'resume_proof_126348.json')
            for k in STATE_KEYS:assert resume[k]==tree_digest(first[k])
            assert resume['confidence_state_sha256']==tree_digest(first['distortion_fork_runtime']['confidence_state']);assert resume['model_cache_sha256']==tree_digest(first['distortion_fork_runtime']['model_cache']);assert first['distortion_fork_runtime']['cache_state']['ids']==final['distortion_fork_runtime']['cache_state']['ids']
            middle=adam_steps(first['optimizers']);after=adam_steps(final['optimizers']);assert all(min(proof['Adam'][k])<min(middle[k])<min(after[k])for k in after);assert len(complete['gradient_checks'])==64 and complete['attempts']==64
            assert all(r['actual_distortion']==ARMS[arm] and r['actual_FR']==.3 and(not r['Adam_updated']or r['combined_gradients_finite'])for r in complete['gradient_checks'])
            for payload in (first,final):assert all(p['normalizer_error']<1e-12 for p in payload['distortion_fork_runtime']['sampling_segment']['packets'])
            if True:assert len(complete['source_control_checks'])==2 and all(r['source_W64_exact']and r['actual_distortion']==ARMS[arm] and r['actual_FR']==.3 and r['extra_optimizer_updates']==0 for r in complete['source_control_checks'])
            assert first['distortion_fork_runtime']['effective_distortion']==final['distortion_fork_runtime']['effective_distortion']==ARMS[arm]
            assert all(r['expected_distortion_factor']==ARMS[arm]/.1 and r['packed_distortion_gradient_max_error']<1e-5 for r in complete['source_control_checks'])
            reports[arm]=dict(effective_distortion=ARMS[arm],packed_factor_and_gradient_reference_pass=True,source_seven_states_exact=True,first32_indices=first['distortion_fork_runtime']['sampling_segment']['indices_sha256'],resume32_indices=final['distortion_fork_runtime']['sampling_segment']['indices_sha256'],first32_global_RNG=tree_digest(first['rng_state']),final_global_RNG=tree_digest(final['rng_state']),final_sampler=tree_digest(final['tuning_runtime']['sampling_state']),first_Adam=middle,final_Adam=after,confidence_and_cache_restored_exact=True)
            del first,final
    if phase!='technical':assert len({r['cache_sha256']for r in reports.values()})==1
    assert not torch.cuda.is_initialized();(root/(phase+'_verification.json')).write_text(json.dumps(dict(passed=True,phase=phase,arms=reports,CPU_only=True,technical_checkpoints_excluded_from_production=True,main_global_streams_observed_not_forced_equal=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['zero','technical'],required=True);a=p.parse_args();main(a.root,a.phase)
