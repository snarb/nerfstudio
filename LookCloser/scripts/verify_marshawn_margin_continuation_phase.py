"""CPU-only validation of zero and 32+32 paired margin checkpoints."""
import argparse,json,os,sys
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())

def verify_cross_arm(reports,phase):
    key='first_batch_indices'if phase=='technical'else'cache_sha256'
    if len({r[key]for r in reports.values()})!=1:raise ValueError('Source first batch/cache identity differs')
    # Native density divergence can alter later RNG consumption; record, never
    # reseed or impose false cross-arm equality after the first actual batch.
    return dict(first_batch_or_cache_equal=True,later_cross_arm_RNG_equality_required=False)


def main(root,phase):
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(root/'code/LookCloser/scripts'),str(root/'code')]
    import torch
    from marshawn_margin_continuation_policy import ARMS
    from marshawn_margin_continuation_factory import STATE_KEYS
    from train_marshawn_semantic_late_distortion import tree_digest,adam_steps,sha
    anchor=read(root/'anchor_audit.json');reports={};inputs={}
    for arm in ARMS:
        out=root/'runs'/f'{phase}_{arm}';complete=read(out/'complete.json');assert complete['passed'];proof=read(out/'resume_proof_126316.json')
        assert proof['source_verified_before_selector']and proof['FR']==.3 and proof['distortion']==.1
        for key in STATE_KEYS:assert proof[key]==anchor['fullstate_hashes'][key]
        for name in ['source_baseline_parity.json','branch_zero_parity.json','density_gate_proof.json']:
            path=out/name;record=read(path);assert record['passed'];inputs[str(path)]=sha(path)
        branch=read(out/'branch_zero_parity.json');assert branch['arm']==arm and branch['selector_state']['binding']['arm']==arm
        if phase=='zero':
            assert complete['status']=='zero_update_state_equal'and complete['render_state_unchanged']and complete['sourceA_verified_before_gate']
            assert complete['own_branch_zero']==branch['own_evaluation']
            reports[arm]=dict(source_seven_states_exact=True,own_zero_metrics=complete['own_branch_zero'],render_parity=complete['render_parity'],cache_sha256=proof['cache_sha256'])
        else:
            firstpath=out/'checkpoints/step-000126348.ckpt';finalpath=out/'checkpoints/step-000126380.ckpt'
            first=torch.load(firstpath,map_location='cpu',weights_only=False);final=torch.load(finalpath,map_location='cpu',weights_only=False);resume=read(out/'resume_proof_126348.json')
            for key in STATE_KEYS:assert resume[key]==tree_digest(first[key])
            f=first['margin_fork_runtime'];g=final['margin_fork_runtime'];assert resume['selector_state_sha256']==tree_digest(f['selector_state']);assert resume['model_cache_sha256']==tree_digest(f['model_cache'])
            assert f['cache_state']['ids']==g['cache_state']['ids']and f['cache_state']['sources']==g['cache_state']['sources']
            assert f['evaluations'][0]==branch['own_evaluation']and g['evaluations'][0]==branch['own_evaluation']
            assert f['selector_state']['binding']==g['selector_state']['binding']and g['selector_state']['binding']['arm']==arm
            middle=adam_steps(first['optimizers']);after=adam_steps(final['optimizers']);assert all(min(proof['Adam'][k])<min(middle[k])<min(after[k])for k in after)
            assert len(complete['gradient_checks'])==64 and complete['attempts']==64
            for r in complete['gradient_checks']:assert r['actual_distortion']==.1 and r['actual_FR']==.3 and(not r['Adam_updated']or r['combined_gradients_finite'])
            for payload in [first,final]:
                assert payload['actual_occupancy_capture']['step']>126316
                assert torch.isfinite(payload['pipeline']['_model.occupancy_grid.occs']).all()
            for runtime in [f,g]:assert all(p['normalizer_error']<1e-12 for p in runtime['sampling_segment']['packets'])
            for step in [126348,126380]:assert read(out/f'technical_render_{step}.json')['finite_fullFHD']
            if arm=='original_mapped':assert len(complete['source_control_checks'])==2 and all(r['source_W64_exact']and r['extra_optimizer_updates']==0 for r in complete['source_control_checks'])
            for path in [firstpath,finalpath]:inputs[str(path)]=sha(path)
            reports[arm]=dict(source_seven_states_exact=True,first_batch_indices=f['sampling_segment']['first_indices_sha256'],first32_indices=f['sampling_segment']['indices_sha256'],resume32_indices=g['sampling_segment']['indices_sha256'],first32_RNG=tree_digest(first['rng_state']),final_RNG=tree_digest(final['rng_state']),first_Adam=middle,final_Adam=after,first_successful_updates=min(middle['fields'])-min(proof['Adam']['fields']),resume_successful_updates=min(after['fields'])-min(middle['fields']),selector_cache_LUT_private_sampling_restored_exact=True,own_initial_evaluation=True)
            del first,final
    cross=verify_cross_arm(reports,phase);assert not torch.cuda.is_initialized()
    result=dict(passed=True,phase=phase,arms=reports,cross_arm=cross,inputs_sha256=inputs,CPU_only=True,technical_checkpoints_excluded_from_production=True,quality_promotion=False)
    (root/(phase+'_verification.json')).write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['zero','technical'],required=True);a=p.parse_args();main(a.root,a.phase)
