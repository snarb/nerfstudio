"""CPU-only proof of zero restoration or actual focus-refresh restart; no optimization."""
import argparse
import json
import os
from pathlib import Path
import sys


def read(path):return json.loads(Path(path).read_text())


def main(root,phase):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    code=root/'code';sys.path[:0]=[str(code/'LookCloser/scripts'),str(code)]
    import torch
    from train_marshawn_focused_clothing_geometry import tree_digest,verify_state_equality,adam_steps
    from marshawn_focused_clothing_geometry import ARMS
    assert read(root/(phase+'_guard/complete.json'))['passed']
    result={}
    for arm in ARMS:
        folder=root/'runs'/phase/arm;complete=read(folder/'complete.json');spec=read(root/'specs'/(phase+'_'+arm+'.json'))
        if phase=='zero':
            assert complete['status']=='zero_update_state_equal' and complete['proof']['auxiliary_draws']==0
            assert complete['proof']['focus_state_exact'] and complete['render_state_unchanged']
            result[arm]=dict(passed=True,source_seven_hashes={k:complete['proof'][k]for k in ['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture']},render_parity=complete['render_parity'])
        else:
            first=torch.load(folder/'checkpoints/step-000015364.ckpt',map_location='cpu',weights_only=False)
            final=torch.load(folder/'checkpoints/step-000015380.ckpt',map_location='cpu',weights_only=False)
            resume=read(folder/'resume_proof_15364.json');hashes=verify_state_equality(first,first)
            assert all(resume[k]==v for k,v in hashes.items())
            assert resume['auxiliary_state_sha256']==tree_digest(first['fork_runtime']['auxiliary_state'])
            assert resume['focus_state_sha256']==tree_digest(first['tuning_runtime']['frequency_focus_state'])
            assert resume['focus_state_exact'] and first['tuning_runtime']['frequency_focus_state']['last_step']==15360
            assert final['tuning_runtime']['frequency_focus_state']['last_step']==15360
            assert first['fork_runtime']['auxiliary_state']['stream']['draws']==176 and final['fork_runtime']['auxiliary_state']['stream']['draws']==192
            parent=torch.load(spec['parent_checkpoint'],map_location='cpu',weights_only=False)
            assert tree_digest(first['pipeline'])!=tree_digest(parent['pipeline']) and tree_digest(final['pipeline'])!=tree_digest(first['pipeline'])
            assert tree_digest(final['tuning_runtime']['frequency_focus_state'])==tree_digest(first['tuning_runtime']['frequency_focus_state'])
            del parent
            assert complete['attempts']==192 and complete['final_step']==15380
            before=read(folder/'resume_proof_15188.json')['Adam'];middle=adam_steps(first['optimizers']);after=adam_steps(final['optimizers'])
            assert all(min(before[k])<min(middle[k])<min(after[k])for k in before)
            assert len(complete['gradient_checks'])==192
            assert all(not r['Adam_updated']or r['combined_gradients_finite']for r in complete['gradient_checks'])
            result[arm]=dict(passed=True,resume_seven_hashes=hashes,first_Adam=middle,final_Adam=after,
                focus_refresh=15360,focus_private_state_exact_resume=True,geometry_private_state_exact_resume=True,
                auxiliary_chain=complete['auxiliary_chain'],auxiliary_draws=192,
                first_main_indices=first['fork_runtime']['sampling_segment']['first_preupdate_ray_indices_sha256'],
                first176_main_chain=first['fork_runtime']['sampling_segment']['ray_indices_sha256'],
                final16_main_chain=final['fork_runtime']['sampling_segment']['ray_indices_sha256'],
                final_global_rng=tree_digest(final['rng_state']),final_main_sampler=tree_digest(final['tuning_runtime']['sampling_state']))
    if phase=='technical':
        assert len({r['auxiliary_chain']for r in result.values()})==1
        assert len({r['first_main_indices']for r in result.values()})==1
        equality={key:len({r[key]for r in result.values()})==1 for key in ['first176_main_chain','final16_main_chain','final_global_rng','final_main_sampler']}
    else:equality={}
    assert not torch.cuda.is_initialized()
    report=dict(passed=True,phase=phase,arms=result,cross_arm_observed_equality=equality,no_CUDA_initialized=True,
                diagnostic_training_authorized=False,technical_checkpoints_excluded_from_production=True)
    path=root/(phase+'_verification.json')
    assert not path.exists(),'Preserve prior proof'
    path.write_text(json.dumps(report,indent=2)+'\n');print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['zero','technical'],required=True);a=p.parse_args();main(a.root,a.phase)
