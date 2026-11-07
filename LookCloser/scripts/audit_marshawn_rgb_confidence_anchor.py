"""CPU-only raw126316 anchor inventory, seven-state/schema2 and Adam ancestry."""
import argparse,hashlib,json,os,sys
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(root):
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(root/'code/LookCloser/scripts'),str(root/'code')]
    import torch,numpy as np
    from train_marshawn_semantic_late_distortion import tree_digest,adam_steps
    from marshawn_rgb_confidence_factory import SOURCE_SHA,SOURCE_STEP,STATE_KEYS
    a=root/'anchor';cp=a/'checkpoint.ckpt';assert sha(cp)==SOURCE_SHA;state=torch.load(cp,map_location='cpu',weights_only=False);assert state['step']==SOURCE_STEP
    grid=json.loads((a/'grid.json').read_text());capture=state['actual_occupancy_capture'];pipe=state['pipeline'];find=lambda suffix:next(v for k,v in pipe.items()if k.endswith(suffix));arr=lambda t:t.detach().cpu().numpy()
    with np.load(a/'grid.npz')as z:
        checks=dict(core_capture=np.array_equal(z['core'],arr(capture['core'])),active_capture=np.array_equal(z['active'],arr(capture['active'])),active_pipeline=np.array_equal(z['active'],arr(find('occupancy_grid.binaries'))),EMA_pipeline=np.array_equal(z['ema'].reshape(-1),arr(find('occupancy_grid.occs')).reshape(-1)),learned_core=np.array_equal(z['core'].reshape(-1),arr(find('semantic_learned_core')).reshape(-1)),active_is_core_and_admission=np.array_equal(z['active'],z['core']&z['admission']),semantic_receipt=grid['semantic_receipt']==capture['semantic_receipt'])
    assert all(checks.values())and grid['checkpoint_sha256']==SOURCE_SHA;adam=adam_steps(state['optimizers']);assert all(v and min(v)>0 for v in adam.values())
    source_files=[p for p in a.rglob('*')if p.is_file()and p.name!='audit.json'];runtime=state['tuning_runtime'];source_eval=json.loads((a/'renders/metrics.json').read_text());assert source_eval['step']==SOURCE_STEP
    result=dict(anchor_status='QUARTER_DISTORTION_RAWMIN_FULLSTATE',step=SOURCE_STEP,checkpoint_sha256=SOURCE_SHA,source_hashes={str(p):sha(p)for p in source_files},fullstate_hashes={k:tree_digest(state[k])for k in sorted(STATE_KEYS)},actual_Adam=adam,schedulers=state['schedulers'],scalers=state['scalers'],optimizer_groups={k:v['param_groups']for k,v in state['optimizers'].items()},sampling_state=runtime['sampling_state'],original_plateau_preserved_as_ancestry=runtime['plateau'],source_eval=source_eval,render_directory=str(a/'renders'),semantic_grid_checks=checks,semantic_capture_metadata={k:v for k,v in capture.items()if not torch.is_tensor(v)},source_pipeline_extra_state_sha256={k:tree_digest(v)for k,v in pipe.items()if k.endswith('_extra_state')},CUDA_initialized=torch.cuda.is_initialized(),model_instantiated=False,optimizer_instantiated=False,baseline_loader_resume_not_yet_run=True)
    assert not result['CUDA_initialized'];(root/'anchor_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('anchor seven states verified CPU')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
