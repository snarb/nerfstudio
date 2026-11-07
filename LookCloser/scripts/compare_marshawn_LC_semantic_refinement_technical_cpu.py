"""Measure technical CUDA restart drift; equality is claimed only at restore START."""
import argparse,json,os
from pathlib import Path
import torch
from run_marshawn_frozen_semantic_probe import nested_digest
from prepare_marshawn_frozen_semantic_cpu import sha,write
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(2)
report={}
for arm in ['coarse128','residual256']:
    paths=[a.root/'runs'/(arm+'_'+phase)for phase in ['technical32','technical_resume32','technical64']]
    completions=[json.loads((p/'complete.json').read_text())for p in paths]
    states=[torch.load(p/'head.pt',map_location='cpu',weights_only=False)for p in paths]
    for path,c,s in zip(paths,completions,states):
        assert sha(path/'head.pt')==c['checkpoint_sha256']and c['native_checkpoint_and_caches_unchanged']
        assert c['native_model_not_loaded_or_optimized']and c['qualified_UNKNOWN_count']==4
    assert all(completions[1]['restoration'].values())
    assert states[0]['iteration']==32 and states[1]['iteration']==states[2]['iteration']==64
    assert states[1]['camera_history']==states[2]['camera_history']
    assert nested_digest(states[1]['rng'])==nested_digest(states[2]['rng'])
    for path in [paths[0],paths[2]]:
        initial=json.loads((path/'initial_state.json').read_text());assert initial['initial_Adam_count']==initial['initial_iteration']==0
    drift={k:float((states[1]['head'][k]-states[2]['head'][k]).abs().max())for k in states[1]['head']}
    adam_drift={}
    for key in ['exp_avg','exp_avg_sq']:
        adam_drift[key]=max(float((states[1]['optimizer']['state'][i][key]-states[2]['optimizer']['state'][i][key]).abs().max())for i in states[1]['optimizer']['state'])
    if arm=='residual256':assert drift['coarse_logits']==0 and completions[1]['frozen_coarse_unchanged']and completions[2]['frozen_coarse_unchanged']
    report[arm]=dict(exact_resume_START_restoration=completions[1]['restoration'],head_maxabs_drift=drift,Adam_maxabs_drift=adam_drift,
        final_RNG_and_camera_history_exact=True,physical_UUIDs=[json.loads((p/'initial_state.json').read_text())['health']['actual_compute_UUID']for p in paths],
        checkpoints={str(p/'head.pt'):sha(p/'head.pt')for p in paths})
assert not torch.cuda.is_initialized()
write(a.root/'CPU_technical_comparison.json',dict(CPU_only=True,CUDA_initialized=False,arms=report,acceptance=False,
    caveat='ROOT must review measured atomic drift and resource/OOM evidence before explicit fresh2064 release.'))
print(json.dumps(report))
