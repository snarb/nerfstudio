"""Read actual LC technical checkpoints: Adam/RNG trajectory and head drift."""
import argparse
import json
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha,write


def run(fit,output):
    import numpy as np
    import torch
    from run_marshawn_frozen_semantic_probe import nested_digest
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=False)
    names=['technical32','technical_resume32','technical64']
    paths=[fit/'runs'/name/'head.pt'for name in names]
    saved=[torch.load(p,map_location='cpu',weights_only=False)for p in paths]
    a,b,c=saved
    if [r['iteration']for r in saved]!=[32,64,64]:raise ValueError('Actual reviewed update counts required')
    if len({r['source_kind']for r in saved})!=1 or a['source_kind']!='DIRECT_FROZEN_LC_NATIVE_VISIBILITY':raise ValueError('Direct LC heads only')
    if len({r['LC_capture_receipt_sha256']for r in saved})!=1 or len({r['model_digest']for r in saved})!=1:raise ValueError('Source capture/field differs')
    restoration=json.loads((fit/'runs/technical_resume32/initial_state.json').read_text())['restoration']
    if not all(restoration.values()):raise ValueError('Strict resume-start failed')
    if b['camera_history']!=c['camera_history']or a['camera_history']!=c['camera_history'][:32]:raise ValueError('Camera packets differ')
    def delta(x,y):
        values=(x.double()-y.double()).numpy()
        return dict(max_abs=float(np.abs(values).max()),RMSE=float(np.sqrt(np.mean(values*values))),bit_identical_fraction=float((values==0).mean()))
    adam=[]
    for key,values in b['optimizer']['state'].items():
        other=c['optimizer']['state'][key]
        if values.keys()!=other.keys():raise ValueError('Adam state structure differs')
        adam.append(dict(parameter_key=key,tensors={name:delta(value,other[name])for name,value in values.items()}))
    rng={name:nested_digest(b['rng'][name])==nested_digest(c['rng'][name])for name in b['rng']}
    report=dict(CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),optimizer_created=False,
        resume_start=restoration,camera_packet_history_exact=True,head_logit_drift=delta(b['head']['logits'],c['head']['logits']),
        head_probability_drift=delta(b['head']['logits'].sigmoid(),c['head']['logits'].sigmoid()),Adam_trajectory_drift=adam,
        final_RNG_and_sampler_bit_identical=rng,checkpoint_sha256={str(p):sha(p)for p in paths},
        initial_probability=a['initial_probability'],source_pipeline_digest=a['model_digest'],source_capture_sha256=a['LC_capture_receipt_sha256'],
        automatic_numeric_acceptance_bound=None,automatic_promotion=False,
        interpretation='Exact resume START is required and passed. Subsequent CUDA grid_sample backward atomic head/Adam numeric differences are reported for root assessment, not claimed bitwise deterministic or automatically accepted.')
    write(output/'Adam_RNG_trajectory.json',report);print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fit-plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.fit_plan,args.output)
