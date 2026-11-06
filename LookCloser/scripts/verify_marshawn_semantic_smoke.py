"""Real native semantic checkpoint/Adam/resume/active-mask technical verification."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from marshawn_semantic_admission import sha
from marshawn_semantic_occupancy import load_snapshot


def inspect(path):
    data=torch.load(path,map_location='cpu',weights_only=False);states=data['optimizers']['fields']['state']
    if not states:raise ValueError('Nonempty Adam state required')
    steps=sorted({int(v['step'])for v in states.values()})
    if min(steps)<1:raise ValueError('Successful Adam step required')
    for state in states.values():
        for key in ('exp_avg','exp_avg_sq'):
            if not bool(torch.isfinite(state[key]).all()):raise ValueError('Nonfinite Adam moment')
    if not all(bool(torch.isfinite(v).all())for v in data['pipeline'].values()if torch.is_tensor(v)and v.is_floating_point()):raise ValueError('Nonfinite model state')
    capture=data['actual_occupancy_capture'];keys=[k for k in data['pipeline']if k.endswith('semantic_learned_core')]
    if len(keys)!=1 or not torch.equal(data['pipeline'][keys[0]],capture['core']):raise ValueError('Persistent learned core differs from actual captured core')
    return dict(path=str(path),sha256=sha(path),attempted_step=data['step'],adam_entries=len(states),adam_steps=steps,scaler=data['scalers'],scheduler=data['schedulers'],sampling_state=data['tuning_runtime']['sampling_state'],semantic_receipt=capture['semantic_receipt'])


def main(args):
    before,after=inspect(args.before),inspect(args.after)
    if min(after['adam_steps'])<=max(before['adam_steps']):raise ValueError('Resume did not perform successful Adam updates')
    if before['semantic_receipt']!=after['semantic_receipt']:raise ValueError('Admission changed across resume')
    resume=json.loads((args.run/'resume_rng_verification.json').read_text())
    if resume['passed']is not True or resume['restored_checkpoint_step']!=before['attempted_step']:raise ValueError('RNG restoration not verified')
    snapshots=[]
    for checkpoint,row in [(args.before,before),(args.after,after)]:
        receipt=args.run/'occupancy'/f"step-{row['attempted_step']:09d}_grid.json"
        core,active,ema,box,meta=load_snapshot(receipt,checkpoint)
        snapshots.append(dict(step=row['attempted_step'],core_cells=int(core.sum()),active_cells=int(active.sum()),receipt_sha256=sha(receipt),semantic_receipt=meta['semantic_receipt'],native_core_ema_separate=True))
    renders=[]
    for view in ('DC','HC','KA'):
        p=args.run/'renders'/f"step-{after['attempted_step']:09d}"/f'{view}.png';a=np.array(Image.open(p));alpha=np.array(Image.open(p.with_name(view+'_alpha.png')))
        if a.shape!=(1012,1920,3)or alpha.shape!=(1012,1920):raise ValueError('Incomplete technical raster')
        renders.append(dict(view=view,rgb_sha256=sha(p),min=int(a.min()),max=int(a.max()),alpha_min=int(alpha.min()),alpha_max=int(alpha.max())))
    report=dict(passed=True,kind='SEMANTIC_NATIVE_ADAM_RESUME_TECHNICAL',before=before,after=after,resume_rng=resume,snapshots=snapshots,renders=renders,quality_claim=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,adam_before=before['adam_steps'],adam_after=after['adam_steps'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('before','after','run','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
