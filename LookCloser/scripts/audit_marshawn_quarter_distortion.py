"""CPU actual-config and callback proof of the quarter × distortion interaction."""
import argparse,json,os,sys
from pathlib import Path
from audit_marshawn_cold_quarterstep import canonical

def main(root,parent):
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(root/'code'),str(root/'code/LookCloser/scripts')]
    import torch
    import train_marshawn_semantic_admission as worker
    spec=json.loads((root/'spec.json').read_text());before=json.loads((parent/'spec.json').read_text());worker.validate_spec(spec)
    oldroot=str(Path(before['output']).parent);newroot=str(Path(spec['output']).parent)
    normalized=json.loads(json.dumps(spec).replace(newroot,oldroot));normalized['distortion_final']=.01
    if normalized!=before or 'frequency_focus'in spec:raise ValueError('Undeclared scientific change')
    old=canonical(worker.build_config(before));new=json.loads(json.dumps(canonical(worker.build_config(spec))).replace(newroot,oldroot))
    if new!=old:raise ValueError('Initial actual configs differ')
    model=new['pipeline']['model']
    if model['adaptive_coarse_step_size']!=.0015625 or model['max_steps_per_ray']!=1024 or model['corrected_arm_allocator']:raise ValueError('Original quarter/1024/legacy required')
    rejected=[]
    for key,value in [('coarse_step',.00625),('coarse_step',.003125),('distortion_final',.01),('width',32),('radius',1),('frequency_focus',{})]:
        try:worker.validate_spec(dict(spec,**{key:value}))
        except ValueError:rejected.append([key,value])
        else:raise ValueError('Undeclared factor accepted')
    text=(root/'code/LookCloser/scripts/train_marshawn_semantic_admission.py').read_text();lines=text.splitlines();fraction=next(x.strip()for x in lines if x.strip().startswith('fraction = min(1.'));coefficient=next(x.strip().split(' = ',1)[1]for x in lines if x.strip().startswith('model.config.distortion_loss_mult ='))
    schedule=[]
    for step in [0,4096,8192,12288,75941,106316]:
        values=dict(spec=spec,step=step);exec(fraction,{},values);actual=eval(coefficient,{},values);assert actual==.01+(.1-.01)*min(1.,max(0.,(step-4096)/8192));schedule.append(dict(step=step,coefficient=actual))
    if torch.cuda.is_initialized():raise ValueError('CPU proof initialized CUDA')
    report=dict(passed=True,initial_actual_config_difference={},scientific_spec_difference={'distortion_final':[.01,.1]},actual_coarse=.0015625,actual_cap=1024,corrected_allocator=False,focused_hook=False,runtime_ramp=schedule,rejected_variants=rejected,CUDA_initialized=False)
    (root/'config_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();main(a.root,a.parent)
