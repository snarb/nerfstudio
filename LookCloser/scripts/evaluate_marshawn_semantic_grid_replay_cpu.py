"""CPU-only fixed V1 metrics and contacts after the GPU lease has released."""
import argparse
import json
import os
from pathlib import Path
from run_marshawn_semantic_grid_replay import validate,sha,write,absolute


def clean(value):
    if isinstance(value,dict):return{k:clean(v)for k,v in value.items()if 'loss'not in k}
    if isinstance(value,list):return[clean(v)for v in value]
    return value


def evaluate(path,execute=False):
    plan,_=validate(path,execute)
    if not execute:print(json.dumps(dict(CPU_only=True,GPU_started=False)));return
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    import lpips
    from PIL import Image,ImageDraw
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():raise ValueError('CPU evaluator must not initialize CUDA')
    root=Path(plan['output']);done=json.loads((root/'complete.json').read_text())
    if not done['passed']or not done['source_restored_after_each_phase']:raise ValueError('Incomplete immutable physical replay')
    metrics=absolute('_grid_V1_metrics',Path(plan['helpers'])/'marshawn_tuning_metrics.py')
    model=lpips.LPIPS(net='alex').eval().cpu();model.requires_grad_(False)
    ready=json.loads(Path(plan['ready']).read_text());records={x['camera']:x for x in ready['records']}
    region=json.loads(Path(plan['regions']).read_text());dest=root/'visual_review';dest.mkdir(exist_ok=False)
    variants={};images={};geometry={}
    for phase in plan['phases']:
        scores=[];images[phase]={}
        details=json.loads((root/phase/'complete.json').read_text())
        for cam in ['DC','HC','KA']:
            rec=records[cam];gt=np.array(Image.open(Path(plan['data'])/rec['image']).convert('RGB')).astype(np.float32)/255
            with np.load(root/phase/(cam+'_physical.npz'))as z:rgb=np.clip(z['RGBA'][...,:3],0,1).copy()
            target=torch.from_numpy(gt).permute(2,0,1)[None];prediction=torch.from_numpy(rgb).permute(2,0,1)[None]
            with np.load(Path(plan['data'])/'semantics'/(cam+'.npz'))as sem:
                for actor in ['black','grey']:
                    score=metrics.masked_actor_metrics(prediction,target,torch.from_numpy(sem[actor].copy())[None,None],model,crop_xyxy=rec['crops'][actor]);scores.append(dict(view=cam,actor=actor,**score))
            images[phase][cam]=Image.fromarray(np.rint(rgb*255).astype(np.uint8))
        variants[phase]=clean(metrics.aggregate_actor_metrics(scores))
        groups={}
        for q in details['queries']:
            key=q['pixel'].get('diagnostic_group',q['group']);groups.setdefault(key,[]).append(q)
        geometry[phase]={group:dict(rays=len(rows),native_cap_hits=sum(r['cap_hit']for r in rows),max_native_samples=max(r['native']['samples']for r in rows),
            mean_uncovered_selected_mass=float(np.mean([r['missed_selected_mass']['native_interval_uncovered_selected_mass']for r in rows])),
            mean_tail_selected_mass=float(np.mean([r['missed_selected_mass']['selected_mass_beyond_native_last_end']for r in rows])),
            max_dense_convergence_RGB=max(r['convergence']['RGB_max_abs_delta']for r in rows),
            max_dense_convergence_opacity=max(r['convergence']['opacity_abs_delta']for r in rows))for group,rows in groups.items()}
    reference=json.loads(Path(plan['P3_reference_metrics']).read_text())['variants']['hard_theta01']
    delta=[]
    for actual,expected in zip(variants['source_grid']['records'],reference['records']):
        if (actual['view'],actual['actor'])!=(expected['view'],expected['actor']):raise ValueError('Reference metrics ordering differs')
        entry={k:abs(actual[k]-expected[k])for k in ['psnr','ssim','lpips']};delta.append(dict(view=actual['view'],actor=actual['actor'],**entry))
        if max(entry.values())>1e-5:raise ValueError('Original P3 selected V1 numeric parity failed '+str(delta[-1]))
    for cam in ['DC','HC','KA']:
        gt=Image.open(Path(plan['data'])/records[cam]['image']).convert('RGB')
        regions=dict(region[cam]);boxes=list(records[cam]['crops'].values());regions['body']=[min(b[0]for b in boxes),min(b[1]for b in boxes),max(b[2]for b in boxes),max(b[3]for b in boxes)]
        for name,box in regions.items():
            patches=[gt.crop(box)]+[images[p][cam].crop(box)for p in plan['phases']];w,h=patches[0].size
            sheet=Image.new('RGB',(4*w,h+28),'white');draw=ImageDraw.Draw(sheet)
            for i,(label,patch)in enumerate(zip(['GT']+plan['phases'],patches)):sheet.paste(patch,(i*w,28));draw.text((i*w+3,4),label,fill='black')
            sheet.save(dest/(cam+'_'+name+'.png'))
    write(dest/'metrics.json',dict(variants=variants,source_P3_metric_absolute_deltas=delta,geometry=geometry,
        fixed_V1=True,head_label_regime=plan['source_label_regime'],semantic_support_not_physical_free_space=True,quality_promoted=False,
        source_complete_sha256=sha(root/'complete.json'),CUDA_initialized=torch.cuda.is_initialized(),manual_review_required=True))
    write(dest/'artifacts.json',{str(p.relative_to(dest)):sha(p)for p in sorted(dest.glob('*'))if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();evaluate(a.plan,a.execute)
