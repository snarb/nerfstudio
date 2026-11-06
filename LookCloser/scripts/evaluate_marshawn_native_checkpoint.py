"""Frozen common actor metrics for an exact native checkpoint; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(a):
    import torch
    from scipy.ndimage import binary_dilation
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from marshawn_tuning_metrics import masked_actor_metrics,aggregate_actor_metrics
    a.output.mkdir(parents=True,exist_ok=False)
    assert sha(a.checkpoint)==a.checkpoint_sha256
    ready=json.loads(a.ready.read_text());box=json.loads(a.box.read_text());v2=json.loads(a.v2_manifest.read_text());v2rows={r['camera']:r for r in v2['records']}
    assert v2['training_use_forbidden'] and not v2['model_renders_used']
    records={r['camera']:r for r in ready['records']};rows={v:[] for v in ['v1','v2']};exterior=[];panels=[]
    # Render before creating LPIPS, keeping peak memory lower beside paused worker.
    for camera in ['DC','HC','KA']:
        command=[sys.executable,str(a.renderer),'--config',str(a.config),'--checkpoint',str(a.checkpoint),'--checkpoint-sha256',a.checkpoint_sha256,'--box',str(a.box),'--camera',camera,'--output',str(a.output/'renders'/camera)]
        for path in a.python_path:command+=['--python-path',str(path)]
        with (a.output/('render_'+camera+'.log')).open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    torch.set_num_threads(2);lp=LearnedPerceptualImagePatchSimilarity(normalize=True).cuda().eval().net.eval()
    for camera in ['DC','HC','KA']:
        r=records[camera];assert sha(a.ready.parent/r['image'])==r['rgb_sha256'];assert sha(r['semantic'])==r['semantic_sha256'];assert sha(v2rows[camera]['v2_archive'])==v2rows[camera]['v2_sha256']
        receipt=json.loads((a.output/'renders'/camera/'receipt.json').read_text());assert receipt['checkpoint_sha256']==a.checkpoint_sha256 and receipt['box_sha256']==sha(a.box) and receipt['updates']==0
        rgba=np.load(a.output/'renders'/camera/'raw_partition.npz')['inside'];target=np.asarray(Image.open(a.ready.parent/r['image']).convert('RGB'),np.float32)/255
        prediction=torch.tensor(rgba[...,:3],device='cuda').permute(2,0,1)[None];truth=torch.tensor(target,device='cuda').permute(2,0,1)[None]
        for version,path in [('v1',r['semantic']),('v2',v2rows[camera]['v2_archive'])]:
            with np.load(path) as masks:
                for actor in ['black','grey']:
                    metric=masked_actor_metrics(prediction,truth,torch.tensor(masks[actor],device='cuda')[None,None],lp,crop_xyxy=r['crops'][actor]);rows[version].append(dict(view=camera,actor=actor,**metric))
                if version=='v2':outside=~binary_dilation(masks['black']|masks['grey'],iterations=3)
        exterior.append(dict(view=camera,alpha_gt_01_outside_v2_mask_dilation3=int(((rgba[...,3]>.1)&outside).sum())))
        for actor in ['black','grey']:
            crop=r['crops'][actor];gt=Image.fromarray(np.rint(target*255).astype('uint8')).crop(crop);pred=Image.fromarray(np.rint(np.clip(rgba[...,:3],0,1)*255).astype('uint8')).crop(crop);panel=Image.new('RGB',(gt.width*2,gt.height+25),(20,20,20));panel.paste(gt,(0,25));panel.paste(pred,(gt.width,25));ImageDraw.Draw(panel).text((3,5),camera+' '+actor+' SOURCE | NATIVE '+str(receipt['step']),fill='white');panel.thumbnail((540,600));panels.append(panel)
    result={version:aggregate_actor_metrics(values) for version,values in rows.items()};result.update(step=receipt['step'],checkpoint=str(a.checkpoint),checkpoint_sha256=a.checkpoint_sha256,ready_sha256=sha(a.ready),box_sha256=sha(a.box),v2_manifest_sha256=sha(a.v2_manifest),metric_helper_sha256=sha(Path(__file__).with_name('marshawn_tuning_metrics.py')),exterior_diagnostic=exterior,no_training_or_model_mutation=True)
    cw=max(p.width for p in panels);ch=max(p.height for p in panels);contact=Image.new('RGB',(3*cw,2*ch),(20,20,20))
    for i,p in enumerate(panels):contact.paste(p,((i//2)*cw,(i%2)*ch))
    contact.save(a.output/'actor_comparison.jpg',quality=95)
    # Final result is the gate completion marker, published only after all artifacts exist.
    tmp=a.output/'evaluation.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output/'evaluation.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','checkpoint','output','ready','box','v2-manifest','renderer']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--python-path',type=Path,action='append',default=[])
    evaluate(p.parse_args())
