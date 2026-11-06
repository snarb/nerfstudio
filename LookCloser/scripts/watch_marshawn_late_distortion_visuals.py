#!/usr/bin/env python3
"""Read-only rendered comparisons at paired late-distortion diagnostic gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
SC='/home/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def quality_only(v):
    if isinstance(v,dict):return {k:quality_only(x)for k,x in v.items()if 'loss'not in k}
    if isinstance(v,list):return list(map(quality_only,v))
    return v

def remote_gate(step):
    script="""import json,pathlib,hashlib
r=pathlib.Path(ROOT);step=STEP;result={}
for label,job,target in [('parent','structural_exp_hplus1',60752),('control','late_distortion_diagnostic_control001',step),('treatment','late_distortion_diagnostic_treatment1',step)]:
 p=r/'jobs'/job/'run';metrics=p/'metrics.json'
 if label=='parent':m=json.loads((p/'complete.json').read_text())['selection']
 else:
  if not metrics.exists():print('{}');raise SystemExit
  found=[x for x in json.loads(metrics.read_text())if x['step']==target]
  if not found:print('{}');raise SystemExit
  m=found[0]
 renders={}
 for camera in ['DC','HC','KA']:
  f=p/'renders'/('step-%09d'%target)/(camera+'.png')
  if not f.exists():print('{}');raise SystemExit
  renders[camera]={'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 result[label]={'metric':m,'renders':renders}
print(json.dumps(result))
""".replace('ROOT',repr(SC)).replace('STEP',str(step))
    result=subprocess.run(['ssh','-o','ConnectTimeout=10','ubuntu@silver-sable','python3','-'],input=script,text=True,capture_output=True,timeout=30,check=True)
    return json.loads(result.stdout)

def build(root,step,evidence):
    from PIL import Image,ImageDraw
    target=root/f'step-{step:09d}';target.mkdir(exist_ok=True)
    if (target/'receipt.json').exists():return
    data=C/'data';ready=read(data/'ready.json');records={r['camera']:r for r in ready['records']};hashes={}
    for camera in ('DC','HC','KA'):
        source=data/records[camera]['image'];hashes['GT_'+camera]=dict(path=str(source),sha256=sha(source));images=[Image.open(source).convert('RGB')]
        for label in ('parent','control','treatment'):
            item=evidence[label]['renders'][camera];dest=target/(camera+'_'+label+'.png')
            subprocess.run(['scp','-q','ubuntu@silver-sable:'+item['path'],str(dest)],check=True,timeout=30)
            if sha(dest)!=item['sha256']:raise ValueError('Render transfer hash mismatch')
            hashes[camera+'_'+label]=item;images.append(Image.open(dest).convert('RGB'))
        if any(im.size!=images[0].size for im in images):raise ValueError('View source/render shapes differ')
        labels=['GT','parent60752','control .01','treatment1']
        for region,box in [('full',None),*records[camera]['crops'].items()]:
            crops=[im if box is None else im.crop(tuple(box))for im in images]
            width=640 if box is None else crops[0].width;height=round(crops[0].height*width/crops[0].width)
            canvas=Image.new('RGB',(width*4,height+32),'#dddddd');draw=ImageDraw.Draw(canvas)
            for index,(im,label)in enumerate(zip(crops,labels)):
                canvas.paste(im.resize((width,height),Image.Resampling.LANCZOS),(index*width,32));draw.text((index*width+5,8),camera+' '+region+' '+label,fill='black')
            canvas.save(target/(camera+'_'+region+'_contact.png'))
    metrics=quality_only({k:v['metric']for k,v in evidence.items()});write(target/'metrics_v1.json',metrics)
    write(target/'receipt.json',dict(time=time.time(),step=step,attempts=step-60752,sources=hashes,protocol='unchanged v1 actor RGB/crops/masks',visual_review_passed=False,geometry_approved=False,optimization_complete=False))
    lines=['# Paired continuation diagnostic — visual review pending','',f'Global step {step}; {step-60752} continuation attempts. Source step60752.','', '| Arm | PSNR | SSIM | LPIPS |','|---|---:|---:|---:|']
    for label,m in metrics.items():
        q=m['metrics'];lines.append(f"| {label} | {q['psnr']:.5f} | {q['ssim']:.5f} | {q['lpips']:.6f} |")
    lines+=['','Per-view/per-actor scores: metrics_v1.json. All nine full/actor contact images are saved. This is a diagnostic gate; geometry and visual approval remain pending.','']
    (target/'report.md').write_text('\n'.join(lines))

def watch(root):
    root.mkdir(parents=True,exist_ok=True);lastlog=0
    while True:
        try:
            state=read(C/'late_distortion_diagnostic_controller/state.json');statuses={k:v['status']for k,v in state['jobs'].items()}
            for step in (61252,64752):
                if not (root/f'step-{step:09d}'/'receipt.json').exists():
                    evidence=remote_gate(step)
                    if evidence:build(root,step,evidence)
            row=dict(time=time.time(),jobs=statuses,ready_gates=[s for s in (61252,64752)if(root/f'step-{s:09d}'/'receipt.json').exists()],manual_review_passed=False)
            write(root/'status.json',row)
            if time.time()-lastlog>=3600:
                with(root/'checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
                lastlog=time.time()
            if len(row['ready_gates'])==2:return
            if any(x in ('failed','blocked','orphaned')for x in statuses.values()):raise RuntimeError('Diagnostic job requires failure review')
        except Exception as exc:write(root/'error.json',dict(time=time.time(),error=repr(exc)))
        time.sleep(60)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--detach',action='store_true');a=p.parse_args()
    if a.detach:
        a.root.mkdir(parents=True,exist_ok=True)
        with(a.root/'watcher.log').open('a')as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--root',str(a.root)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write(a.root/'watcher.json',dict(pid=proc.pid,time=time.time(),source_sha256=sha(__file__)));print(proc.pid)
    else:watch(a.root)
