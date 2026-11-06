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
    script="import json,pathlib,hashlib,os,re,time\nr=pathlib.Path(ROOT);step=STEP;result={};generation=r/'semantic_late_distortion_candidate_v2'\nsha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()\ndef preserve(source,target):\n if target.exists():\n  if sha(target)!=sha(source):raise ValueError('Existing immutable archive differs')\n else:os.link(source,target)\nfor label,arm,target in [('parent',None,30376),('control','control001',step),('treatment','treatment1',step)]:\n if label=='parent':\n  p=r/'independent_occupancy/semantic_actor_progress/step-000030376';m=json.loads((p/'renders/metrics.json').read_text());folder=p/'renders';checkpoint=p/'checkpoint.ckpt';grid=p/'step-000030376_grid.json';cfg=p/'config.yml'\n else:\n  p=generation/'runs'/('diagnostic_'+arm);metrics=p/'metrics.json'\n  if not metrics.exists():print('{}');raise SystemExit\n  found=[x for x in json.loads(metrics.read_text())if x['step']==target]\n  if not found:print('{}');raise SystemExit\n  m=found[0];folder=p/'renders'/('step-%09d'%target);checkpoint=p/'checkpoints'/('step-%09d.ckpt'%target);grid=p/'checkpoints'/('step-%09d_grid.json'%target);cfg=p/'worker/lookcloser/frozen/config.yml'\n  if not checkpoint.exists()or not grid.exists()or not cfg.exists():print('{}');raise SystemExit\n native=json.loads(grid.read_text())\n if native['schema']!=2 or native['checkpoint_step']!=target or native['checkpoint_sha256']!=sha(checkpoint)or native['grid_archive_sha256']!=sha(grid.with_suffix('.npz')):raise ValueError('Actual schema2 snapshot/checkpoint binding differs')\n archive=None\n if arm:\n  archive=r/'independent_occupancy/semantic_late_pair_v2'/arm/('step-%09d'%target);archive.mkdir(parents=True,exist_ok=True)\n  specpath=generation/'specs'/('diagnostic_'+arm)/'spec.json';spec=json.loads(specpath.read_text());values=re.findall(r'^\\s*distortion_loss_mult:\\s*([0-9.eE+-]+)\\s*$',cfg.read_text(),re.M)\n  if len(values)!=1 or float(values[0])!=spec['distortion_coefficient']:raise ValueError('Saved effective distortion coefficient differs')\n  sources={'checkpoint.ckpt':checkpoint,grid.name:grid,grid.with_suffix('.npz').name:grid.with_suffix('.npz'),'config.yml':cfg,'spec.json':specpath,'root_release.json':specpath.parent/'root_release.json','source_inventory.json':generation/'source_inventory.json'}\n  for name,path in sources.items():preserve(path,archive/name)\n  record=dict(step=target,source_checkpoint_sha256=spec['parent_checkpoint_sha256'],effective_distortion_coefficient=float(values[0]),artifacts_sha256={name:sha(archive/name)for name in sources},actual_grid_schema=2,geometry_approved=False,source_anchor_status='EVALUATED_INTERMEDIATE',time=time.time())\n  receipt=archive/'archive_receipt.json'\n  if not receipt.exists():receipt.write_text(json.dumps(record,indent=2)+'\\n')\n renders={}\n for camera in ['DC','HC','KA']:\n  f=folder/(camera+'.png')\n  if not f.exists():print('{}');raise SystemExit\n  renders[camera]={'path':str(f),'sha256':sha(f)}\n result[label]={'metric':m,'renders':renders,'checkpoint_sha256':native['checkpoint_sha256'],'grid_json_sha256':sha(grid),'archive':str(archive)if archive else str(p),'actual_grid_schema':2}\nprint(json.dumps(result))\n".replace('ROOT',repr(SC)).replace('STEP',str(step))
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
        labels=['GT','parent30376','control .01','treatment1']
        for region,box in [('full',None),*records[camera]['crops'].items()]:
            crops=[im if box is None else im.crop(tuple(box))for im in images]
            width=640 if box is None else crops[0].width;height=round(crops[0].height*width/crops[0].width)
            canvas=Image.new('RGB',(width*4,height+32),'#dddddd');draw=ImageDraw.Draw(canvas)
            for index,(im,label)in enumerate(zip(crops,labels)):
                canvas.paste(im.resize((width,height),Image.Resampling.LANCZOS),(index*width,32));draw.text((index*width+5,8),camera+' '+region+' '+label,fill='black')
            canvas.save(target/(camera+'_'+region+'_contact.png'))
    metrics=quality_only({k:v['metric']for k,v in evidence.items()});write(target/'metrics_v1.json',metrics)
    write(target/'receipt.json',dict(time=time.time(),step=step,attempts=step-30376,sources=hashes,checkpoint_archives={k:{x:v[x]for x in ('archive','checkpoint_sha256','grid_json_sha256','actual_grid_schema')}for k,v in evidence.items()},protocol='unchanged v1 actor RGB/crops/masks',visual_review_passed=False,geometry_approved=False,optimization_complete=False))
    lines=['# Paired continuation diagnostic — visual review pending','',f'Global step {step}; {step-30376} continuation attempts. Source step30376.','', '| Arm | PSNR | SSIM | LPIPS |','|---|---:|---:|---:|']
    for label,m in metrics.items():
        q=m['metrics'];lines.append(f"| {label} | {q['psnr']:.5f} | {q['ssim']:.5f} | {q['lpips']:.6f} |")
    lines+=['','Per-view/per-actor scores: metrics_v1.json. All nine full/actor contact images are saved. This is a diagnostic gate; geometry and visual approval remain pending.','']
    (target/'report.md').write_text('\n'.join(lines))

def watch(root):
    root.mkdir(parents=True,exist_ok=True);lastlog=0
    while True:
        try:
            state=read(C/'semantic_late_distortion_diagnostic_v2_controller/state.json');statuses={k:v['status']for k,v in state['jobs'].items()}
            for step in (30876,31376,32376,34376):
                if not (root/f'step-{step:09d}'/'receipt.json').exists():
                    evidence=remote_gate(step)
                    if evidence:build(root,step,evidence)
            row=dict(time=time.time(),jobs=statuses,ready_gates=[s for s in (30876,31376,32376,34376)if(root/f'step-{s:09d}'/'receipt.json').exists()],manual_review_passed=False)
            write(root/'status.json',row)
            if time.time()-lastlog>=3600:
                with(root/'checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
                lastlog=time.time()
            if len(row['ready_gates'])==4:return
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
