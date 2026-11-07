#!/usr/bin/env python3
"""Durable read-only preservation and local fetch for the qualified-tau diagnostic trio."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time,shlex
from pathlib import Path
ARMS=('control','tau0001','tau001');GATES=(111816,112316,113316,115316)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p,default=None):return json.loads(Path(p).read_text())if Path(p).exists()else default
def write(p,v):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def process(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split();return dict(pid=pid,start=f[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=f[0])
 except FileNotFoundError:return None
def alive(p):
 current=process(p['pid'])if p else None
 return bool(current and current['state']!='Z'and all(current[k]==p[k]for k in('pid','start','boot')))
def archive(root):
 root=Path(root);records=[];statuses={};out=root/'archives';out.mkdir(exist_ok=True)
 for arm in ARMS:
  run=root/'runs'/('diagnostic_'+arm);job='qualified_tau111316_diagnostic_'+arm;statuses[arm]=dict(progress=read(run/'progress.json'),complete=read(run/'complete.json'),worker_receipt=read(root/'remote_queue/jobs'/job/'receipt.json'))
  terminal=statuses[arm]['complete']
  if terminal is not None and not(terminal.get('status')=='diagnostic_visual_gate_pending'and terminal.get('technical_smoke')is False and terminal.get('final_step')==115316 and terminal.get('attempts')==4000 and terminal.get('auxiliary_draws')==4000):raise ValueError('Malformed diagnostic completion for '+arm)
  for step in GATES:
   name=f'step-{step:09d}';render=run/'renders'/name;grid=run/'checkpoints'/(name+'_grid.json');cp=run/'checkpoints'/(name+'.ckpt');clear=run/'clear_targets'/(name+'.json')
   if not all(p.exists()for p in[render/'metrics.json',grid,cp,clear]):continue
   meta=read(grid)
   if meta['schema']!=2 or meta['checkpoint_step']!=step:raise ValueError('Actual schema2 checkpoint gate mismatch')
   dest=out/arm/name;dest.mkdir(parents=True,exist_ok=True);receipt=dest/'archive_receipt.json'
   if receipt.exists():
    old=read(receipt)
    if old['checkpoint_sha256']!=meta['checkpoint_sha256']or old['grid_archive_sha256']!=meta['grid_archive_sha256']:raise ValueError('Previously archived gate was re-executed with different state')
    records.append(old);continue
   if sha(cp)!=meta['checkpoint_sha256']or sha(grid.with_suffix('.npz'))!=meta['grid_archive_sha256']:raise ValueError('Source checkpoint/grid changed during archive')
   try:os.link(cp,dest/'checkpoint.ckpt')
   except FileExistsError:
    if sha(dest/'checkpoint.ckpt')!=meta['checkpoint_sha256']:raise ValueError('Existing archive weights differ')
   sources={'grid.json':grid,'grid.npz':grid.with_suffix('.npz'),'config.yml':run/'worker/lookcloser/frozen/config.yml','spec.json':root/'specs'/('diagnostic_'+arm)/'spec.json','source_inventory.json':root/'source_inventory.json','targets.json':root/'targets.json','root_release.json':root/'specs'/('diagnostic_'+arm)/'root_release.json','clear_targets.json':clear}
   for name,p in sources.items():shutil.copy2(p,dest/name)
   shutil.copytree(render,dest/'renders',dirs_exist_ok=True);hashes={str(p.relative_to(dest)):sha(p)for p in dest.rglob('*')if p.is_file()and p.name not in('checkpoint.ckpt','archive_receipt.json')};row=dict(arm=arm,step=step,archive=str(dest),checkpoint_sha256=meta['checkpoint_sha256'],grid_archive_sha256=meta['grid_archive_sha256'],small_files_sha256=hashes,immutable_weights_hardlink=True,manual_review_passed=False,geometry_approved=False,time=time.time());write(receipt,row);records.append(row)
 return dict(time=time.time(),arms=statuses,archives=records,all_complete=len(records)==12 and all(v['complete']is not None for v in statuses.values()),status='diagnostic_only_manual_review_pending')
def comparisons(root,step):
 from PIL import Image,ImageDraw
 folders={arm:root/'archives'/arm/f'step-{step:09d}' for arm in ARMS}
 if not all((p/'fetch_complete.json').exists()for p in folders.values()):return
 out=root/'comparisons'/f'step-{step:09d}';out.mkdir(parents=True,exist_ok=True)
 if(out/'receipt.json').exists():return
 campaign=root.parent;ready=read(campaign/'data/ready.json');parent=campaign/'independent_occupancy/semantic_selected_geometry/R0_W64/renders'
 metrics={name:read(p/'renders/metrics.json')for name,p in folders.items()};proof={}
 for row in ready['records']:
  camera=row['camera']
  if camera not in('DC','HC','KA'):continue
  crops=list(row['crops'].values());box=(max(0,min(v[0]for v in crops)-24),max(0,min(v[1]for v in crops)-24),min(1920,max(v[2]for v in crops)+24),min(1012,max(v[3]for v in crops)+32));w=box[2]-box[0];h=box[3]-box[1];canvas=Image.new('RGB',(w*5,h+32),'#dddddd');draw=ImageDraw.Draw(canvas)
  sources=[('GT',campaign/'data'/row['image']),('parent111316',parent/(camera+'.png'))]+[(arm,folders[arm]/'renders'/(camera+'.png'))for arm in ARMS]
  for i,(label,path)in enumerate(sources):canvas.paste(Image.open(path).convert('RGB').crop(box),(i*w,32));draw.text((i*w+4,8),camera+' '+label,fill='black');proof[str(path)]=sha(path)
  canvas.save(out/(camera+'_GT_parent_control_tau0001_tau001.png'))
 def clean(x):return {k:clean(v)for k,v in x.items()if'loss'not in k}if isinstance(x,dict)else[clean(v)for v in x]if isinstance(x,list)else x
 guard={arm:{actor:metrics[arm]['by_actor'][actor]['lpips']<=1.03*metrics['control']['by_actor'][actor]['lpips']for actor in('black','grey')}for arm in ARMS if arm!='control'}
 write(out/'metrics.json',dict(step=step,attempts=step-111316,V1=clean(metrics),per_actor_3percent_guard=guard,status='provisional_manual_geometry_review_pending',quality_promoted=False));write(out/'receipt.json',dict(time=time.time(),source_images_sha256=proof,manual_review_passed=False,geometry_approved=False))

def watch(root,remote,host,remote_script):
 out=root/'watcher';out.mkdir(exist_ok=True);lock=(out/'watcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);last=0;oldcount=-1
 while True:
  try:
   cmd=[str(Path(remote).parent/'python'),remote_script,'--remote-once','--root',remote];r=subprocess.run(['ssh','-o','ConnectTimeout=10',host,shlex.join(cmd)],capture_output=True,text=True,timeout=60,check=True);s=json.loads(r.stdout)
   for row in s['archives']:
    dst=root/'archives'/row['arm']/f"step-{row['step']:09d}"
    if not(dst/'fetch_complete.json').exists():
     dst.mkdir(parents=True,exist_ok=True);subprocess.run(['rsync','-a','--exclude=checkpoint.ckpt',host+':'+row['archive']+'/',str(dst)+'/'],check=True,timeout=180,stdout=subprocess.DEVNULL)
     for name,h in row['small_files_sha256'].items():
      if sha(dst/name)!=h:raise ValueError('Fetched artifact differs '+name)
     write(dst/'fetch_complete.json',dict(time=time.time(),all_small_hashes_verified=True,weights_retained_remote=True,manual_review_passed=False))
   for step in GATES:comparisons(root,step)
   write(out/'status.json',s)
   if len(s['archives'])!=oldcount or time.time()-last>=3600 or s['all_complete']:
    with(out/'checks.jsonl').open('a')as f:f.write(json.dumps(s)+'\n')
    oldcount=len(s['archives']);last=time.time()
   if s['all_complete']:return
  except Exception as e:write(out/'error.json',dict(time=time.time(),error=repr(e)))
  time.sleep(60)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--remote-root');p.add_argument('--host',default='ubuntu@silver-sable');p.add_argument('--remote-script');p.add_argument('--remote-once',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
 if a.remote_once:print(json.dumps(archive(a.root)))
 elif a.detach:
  out=a.root/'watcher';out.mkdir(exist_ok=True);previous=read(out/'launch.json',{})
  if alive(previous.get('process')):print(previous['process']['pid']);raise SystemExit
  with(out/'watcher.log').open('ab')as log:child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--root',str(a.root),'--remote-root',a.remote_root,'--host',a.host,'--remote-script',a.remote_script],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(out/'launch.json',dict(process=process(child.pid),source_sha256=sha(__file__),time=time.time()));print(child.pid)
 else:watch(a.root,a.remote_root,a.host,a.remote_script)
