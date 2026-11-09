#!/usr/bin/env python3
"""Launch an exclusive-GPU batch and report every experiment in one command.

Standard-library bootstrap tool. Workers run through the existing bounded quiet
supervisor. Dataset/cache paths may be volatile; campaign roots and model outputs
belong on the persistent boot disk. Archive uses rclone copy, never sync/delete.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path


def now():return datetime.now(timezone.utc).isoformat()
def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError,TypeError):return default

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name('.'+path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def tail(path,limit=65536):
 try:
  with Path(path).open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-limit));return f.read().decode(errors='replace')
 except OSError:return ''

def gpu_snapshot():
 def query(kind,fields):
  r=subprocess.run(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
  if r.returncode:raise RuntimeError(r.stderr.strip()[:250])
  return list(csv.reader(r.stdout.splitlines(),skipinitialspace=True))
 gpus=[]
 for row in query('gpu','index,uuid,name,memory.total,memory.used,utilization.gpu'):
  gpus.append(dict(index=int(row[0]),uuid=row[1],name=row[2],total_mib=int(row[3]),used_mib=int(row[4]),utilization=int(row[5]),processes=[]))
 by={g['uuid']:g for g in gpus}
 for row in query('compute-apps','gpu_uuid,pid,process_name,used_memory'):
  if len(row)==4 and row[0] in by:by[row[0]]['processes'].append(dict(pid=int(row[1]),name=row[2],memory_mib=int(row[3]) if row[3].isdigit() else None))
 return gpus

def live_controller(pid,script):
 try:return str(script).encode() in Path('/proc',str(pid),'cmdline').read_bytes()
 except OSError:return False

def process_alive(pid):
 try:return Path('/proc',str(pid)).is_dir() and Path('/proc',str(pid),'stat').read_text().split(') ')[1][0]!='Z'
 except OSError:return False

def compact_progress(spec):
 available=[]
 for name in spec.get('progress_files',[]):
  path=Path(name)
  if path.is_file():available.append(path)
 path=max(available,key=lambda p:p.stat().st_mtime) if available else None
 value=read(path,{}) if path else {}
 keys=['status','step','completed','total','camera','elapsed_seconds']
 progress={k:value[k] for k in keys if k in value};progress['phase']=path.parent.name if path else None
 metric=read(spec.get('metrics_file'),{}) or value.get('selection',{})
 if isinstance(metric,list):metric=metric[-1] if metric else {}
 if not isinstance(metric,dict):metric={}
 if 'aggregate' in metric:metric=metric['aggregate'].get('hair',{});scope='hair'
 else:scope='native-best'
 scores={k:metric.get('eval_all_'+k,metric.get(k)) for k in ['psnr','ssim','lpips']}
 return progress,dict(scope=scope,**{k:v for k,v in scores.items() if isinstance(v,(int,float))})

def status(root):
 manifest=read(root/'manifest.json',dict(jobs=[]));gpus=[];error=None
 try:gpus=gpu_snapshot()
 except (OSError,subprocess.SubprocessError,ValueError,RuntimeError) as e:error=str(e)
 jobs=[];gpu_seconds=0.
 for spec in manifest['jobs']:
  control=Path(spec['control_root']);request=read(control/'request.json',{});result=read(control/'result.json');last=read_last_json(control/'checks.jsonl');pid=request.get('pid',spec.get('controller_pid'));alive=live_controller(pid,manifest.get('supervisor','run_dec5_hair_campaign.py')) if pid else False
  state=('complete' if result['returncode']==0 else 'failed') if result is not None else ('running' if alive else 'controller-missing' if request else 'setup-failed' if spec.get('controller_pid') else 'starting')
  elapsed=float(result.get('elapsed_seconds',0)) if result is not None else max(float(last.get('elapsed_seconds',0)),time.time()-spec['launched_epoch'] if alive else 0)
  gpu_seconds+=elapsed;progress,scores=compact_progress(spec);logs=tail(control/'worker.log');oom=('out of memory' in logs.lower() or 'cudnn_status_alloc_failed' in logs.lower())
  jobs.append(dict(name=spec['name'],gpu=spec['gpu'],state=state,controller_alive=alive,worker_pid=last.get('worker_pid'),worker_alive=process_alive(last.get('worker_pid')) if last.get('worker_pid') else None,elapsed_seconds=elapsed,progress=progress,metrics=scores,oom=bool(result.get('oom_evidence',oom)) if result is not None else oom,returncode=result.get('returncode') if result else None,stop_reason=result.get('stop_reason') if result else None,heartbeat=last.get('time')))
 return dict(time=now(),root=str(root),gpu_hours=gpu_seconds/3600,max_gpu_hours=manifest.get('max_gpu_hours'),gpus=gpus,gpu_query_error=error,jobs=jobs,free_root_gib=shutil.disk_usage(root if root.exists() else root.parent).free/(1<<30))

def read_last_json(path):
 lines=tail(path).splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except ValueError:continue
 return {}

def display(d,active_only=False):
 print('%s | %d GPUs | %.3f GPUh | disk free %.1f GiB'%(d['time'],len(d['gpus']),d['gpu_hours'],d['free_root_gib']))
 print('GPU memory MiB: '+', '.join('%d:%d/%d (%d%%)'%(g['index'],g['used_mib'],g['total_mib'],g['utilization']) for g in d['gpus']))
 if d['gpu_query_error']:print('GPU query: '+d['gpu_query_error'])
 print('JOB                         GPU STATE                PROGRESS              P / S / L                 MIN OOM')
 for j in d['jobs']:
  if active_only and j['state'] in ['complete','failed','setup-failed']:continue
  p=j['progress'];phase=('step='+str(p['step'])) if 'step' in p else ('%s/%s'%(p.get('completed','?'),p.get('total','?'))) if 'completed' in p else p.get('phase') or '-';phase=(str(p.get('phase'))+':'+phase) if p.get('phase') and 'completed' in p else phase;m=j['metrics'];scores=' / '.join('%.4f'%m[k] if k in m else '-' for k in ['psnr','ssim','lpips'])
  print('%-27.27s %3d %-20.20s %-21.21s %-25.25s %4.1f %s'%(j['name'],j['gpu'],j['state'],phase,scores,j['elapsed_seconds']/60,'YES' if j['oom'] else '-'))
 if active_only:print('Closed: %d; active/unresolved: %d'%(sum(j['state'] in ['complete','failed','setup-failed'] for j in d['jobs']),sum(j['state'] not in ['complete','failed','setup-failed'] for j in d['jobs'])))
 if not d['jobs']:print('(no registered experiments)')

def launch(a):
 plan=read(a.plan)
 if not plan or not isinstance(plan.get('jobs'),list) or not plan['jobs']:raise ValueError('Plan needs nonempty jobs')
 previous=read(a.root/'manifest.json')
 if previous and not a.append:raise FileExistsError('Campaign exists; use --append for another bounded batch')
 if a.append and not previous:raise FileNotFoundError('Append requires an existing campaign manifest')
 gpus={g['index']:g for g in gpu_snapshot()};names={j['name'] for j in previous['jobs']} if previous else set();assigned=set();maximum=0.;jobs=[]
 for old in previous['jobs'] if previous else []:
  receipt=read(Path(old['control_root'])/'result.json')
  if receipt is not None:maximum+=float(receipt.get('elapsed_seconds',old['max_seconds']))
  elif live_controller(old.get('controller_pid'),previous['supervisor']):maximum+=old['max_seconds'];assigned.add(old['gpu'])
  elif read(Path(old['control_root'])/'request.json'):raise RuntimeError('Resolve missing controller before appending jobs')
 for raw in plan['jobs']:
  j=dict(raw);name=j['name'];gpu=int(j['gpu']);seconds=float(j['max_seconds'])
  if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name) or name in names:raise ValueError('Unique simple job name required')
  if gpu not in gpus or gpu in assigned:raise ValueError('One job per existing physical GPU in this batch')
  if gpus[gpu]['processes']:raise RuntimeError('GPU %d already has compute processes; nothing launched'%gpu)
  if j.get('cwd') and not Path(j['cwd']).is_dir():raise FileNotFoundError('Worker cwd missing')
  if seconds<=0 or not isinstance(j.get('command'),list) or not j['command'] or not all(isinstance(v,str) for v in j['command']):raise ValueError('Positive time limit and argv command required')
  names.add(name);assigned.add(gpu);maximum+=seconds;j.update(gpu=gpu,gpu_uuid=gpus[gpu]['uuid'],control_root=str(a.root/(name+'_control')));jobs.append(j)
 budget=float(plan['max_gpu_hours'])
 if previous and budget!=previous['max_gpu_hours']:raise ValueError('Append cannot change campaign GPU-hour budget')
 deadline=previous.get('wall_deadline_epoch') if previous else time.time()+float(plan.get('wall_hours',16))*3600
 if any(time.time()+j['max_seconds']>deadline for j in jobs):raise ValueError('Job limit exceeds shared campaign wall deadline')
 if budget<=0 or maximum/3600>budget+1e-9:raise ValueError('Sum of per-job limits exceeds aggregate GPU-hour budget')
 if not a.supervisor.is_file() or not a.python.is_file():raise FileNotFoundError('Python or supervisor missing')
 if a.dry_run:print(json.dumps(dict(jobs=len(jobs),gpus=sorted(assigned),maximum_gpu_hours=maximum/3600,budget_gpu_hours=budget),indent=2));return
 a.root.mkdir(parents=True,exist_ok=True);manifest=previous or dict(created=now(),supervisor=str(a.supervisor),max_gpu_hours=budget,wall_deadline_epoch=deadline,jobs=[],batches=[])
 manifest['batches'].append(dict(time=now(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),jobs=[j['name'] for j in jobs]));manifest['jobs'].extend(jobs);atomic(a.root/'manifest.json',manifest)
 for j in jobs:
  env=os.environ.copy();env.update(plan.get('env',{}));env.update(j.get('env',{}));env['CUDA_VISIBLE_DEVICES']=j['gpu_uuid'];env['LOOKCLOSER_GPU_INDEX']=j['gpu_uuid'];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
  command=[str(a.python),str(a.supervisor),'supervise','--root',j['control_root'],'--hours',str(j['max_seconds']/3600),'--task-memory-gib',str(j.get('task_memory_gib',24)),'--reserve-gib',str(j.get('reserve_gib',16)),'--disk-reserve-gib',str(j.get('disk_reserve_gib',20)),'--']+j['command']
  with (a.root/(j['name']+'_supervisor.log')).open('w') as log:worker=subprocess.Popen(command,env=env,cwd=j.get('cwd'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  j.update(controller_pid=worker.pid,launched_epoch=time.time());atomic(a.root/'manifest.json',manifest)
 print('Launched %d bounded independent workers; one-command status: %s %s status --root %s'%(len(jobs),sys.executable,__file__,a.root))

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 s=sub.add_parser('status');s.add_argument('--root',type=Path,required=True);s.add_argument('--json',action='store_true');s.add_argument('--active-only',action='store_true',help='Compact console output; JSON/record retain all experiments');s.add_argument('--record',action='store_true',help='Append this complete supervision snapshot to the campaign log')
 l=sub.add_parser('launch');l.add_argument('--root',type=Path,required=True);l.add_argument('--plan',type=Path,required=True);l.add_argument('--python',type=Path,required=True);l.add_argument('--supervisor',type=Path,required=True);l.add_argument('--dry-run',action='store_true');l.add_argument('--append',action='store_true')
 b=sub.add_parser('archive');b.add_argument('--root',type=Path,required=True);b.add_argument('--remote',required=True)
 a=p.parse_args()
 if a.action=='launch':launch(a)
 elif a.action=='status':
  d=status(a.root)
  if a.record:
   a.root.mkdir(parents=True,exist_ok=True)
   with (a.root/'supervision_checks.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
  print(json.dumps(d)) if a.json else display(d,a.active_only)
 else:
  d=status(a.root)
  if not d['jobs'] or any(j['state'] not in ['complete','failed','setup-failed'] for j in d['jobs']):raise RuntimeError('Archive requires closed controller receipts for every job')
  subprocess.run(['rclone','copy',str(a.root),a.remote,'--immutable'],check=True)
if __name__=='__main__':main()
