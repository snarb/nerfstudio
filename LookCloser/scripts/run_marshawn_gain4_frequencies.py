#!/usr/bin/env python3
"""Supervised fresh gain4-only train-map queue; canonical dataset stays unchanged."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['gain4','gain4_scalar'],default='gain4');p.add_argument('--workers',type=int,default=2);p.add_argument('--dev3',action='store_true');a=p.parse_args();r=a.root;task=r/('frequency_'+a.variant);task.mkdir(exist_ok=True);data=task/'data';maps=task/'maps';data.mkdir(exist_ok=True);maps.mkdir(exist_ok=True);os.chmod(maps,0o777);source=r/('color_variants/gain4_scalar_trainfit_poseeval/sdr_png' if a.variant=='gain4_scalar' else 'color_variants/gain4/sdr_png');meta=json.loads((source/'transforms.json').read_text());train=meta['train_filenames'];assert len(train)==63 and len(meta['val_filenames'])==3 and not set(train)&set(meta['val_filenames']);queue=Path(__file__).parent/'fit_marshawn_frequency_queue.py';assert sha(queue)=='6ef90cbbbd6f277886c3aaa2112d4b151f08c325c4ae5fb7b0b63da3d3b75bd7'
 for name,target in [('images',source/'images'),('masks',source/'masks'),('lookcloser_frequencies',maps)]:
  dst=data/name
  if not dst.exists():dst.symlink_to(target,target_is_directory=True)
  assert dst.resolve()==target.resolve()
 dst=data/'transforms.json'
 if dst.exists():assert dst.read_bytes()==(source/'transforms.json').read_bytes()
 else:shutil.copyfile(source/'transforms.json',dst)
 frozen=task/'fit_queue_frozen.py'
 if frozen.exists():assert sha(frozen)==sha(queue)
 else:shutil.copyfile(queue,frozen)
 contract=dict(kind='FRESH correctedgain4SDR trainfrequency fits; nooldmaps reused',data=str(data),source_data=str(source),output_maps=str(maps),source_color_gate_sha256=sha(r/'color_variants/gain4/color_gate.json'),transforms_sha256=sha(dst),fitter_sha256=sha(frozen),preprocess_sha256=sha(r/'code_native/nerfstudio/scripts/lookcloser_preprocess.py'),recipe=dict(ssim_threshold=.95,train_steps_per_level=1000,min_res=16,max_res=8192,n_levels=16,n_features=2,patch_size=8,stride=8,eval_patch_batch_size=1024),source_images={Path(n).stem:sha(source/n) for n in train})
 if a.variant=='gain4_scalar':
  contract['kind']='FRESH scalar-normalized gain4SDR trainfrequency fits; nooldmaps reused'
  contract.pop('source_color_gate_sha256')
  contract['source_color_contract_sha256']=sha(source/'color_contract.json')
  contract['source_complete_sha256']=sha(source.parent/'complete.json')
  staged=json.loads((task/'input_contract_STAGED.json').read_text())
  for key in ['source_images','source_color_contract_sha256','source_complete_sha256','transforms_sha256','fitter_sha256','preprocess_sha256','recipe']:assert staged[key]==contract[key]
  contract['source_stage_contract_sha256']=sha(task/'input_contract_STAGED.json')
 if (task/'input_contract.json').exists():assert json.loads((task/'input_contract.json').read_text())==contract
 else:write(task/'input_contract.json',contract)
 python='/home/pavlo/.conda/envs/nerfstudio/bin/python';env=dict(os.environ,PYTHONPATH=str(r/'code_native'),TORCH_CUDA_ARCH_LIST='12.0+PTX',TORCH_EXTENSIONS_DIR=str(r/'native_ngp4_corrected_sdr/local_torch_extensions_py311_cu128_sm120_ptx'));workers=[]
 def event(kind,**kw):
  with (task/'supervision.jsonl').open('a') as f:f.write(json.dumps(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw))+'\n')
 def gpu(host=None):
  cmd=['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
  if host:cmd=['ssh','-o','BatchMode=yes',host]+cmd
  raw=subprocess.check_output(cmd,text=True,timeout=20).strip();return dict(raw=raw,used_mib=int(raw.split(',')[0]))
 if gpu()['used_mib']>70000:raise RuntimeError('Local initialGPUbudget exceeded')
 if a.dev3 and gpu('ubuntu@dev3')['used_mib']>30000:raise RuntimeError('DEV3 initialGPUbudget exceeded')
 for i in range(a.workers):
  log=(task/f'local_worker{i}.log').open('a');child=subprocess.Popen([python,str(frozen),'--data',str(data)],env=env,stdout=log,stderr=subprocess.STDOUT);workers.append((f'local{i}',child,log));event('launch',worker=f'local{i}',pid=child.pid)
 if a.dev3:
  if gpu('ubuntu@dev3')['used_mib']>30000:raise RuntimeError('DEV3 initialGPUbudget exceeded')
  remoteenv='PYTHONPATH='+str(r/'compat_py38')+':'+str(r/'code_sfm')+':'+str(r/'code_sfm/LookCloser/scripts')
  cmd=['ssh','-o','BatchMode=yes','ubuntu@dev3','env',remoteenv,'/home/ubuntu/anaconda3/envs/nerfstudio/bin/python',str(frozen),'--data',str(data)];log=(task/'dev3_worker.log').open('a');child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);workers.append(('dev3',child,log));event('launch',worker='dev3',ssh_pid=child.pid)
 write(task/'status.json',dict(stage='running',controller_pid=os.getpid(),workers=[dict(name=n,pid=c.pid) for n,c,l in workers]))
 while any(c.poll() is None for n,c,l in workers):
  local=gpu();remote=gpu('ubuntu@dev3') if a.dev3 else None;done=sum((maps/(Path(n).stem+'.pt')).exists() and (maps/(Path(n).stem+'.json')).exists() for n in train);health=dict(complete_pairs=done,expected=63,local_gpu=local,dev3_gpu=remote,workers=[dict(name=n,pid=c.pid,alive=c.poll() is None,returncode=c.poll()) for n,c,l in workers]);event('health',**health);write(task/'health.json',health)
  if local['used_mib']>80000 or remote and remote['used_mib']>35000:
   # Terminate only our local queue children and the remote PID whose full
   # dataset/frozen-fitter command matches this campaign. Never touch model jobs.
   for n,c,l in workers:
    if c.poll() is None and n!='dev3':c.terminate()
   if a.dev3:
    import re
    matches=re.findall(r'start \S+ dev3 (\d+)',(task/'dev3_worker.log').read_text(errors='replace'))
    if matches:
     owned=int(matches[-1])
     code="import os,signal;from pathlib import Path;p=Path('/proc/"+str(owned)+"/cmdline');cmd=p.read_bytes().decode().replace(chr(0),' ') if p.exists() else '';assert "+repr(str(frozen))+" in cmd and "+repr(str(data))+" in cmd;os.kill("+str(owned)+",signal.SIGTERM)"
     subprocess.run(['ssh','-o','BatchMode=yes','ubuntu@dev3','/home/ubuntu/anaconda3/envs/nerfstudio/bin/python','-c',shlex.quote(code)],timeout=20,check=True)
   event('own_worker_budget_stop',**health);raise RuntimeError('Ownfrequency workers stopped for resourcebudget; foreign/modeljobs untouched')
  time.sleep(30)
 for n,c,l in workers:l.close();event('worker_exit',worker=n,returncode=c.returncode)
 write(task/'status.json',dict(stage='queue_workers_exited',complete_pairs=sum((maps/(Path(n).stem+'.pt')).exists() and (maps/(Path(n).stem+'.json')).exists() for n in train),worker_returncodes={n:c.returncode for n,c,l in workers}));event('done')
if __name__=='__main__':main()
