"""Read-only, CPU-only native arm artifact mirror; never copies checkpoints."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time
from marshawn_native_crop_queue import read,write,sha,config,call
from marshawn_native_crop_arms import rebind

REMOTE = """from pathlib import Path
import json,hashlib,sys
r=Path(sys.argv[1]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={'metadata':{},'gates':[]}
for n in ['progress.json','runtime.json','metrics.json','latest.json','cold_setup_audit.json','cold_provenance.json','resume_setup_audit.json','stage_paused.json','checkpoint_inventory.json']:
 p=r/n
 if p.exists():out['metadata'][n]={'bytes':p.read_text(),'sha256':sha(p)}
ledger=json.loads((r/'checkpoint_inventory.json').read_text())if(r/'checkpoint_inventory.json').exists()else{}
for p in sorted((r/'renders').glob('step-*')):
 m=p/'metrics.json'
 if not m.exists():continue
 step=int(p.name.split('-')[1]);metric=json.loads(m.read_text());cp=ledger.get(str(step))
 if metric.get('step')!=step or not cp:continue
 files={q.name:sha(q)for q in p.iterdir()if q.is_file()and q.suffix in ['.png','.json']}
 out['gates'].append({'step':step,'directory':str(p),'metrics_sha256':sha(m),'files_sha256':files,'checkpoint_inventory_entry':cp})
print(json.dumps(out))
"""


def once(root,matrix_path):
 root=Path(root);c=config(root);matrix=read(matrix_path);base=root/'review/arms';base.mkdir(parents=True,exist_ok=True);counts={}
 for arm in matrix['arms']:
  host=c['hosts'][arm['host']];spec=read(arm['spec']);output=Path(rebind(spec['output'],root,Path(host['root'])))if Path(host['root'])!=root else Path(spec['output']);destination=base/arm['name'];destination.mkdir(exist_ok=True)
  try:proof=json.loads(call(host,[host['python'],'-c',REMOTE,output],timeout=90).stdout)
  except Exception as e:counts[arm['name']]=dict(error=str(e));continue
  # Mutable progress is explicitly a live observation, not a coherent model snapshot.
  metadata=destination/'live_metadata';metadata.mkdir(exist_ok=True)
  for name,row in proof['metadata'].items():
   if __import__('hashlib').sha256(row['bytes'].encode()).hexdigest()!=row['sha256']:raise ValueError('Transported metadata differs')
   t=metadata/(name+'.tmp');t.write_text(row['bytes']);t.replace(metadata/name)
  done=[]
  for gate in proof['gates']:
   folder=destination/'renders'/('step-%09d'%gate['step']);receipt=folder/'mirror_receipt.json';prior=read(receipt,{})
   if prior.get('source_files_sha256')==gate['files_sha256']:
    if all(sha(folder/n)==digest for n,digest in gate['files_sha256'].items()):done.append(gate['step']);continue
   folder.mkdir(parents=True,exist_ok=True)
   source=gate['directory']+'/'
   if host.get('address')and Path(host['root'])!=root:source=host['address']+':'+source
   subprocess.run(['rsync','-a','--include=*.png','--include=*.json','--exclude=*',source,str(folder)+'/'],check=True,capture_output=True,timeout=180)
   if not all(sha(folder/n)==digest for n,digest in gate['files_sha256'].items()):raise ValueError('Native gate PNG/metric checksum differs')
   write(receipt,dict(kind='SOURCE_BOUND_NATIVE_ARM_GATE_MIRROR_V1',arm=arm['name'],source_host=arm['host'],source_output=str(output),step=gate['step'],source_files_sha256=gate['files_sha256'],source_checkpoint_inventory_entry=gate['checkpoint_inventory_entry'],checkpoint_payload_not_copied=True,spec_sha256=sha(arm['spec']),no_resize=True,time=time.time()));done.append(gate['step'])
  counts[arm['name']]=dict(gates=done,source_host=arm['host'],live_metadata_files=len(proof['metadata']))
 write(base/'mirror_status.json',dict(time=time.time(),arms=counts,CPU_only=True,no_GPU_or_checkpoint_transport=True))
 return counts


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args();O=a.root/'ops';lock=(O/'artifact_mirror.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 write(O/'artifact_mirror_identity.json',dict(pid=os.getpid(),start=time.time(),source_sha256=sha(__file__),interval_seconds=60))
 while not(a.root/'STOP.json').exists():
  try:
   if a.matrix.exists():once(a.root,a.matrix)
  except Exception as e:write(O/'artifact_mirror_last_error.json',dict(time=time.time(),error=str(e),scientific_jobs_untouched=True))
  if a.once:return
  time.sleep(60)
if __name__=='__main__':main()
