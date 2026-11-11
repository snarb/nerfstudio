import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path('/cache/ubuntu/lookcloser/emergency_workstations_20261111')
CAMPAIGN = Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109')
sys.path.insert(0, str(CAMPAIGN / 'tools'))
from campaign import Host

INVENTORY = """
import os,json,hashlib
from pathlib import Path
roots=[Path('/home/pavlo/marshawn_campaign_20261109'),Path('/home/nvuser/lookcloser/campaigns/marshawn908_20261109')]
records=[];errors=[];paths=set()
for root in roots:
 if not root.exists():continue
 for frame in (root/'frames').glob('*'):
  model=frame/'model/inference.ckpt'
  if model.is_file():paths.add(model)
  latest=frame/'train/latest.json'
  if latest.is_file():
   try:
    data=json.loads(latest.read_text())
    checkpoint=latest.parent/data['checkpoint']
    if not checkpoint.is_file():raise RuntimeError('latest checkpoint missing '+str(checkpoint))
    paths.add(checkpoint)
   except Exception as e:errors.append(str(e))
for path in sorted(paths):
 try:
  before=path.stat();h=hashlib.sha256()
  with path.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8388608),b''):h.update(chunk)
  after=path.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('changed during hashing '+str(path))
  records.append(dict(path=str(path),size=after.st_size,sha256=h.hexdigest()))
 except Exception as e:errors.append(str(e))
print(json.dumps(dict(records=records,errors=errors,policy='latest full checkpoint per frame plus selected inference export')))
"""

def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def run_host(row):
    host = Host(row)
    out = ROOT / row['name']
    out.mkdir(exist_ok=True)
    remote_shell = ROOT / ('ssh_' + row['name'])
    remote_shell.write_text(chr(10).join(['#!/bin/sh', 'exec ssh -o ServerAliveInterval=30 ubuntu@100.87.142.86 ssh "$@"', '']))
    remote_shell.chmod(0o700)
    log = (out / 'transfer.log').open('ab')
    inventory = json.loads((out / 'source_inventory.json').read_text()) if (out / 'source_inventory.json').exists() else json.loads(host.command('python3 -', input=INVENTORY, timeout=1800))
    snapshot='/home/nvuser/lookcloser/emergency_snapshot_20261111/filesystem'
    pin_code='from pathlib import Path;import os,json;inventory='+repr(inventory)+chr(10)+'root=Path('+repr(snapshot)+')'+chr(10)+'for record in inventory["records"]:'+chr(10)+' source=Path(record["path"]);target=root/record["path"].lstrip("/")'+chr(10)+' target.parent.mkdir(parents=True,exist_ok=True)'+chr(10)+' if not target.exists():os.link(source,target)'+chr(10)
    host.command('python3 -',input=pin_code,timeout=180)
    save(out / 'source_inventory.json', inventory)
    paths = out / 'checkpoint_paths.txt'
    paths.write_text(''.join(r['path'].lstrip('/') + chr(10) for r in inventory['records']))
    destination = out / 'filesystem'
    destination.mkdir(exist_ok=True)
    source = row['ssh'] + ':' + snapshot + '/'
    base = ['rsync','-az','--compress-level=1','--partial','--timeout=180','-e',str(remote_shell)]
    critical = out / 'critical_paths.txt'
    critical.write_text(''.join(r['path'].lstrip('/') + chr(10) for r in inventory['records'] if '/frames/' in r['path'] and '/model/' in r['path']))
    def transfer_parallel(path_list):
        entries=path_list.read_text().splitlines()
        chunks=[entries[index::8] for index in range(8)]
        def transfer_chunk(index):
            if not chunks[index]: return
            file_list=out/('transfer_'+path_list.stem+'_'+str(index)+'.txt')
            file_list.write_text(chr(10).join(chunks[index])+chr(10))
            subprocess.run(base+['--files-from='+str(file_list),source,str(destination)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as transfers:
            list(transfers.map(transfer_chunk,range(8)))
    transfer_parallel(critical)
    save(out/'critical_copied.json',dict(time=time.time()))
    large=[record for record in inventory['records'] if record['size']>4*1024**3]
    for record in large:
        chunk_root='/home/nvuser/lookcloser/emergency_snapshot_20261111/chunks/'+record['sha256']
        source_file=snapshot+'/'+record['path'].lstrip('/')
        split_code='from pathlib import Path;import subprocess;root=Path('+repr(chunk_root)+');root.mkdir(parents=True,exist_ok=True);marker=root/"split_complete";'+chr(10)+'if not marker.exists():'+chr(10)+' subprocess.run(["split","-b","536870912","-d","-a","3",'+repr(source_file)+',str(root/"part.")],check=True)'+chr(10)+' marker.touch()'
        host.command('python3 -',input=split_code,timeout=300)
        count=(record['size']+536870911)//536870912
        chunk_destination=out/'chunks'/record['sha256']
        chunk_destination.mkdir(parents=True,exist_ok=True)
        def copy_part(index):
            name='part.'+str(index).zfill(3)
            subprocess.run(base+[row['ssh']+':'+chunk_root+'/'+name,str(chunk_destination/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as parts:
            list(parts.map(copy_part,range(count)))
        target=destination/record['path'].lstrip('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.assembling')
        digest=hashlib.sha256()
        with temporary.open('wb') as output:
            for index in range(count):
                with (chunk_destination/('part.'+str(index).zfill(3))).open('rb') as part:
                    for chunk in iter(lambda:part.read(8388608),b''):
                        output.write(chunk);digest.update(chunk)
        if temporary.stat().st_size!=record['size'] or digest.hexdigest()!=record['sha256']:
            raise ValueError('Reassembled latest checkpoint checksum differs')
        temporary.replace(target)
    remaining=out/'remaining_checkpoint_paths.txt'
    remaining.write_text(''.join(record['path'].lstrip('/')+chr(10) for record in inventory['records'] if record not in large))
    transfer_parallel(remaining)
    checked=[]; errors=[]
    for record in inventory['records']:
        target=destination/record['path'].lstrip('/')
        digest=hashlib.sha256()
        try:
            with target.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8388608),b''):digest.update(chunk)
            if target.stat().st_size != record['size'] or digest.hexdigest()!=record['sha256']:
                raise ValueError('checksum differs')
            checked.append(record['path'])
        except Exception as error:
            errors.append(dict(path=record['path'],error=str(error)))
    save(out/'checkpoint_verification.json',dict(time=time.time(),verified=len(checked),errors=errors,inventory_errors=inventory['errors']))
    if errors:raise RuntimeError('Checkpoint verification failed')
    roots = (['/home/pavlo/marshawn_campaign_20261109','/home/pavlo/marshawn_runtime_base_20261109','/home/pavlo/nerfstudio/LookCloser/lookcloser_debug_outputs/marshawn_ws07_aabb']
             if row['name']=='ws07' else ['/home/nvuser/lookcloser/campaigns','/home/nvuser/lookcloser/logs'])
    for path in roots:
        subprocess.run(base+['--exclude=*.ckpt','--exclude=checkpoints/***','-R',row['ssh']+':'+path,str(destination)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
    save(out/'first_pass_complete.json',dict(time=time.time(),checkpoint_count=len(checked),roots=roots))
    print(row['name'],'COPIED',len(checked),flush=True)

rows=json.loads((CAMPAIGN/'fleet.json').read_text())['hosts']
rows=[row for row in rows if row['name'] in ['ws07','ws08']]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(run_host,row) for row in rows]
    for future in futures:future.result()
save(ROOT/'first_pass_complete.json',dict(time=time.time()))
