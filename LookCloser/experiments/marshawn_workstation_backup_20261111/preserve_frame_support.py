from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

ROOT=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109')
BACKUP=ROOT.parent.parent/'emergency_workstations_20261111'
sys.path.insert(0,str(ROOT/'tools'))
from campaign import Host
from common import atomic,read,sha

def preserve(item):
    row,frame=item
    host=Host(row)
    source=host.root/'frames'/f'{frame:06d}'
    archive=ROOT/'external_models'/row['name']/f'{frame:06d}'
    archive.mkdir(parents=True,exist_ok=True)
    receipt=archive/'support_preserved.json'
    if receipt.exists():return read(receipt)
    command=['tar','-C',str(source),'--exclude=*.ckpt','--exclude=*.png','--exclude=*.jpg','--exclude=*.log','--exclude=linear','--exclude=train/checkpoints','--exclude=train/worker','--exclude=events.out.tfevents.*','-cf','-','.']
    process=subprocess.Popen(host.argv(shlex.join(command)),stdout=subprocess.PIPE)
    receiver=subprocess.run(['tar','-C',str(archive),'-xf','-'],stdin=process.stdout)
    process.stdout.close()
    if receiver.returncode or process.wait():raise RuntimeError('Frame support transfer failed')
    for link in archive.rglob('*'):
        if not link.is_symlink():continue
        original=source/link.relative_to(archive)
        target=Path(os.readlink(link))
        if not target.is_absolute():target=Path(os.path.normpath(str(original.parent/target)))
        try:relative=target.relative_to(source)
        except ValueError:continue
        destination=archive/relative
        link.unlink()
        link.symlink_to(os.path.relpath(destination,link.parent))
    native=ROOT/'frames'/f'{frame:06d}'/'native'
    if not (native/'native_ready.json').exists():
        command=['tar','-C',str(source/'native'),'--exclude=linear','-cf','-','.']
        process=subprocess.Popen(host.argv(shlex.join(command)),stdout=subprocess.PIPE)
        receiver=subprocess.run(['tar','-C',str(archive/'native'),'-xf','-'],stdin=process.stdout)
        process.stdout.close()
        if receiver.returncode or process.wait():raise RuntimeError('Additional completed-frame RGB transfer failed')
    for name in ['images','masks']:
        destination=archive/'native'/name
        if destination.is_dir() and not destination.is_symlink() and not any(destination.iterdir()):destination.rmdir()
        if not destination.exists() and not destination.is_symlink():destination.symlink_to(native/name,target_is_directory=True)
    manifest=archive/'model/manifest.json'
    if manifest.exists():
        info=read(manifest)
        original_backup=BACKUP/row['name']/'filesystem'/str(source/'model/inference.ckpt').lstrip('/')
        model=archive/'model/inference.ckpt'
        if original_backup.is_file() and sha(original_backup)==info['sha256']:
            if not model.exists():os.link(original_backup,model)
        elif not model.exists():
            shell=BACKUP/('ssh_'+row['name'])
            subprocess.run(['rsync','-az','--partial','-e',str(shell),row['ssh']+':'+str(source/'model/inference.ckpt'),str(model)],check=True)
        if sha(model)!=info['sha256']:raise ValueError('Preserved model checksum differs')
    result=dict(frame=frame,host=row['name'],time=time.time(),path=str(archive),weights_present=(archive/'model/inference.ckpt').is_file())
    atomic(receipt,result)
    print('FRAME_SUPPORT_PRESERVED',row['name'],frame,flush=True)
    return result

rows=[r for r in read(ROOT/'fleet.json')['hosts'] if r['name'] in ['ws07','ws08']]
items=[]
for row in rows:
    code='from pathlib import Path;import json;root=Path('+repr(str(Path(row['root'])/'frames'))+');print(json.dumps([int(p.name) for p in root.iterdir() if (p/"train/spec.json").is_file()]))'
    for frame in json.loads(Host(row).command('python3 -c '+shlex.quote(code),timeout=60)):
        if 908<=frame<=1268:items.append((row,frame))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(preserve,items))
atomic(BACKUP/'frame_support_complete.json',dict(time=time.time(),records=results))
