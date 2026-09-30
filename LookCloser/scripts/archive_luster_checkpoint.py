"""Archive a campaign-owned checkpoint through dev3 before releasing local cache."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from prepare_luster_video import write


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):result.update(chunk)
    return result.hexdigest()


def archive(root, checkpoint, remote_root, host='ubuntu@dev3', release=False):
    root=Path(root).resolve();checkpoint=Path(checkpoint).resolve()
    relative=checkpoint.relative_to(root)
    if 'runs' not in relative.parts or checkpoint.suffix!='.ckpt':raise ValueError('Only campaign training checkpoints may be released')
    destination=Path(remote_root)/'artifacts'/relative
    receipt=checkpoint.with_suffix('.archive.json')
    if checkpoint.exists():
        digest=sha(checkpoint);size=checkpoint.stat().st_size
        code='from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)'
        subprocess.run(['ssh','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(code)+' '+shlex.quote(str(destination.parent))],check=True)
        subprocess.run(['rsync','-a','--partial',str(checkpoint),f'{host}:{destination}'],check=True)
    elif receipt.exists():
        previous=json.loads(receipt.read_text());digest=previous['sha256'];size=previous['bytes']
    else:raise FileNotFoundError(checkpoint)
    code='''import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
print(json.dumps(dict(sha256=h.hexdigest(),bytes=p.stat().st_size)))
'''
    result=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(code)+' '+shlex.quote(str(destination))],text=True))
    if result!={'sha256':digest,'bytes':size}:raise ValueError('Archived checkpoint differs; local file retained')
    record=dict(local_path=str(checkpoint),host=host,remote_path=str(destination),sha256=digest,bytes=size,verified_at=time.time(),local_released=release)
    write(receipt,record)
    if release:checkpoint.unlink(missing_ok=True)
    return record


def restore(checkpoint):
    checkpoint=Path(checkpoint)
    if checkpoint.exists():
        receipt=checkpoint.with_suffix('.archive.json')
        if receipt.exists() and sha(checkpoint)!=json.loads(receipt.read_text())['sha256']:
            raise ValueError('Local checkpoint differs from its archive receipt')
        return checkpoint
    record=json.loads(checkpoint.with_suffix('.archive.json').read_text())
    temporary=checkpoint.with_suffix('.restore-partial')
    subprocess.run(['rsync','-a','--checksum','--partial',f'{record["host"]}:{record["remote_path"]}',str(temporary)],check=True)
    if sha(temporary)!=record['sha256']:
        temporary.unlink()
        raise ValueError('Restored checkpoint differs from archive receipt')
    temporary.replace(checkpoint)
    record.update(local_released=False,restored_at=time.time());write(checkpoint.with_suffix('.archive.json'),record)
    return checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('checkpoint',type=Path)
    p.add_argument('--remote-root',required=True);p.add_argument('--host',default='ubuntu@dev3');p.add_argument('--release',action='store_true')
    args=p.parse_args();print(json.dumps(archive(args.root,args.checkpoint,args.remote_root,args.host,args.release)))


if __name__=='__main__':main()
