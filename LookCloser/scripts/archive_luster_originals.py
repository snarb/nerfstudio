"""Keep calibrated JPG sources local; archive redundant HD originals after audit."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from prepare_luster_video import write


def verify_archived_originals(data):
    record=json.loads((Path(data)/'original_hd_archive.json').read_text())
    code='''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);names=json.loads(sys.argv[2]);out={}
for name in names:
 h=hashlib.sha256()
 with (root/name).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 out[name]=h.hexdigest()
print(json.dumps(out))
'''
    actual=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',record['host'],
           'python3 -c '+shlex.quote(code)+' '+shlex.quote(record['remote_dir'])+' '+shlex.quote(json.dumps(list(record['files'])))],text=True))
    expected={k:v['sha256'] for k,v in record['files'].items()}
    if actual!=expected:raise ValueError('Archived HD originals differ from the preprocessing audit')
    return record['files']


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--host',default='ubuntu@dev3')
    p.add_argument('--remote-root',default='/fsx/tmp/luster/lookcloser_video_000470_000529_20260930')
    args=p.parse_args();frames=json.loads((args.root/'manifest.json').read_text())['frames']
    while True:
        done=0
        for frame in frames:
            data=args.root/'frames'/frame/'data';receipt=data/'original_hd_archive.json'
            if receipt.exists() and not (data/'original_hd').exists():done+=1;continue
            if not (data/'audit_preprocessing.json').exists():continue
            destination=Path(args.remote_root)/'artifacts/frames'/frame/'data/original_hd'
            code='from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)'
            subprocess.run(['ssh','-o','BatchMode=yes',args.host,'python3 -c '+shlex.quote(code)+' '+shlex.quote(str(destination))],check=True)
            subprocess.run(['rsync','-a',str(data/'original_hd')+'/',f'{args.host}:{destination}/'],check=True)
            derived=json.loads((data/'derived_manifest.json').read_text());meta=json.loads((data/'transforms.json').read_text())
            files={Path(row['file_path']).name:dict(sha256=derived['original_hd/'+Path(row['file_path']).name],size=[row['w'],row['h']]) for row in meta['frames']}
            write(receipt,dict(host=args.host,remote_dir=str(destination),files=files))
            verify_archived_originals(data)
            record=json.loads(receipt.read_text());record['verified_at']=time.time();write(receipt,record)
            shutil.rmtree(data/'original_hd');done+=1
            print(json.dumps(dict(frame=frame,archived_originals=165,done=done,total=len(frames))),flush=True)
        write(args.root/'originals_archive_progress.json',dict(done=done,total=len(frames),time=time.time()))
        if done==len(frames):break
        time.sleep(30)


if __name__=='__main__':main()
