"""CPU-only byte-exact silver staging; never launches a controller or GPU job."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from prepare_marshawn_lc_semantic_capture import CAMPAIGN,sha,write

SC=Path('/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106')
HOST='ubuntu@silver-sable'


def ssh(code,args=(),stdin=None):
    command=shlex.join(['python3','-c',code,*map(str,args)])
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,command],
        stdin=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)


INVENTORY=r'''
import sys,json,hashlib,os
from pathlib import Path
data=json.load(sys.stdin)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
rows=[]
for r in data:
 p=Path(r['remote']);state='missing'
 if p.exists():state='verified'if p.is_file()and sha(p)==r['sha256']else'CONFLICT'
 rows.append(dict(**r,state=state))
print(json.dumps(rows))
'''

UPLOAD=r'''
import sys,hashlib,os,uuid,json
from pathlib import Path
p=Path(sys.argv[1]);expected=sys.argv[2];p.parent.mkdir(parents=True,exist_ok=True)
temporary=p.with_name(p.name+'.stagepartial.'+uuid.uuid4().hex);digest=hashlib.sha256()
with temporary.open('xb')as f:
 for block in iter(lambda:sys.stdin.buffer.read(8<<20),b''):f.write(block);digest.update(block)
 f.flush();os.fsync(f.fileno())
if digest.hexdigest()!=expected:raise ValueError('Upload SHA mismatch: '+str(p))
if p.exists():
 prior=hashlib.sha256(p.read_bytes()).hexdigest()
 if prior!=expected:raise ValueError('Never replace conflicting existing bytes: '+str(p))
else:
 os.link(temporary,p);os.chmod(p,0o444)
temporary.unlink()
print(json.dumps(dict(path=str(p),sha256=expected,verified=True)))
'''

CP_ALIAS=r'''
import sys,json,hashlib,os
from pathlib import Path
source,target=map(Path,sys.argv[1:3]);expected=sys.argv[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert sha(source)==expected
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():assert sha(target)==expected
else:os.link(source,target)
print(json.dumps(dict(source=str(source),alias=str(target),sha256=sha(target),same_inode=os.stat(source).st_ino==os.stat(target).st_ino,source_mode_unchanged=True,bytes_transferred=0)))
'''


def main(a):
    root=a.root;root.mkdir(exist_ok=False)
    head=CAMPAIGN/'frozen_LC_semantic_head_plan_v1'
    release=json.loads((head/'release_TEMPLATE.json').read_text())
    bindings=dict(release['bound_files'])
    labels=json.loads((head/'label_qualification.json').read_text())
    for query in labels['unknown_queries']:
        assert sha(query['qualification'])==query['qualification_sha256']
        bindings[query['qualification']]=query['qualification_sha256']
    extras=[head/'release_TEMPLATE.json',head/'CPU_readiness.json',*list((head/'specs').glob('*.json'))]
    capture=CAMPAIGN/'frozen_LC_semantic_capture_plan_v4'
    extras += [capture/'CPU_cache_audit.json',capture/'terminal_supervision.json',
        capture/'code/marshawn_tuning_controller.py']
    for p in extras:bindings[str(p)]=sha(p)
    rows=[]
    for path,digest in bindings.items():
        p=Path(path);assert p.is_relative_to(CAMPAIGN);assert sha(p)==digest
        rows.append(dict(local=path,remote=str(SC/p.relative_to(CAMPAIGN)),sha256=digest,bytes=p.stat().st_size))
    write(root/'source_inventory.json',rows)
    # The source2GB checkpoint is already on silver: verify and hard-link only.
    cp=next(r for r in rows if r['local'].endswith('/frozen_LC_semantic_capture_plan_v2/local_source/checkpoint.ckpt'))
    actual=SC/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin/checkpoint.ckpt'
    proof=json.loads(ssh(CP_ALIAS,[actual,cp['remote'],cp['sha256']]).stdout)
    write(root/'silver_checkpoint_alias.json',proof)
    with(root/'source_inventory.json').open('rb')as f:status=json.loads(ssh(INVENTORY,stdin=f).stdout)
    write(root/'initial_remote_inventory.json',status)
    conflicts=[r['remote']for r in status if r['state']=='CONFLICT']
    if conflicts:raise ValueError('Conflicting immutable remote files: '+str(conflicts))
    missing=[r for r in status if r['state']=='missing']
    print(json.dumps(dict(phase='upload',files=len(rows),already_verified=len(rows)-len(missing),missing=len(missing),upload_bytes=sum(r['bytes']for r in missing),CP_bytes_transferred=0,GPU_started=False)),flush=True)
    start=time.time();done=[]
    def upload(row):
        with Path(row['local']).open('rb')as f:result=json.loads(ssh(UPLOAD,[row['remote'],row['sha256']],stdin=f).stdout)
        return result
    with ThreadPoolExecutor(max_workers=8)as pool:
        futures={pool.submit(upload,row):row for row in missing}
        for future in as_completed(futures):
            result=future.result();done.append(result)
            write(root/'upload_progress.json',dict(completed=len(done),total=len(missing),elapsed_seconds=time.time()-start,last=result['path'],GPU_started=False))
            if len(done)%10==0:print(json.dumps(dict(uploaded_files=len(done),total=len(missing),elapsed_seconds=time.time()-start)),flush=True)
    with(root/'source_inventory.json').open('rb')as f:final=json.loads(ssh(INVENTORY,stdin=f).stdout)
    if any(r['state']!='verified'for r in final):raise ValueError('Incomplete remote byte proof')
    write(root/'final_remote_inventory.json',final)
    receipt=dict(passed=True,host=HOST,remote_campaign=str(SC),source_head_release_template_sha256=sha(head/'release_TEMPLATE.json'),files_verified=len(final),original_head_bound_files=len(release['bound_files']),uploaded_files=len(missing),uploaded_bytes=sum(r['bytes']for r in missing),source_checkpoint_alias=proof,source_metadata_bytes_preserved=True,GPU_started=False,head_training_started=False,local_contracts_changed=False)
    write(root/'stage_receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=CAMPAIGN/'frozen_LC_semantic_head_silver_stage_v1');main(p.parse_args())
