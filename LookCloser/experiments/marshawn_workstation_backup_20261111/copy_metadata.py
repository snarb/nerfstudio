from pathlib import Path
import concurrent.futures
import json
import subprocess
import sys
import time

ROOT=Path('/cache/ubuntu/lookcloser/emergency_workstations_20261111')
CAMPAIGN=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109')
sys.path.insert(0,str(CAMPAIGN/'tools'))
from campaign import Host
from common import atomic

def copy(row):
    host=Host(row)
    folder=ROOT/row['name']
    code='from pathlib import Path;import json;root=Path('+repr(str(host.root))+');paths=list(root.glob("*.json"));'+chr(10)+'for name in ["frames","tools","production_tools","frozen_helpers","production_frozen_helpers"]:'+chr(10)+' for path in (root/name).rglob("*"):'+chr(10)+'  if path.is_file() and not path.is_symlink() and path.suffix in [".json",".yml",".yaml",".py",".sh"] and path.stat().st_size<16777216:paths.append(path)'+chr(10)+'print(json.dumps([str(p).lstrip("/") for p in paths]))'
    paths=json.loads(host.command('python3 -',input=code,timeout=120))
    listing=folder/'metadata_paths.txt'
    listing.write_text(chr(10).join(paths)+chr(10))
    log=(folder/'metadata_transfer.log').open('ab')
    command=['rsync','-az','--partial','--timeout=180','-e',str(ROOT/('ssh_'+row['name'])),'--files-from='+str(listing),row['ssh']+':/',str(folder/'filesystem')+'/']
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    missing=[name for name in paths if not (folder/'filesystem'/name).is_file()]
    if missing:raise ValueError(missing[:3])
    atomic(folder/'metadata_complete.json',dict(time=time.time(),files=len(paths),missing=missing))
    print(row['name'],'METADATA_COPIED',len(paths),flush=True)

rows=[r for r in json.loads((CAMPAIGN/'fleet.json').read_text())['hosts'] if r['name'] in ['ws07','ws08']]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
    list(workers.map(copy,rows))
