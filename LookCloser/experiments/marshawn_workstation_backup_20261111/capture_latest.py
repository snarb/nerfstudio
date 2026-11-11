import ast
import json
from pathlib import Path
import sys
import subprocess

ROOT=Path('/cache/ubuntu/lookcloser/emergency_workstations_20261111')
CAMPAIGN=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109')
sys.path.insert(0,str(CAMPAIGN/'tools'))
from campaign import Host
from common import atomic

row=next(r for r in json.loads((CAMPAIGN/'fleet.json').read_text())['hosts'] if r['name']=='ws07')
host=Host(row)
code="""from pathlib import Path
import os,json,hashlib
root=Path('/home/pavlo/marshawn_campaign_20261109/frames/001108/train')
latest=json.loads((root/'latest.json').read_text())
source=root/latest['checkpoint']
target=Path('/home/nvuser/lookcloser/emergency_snapshot_20261111/filesystem')/str(source).lstrip('/')
target.parent.mkdir(parents=True,exist_ok=True)
if not target.exists():os.link(source,target)
digest=hashlib.sha256()
with target.open('rb') as stream:
 for chunk in iter(lambda:stream.read(8388608),b''):digest.update(chunk)
print(json.dumps(dict(path=str(source),size=target.stat().st_size,sha256=digest.hexdigest())))
"""
record=json.loads(host.command('python3 -',input=code,timeout=300))
folder=ROOT/'ws07'
inventory=json.loads((folder/'source_inventory.json').read_text())
inventory['records']=[r for r in inventory['records'] if '/frames/001108/train/checkpoints/' not in r['path']]
inventory['records'].append(record)
inventory['errors']=[e for e in inventory['errors'] if '/frames/001108/' not in e]
atomic(folder/'source_inventory.json',inventory)
(folder/'checkpoint_verification.json').unlink(missing_ok=True)
print(record['path'],flush=True)
subprocess.run(['sudo','systemctl','restart','marshawn-emergency-backup.service'],check=True)
