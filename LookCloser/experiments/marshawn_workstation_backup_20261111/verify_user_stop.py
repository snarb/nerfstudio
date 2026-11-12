from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys,shlex,time
ROOT=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109');sys.path.insert(0,str(ROOT/'tools'))
from campaign import Host
from common import read,atomic
rows=read(ROOT/'fleet.json')['hosts'];rows=[r for r in rows if r['name'] not in [f'silver{i}' for i in range(1,8)]]

def verify(row):
 h=Host(row)
 code='''from pathlib import Path
import json,os
root=ROOT;matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  scripts={'production.py','worker.py','worker_background.py','agent.py','producer.py','parts.py','qualify_parts.py','native_transfer.py','render_fullhd.py','dense_scheduler.py','campaign.py'}
  if any(Path(x).name in scripts for x in args[1:]) and any(root in x for x in args):matches.append(dict(pid=int(p.name),args=args))
 except (OSError,ValueError):pass
print(json.dumps(matches))
'''.replace('root=ROOT','root='+repr(str(h.root)))
 processes=json.loads(h.command('python3 -c '+shlex.quote(code)))
 units=h.command("systemctl list-units --type=service --state=running --no-legend | grep -E 'marshawn908|marshawn-sequence|marshawn923' || true").strip()
 return dict(host=row['name'],remaining_processes=processes,running_units=units)
with ThreadPoolExecutor(5) as pool:results=list(pool.map(verify,rows))
storage=read(ROOT/'sparse24_storage_audit.json');assert storage['verified']==72 and len(storage['results'])==72
assert all(r['status']=='verified' and Path(r['path']).is_file() and Path(r['path']).stat().st_size>0 for r in storage['results'])
assert not any(r['remaining_processes'] or r['running_units'] for r in results),results
receipt=dict(time=time.time(),reason='User narrowed final scope to72 sparse24 models and requested stop everywhere',hosts=results,selected_models_present=72,previous_streamed_sha_verified=72,all_training_stopped=True)
atomic(ROOT/'user_stop_verified_20261112.json',receipt)
print(json.dumps(receipt,indent=2))
