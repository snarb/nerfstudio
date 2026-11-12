from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,shlex,time
ROOT=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109');sys.path.insert(0,str(ROOT/'tools'))
from campaign import Host
from common import read,atomic
rows=read(ROOT/'fleet.json')['hosts']

def stop(row):
 h=Host(row);a=h.get(h.agent_root/'agent_state.json',{});active=a.get('active');j=a.get('jobs',{}).get(active,{})
 h.put(h.agent_root/'STOP',dict(time=time.time(),reason='User cancelled all training; only completed sparse24 needed'))
 unit='marshawn-sequence-agent.service' if row['name'] in ('ws07','ws08') else 'marshawn908-agent@'+row['name'][6:]+'.service' if row['name'].startswith('silver') else 'marshawn908-agent.service'
 # STOP marker and disabled executor prevent future jobs. KillMode=process
 # means live job process groups must also be terminated explicitly.
 h.command('sudo -n systemctl stop '+shlex.quote(unit)+'; sudo -n systemctl disable '+shlex.quote(unit)+' 2>/dev/null || true',timeout=120)
 code='''import os,signal,json
from pathlib import Path
root=ROOT
skip={os.getpid(),os.getppid()};targets=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name) in skip:continue
 try:
  args=(p/'cmdline').read_bytes().split(bytes([0]));args=[x.decode(errors='replace') for x in args if x]
  scripts={'production.py','worker.py','worker_background.py','agent.py','producer.py','parts.py','qualify_parts.py','native_transfer.py','render_fullhd.py','dense_scheduler.py','campaign.py'}
  owned=any(Path(x).name in scripts for x in args[1:]) and any(root in x for x in args)
  if owned:targets.append(int(p.name))
 except (OSError,ValueError):pass
for pid in targets:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
print(json.dumps(dict(signalled=targets)))
'''.replace('root=ROOT','root='+repr(str(h.root)))
 result=h.command('python3 -c '+shlex.quote(code),timeout=120)
 return dict(host=row['name'],active_before=active,result=json.loads(result))
results=[]
with ThreadPoolExecutor(6) as pool:
 for result in pool.map(stop,rows):results.append(result);print(json.dumps(result),flush=True)
atomic(ROOT/'user_stop_all_training_20261112.json',dict(time=time.time(),reason='User only needs72 sparse24 models; cancel remaining training',hosts=results))
