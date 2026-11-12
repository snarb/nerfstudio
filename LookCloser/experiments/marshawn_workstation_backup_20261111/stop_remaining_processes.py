from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,shlex,json
ROOT=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109');sys.path.insert(0,str(ROOT/'tools'))
from campaign import Host
from common import read
rows=[r for r in read(ROOT/'fleet.json')['hosts'] if r['name'] in ('ws07','ws08','silver0','olive','clever')]
def clean(row):
 h=Host(row)
 code='''import os,signal,json
from pathlib import Path
root=ROOT;killed=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name) in (os.getpid(),os.getppid()):continue
 try:
  args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  scripts={'production.py','worker.py','worker_background.py','agent.py','producer.py','parts.py','qualify_parts.py','native_transfer.py','render_fullhd.py','dense_scheduler.py','campaign.py'}
  owned=any(Path(x).name in scripts for x in args[1:]) and any(root in x for x in args)
  transport=bool(args and Path(args[0]).name=='tar' and '-C' in args and args.index('-C')+1<len(args) and args[args.index('-C')+1].startswith(root+'/'))
  if owned or transport:os.kill(int(p.name),signal.SIGKILL);killed.append(int(p.name))
 except (OSError,ValueError):pass
print(json.dumps(dict(host=HOST,killed=killed)))
'''.replace('root=ROOT','root='+repr(str(h.root))).replace('host=HOST','host='+repr(row['name']))
 return h.command('python3 -c '+shlex.quote(code))
with ThreadPoolExecutor(5) as pool:
 for result in pool.map(clean,rows):print(result.strip(),flush=True)
