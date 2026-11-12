from pathlib import Path
import json,shlex,subprocess,sys,time
ROOT=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109')
BACKUP=ROOT.parent.parent/'emergency_workstations_20261111'
sys.path.insert(0,str(ROOT/'tools'))
from campaign import Host
from common import atomic,read,sha
host=Host(next(r for r in read(ROOT/'fleet.json')['hosts'] if r['name']=='ws07'))
settings=host.get(host.root/'settings.json');source=Path(settings['seed_root'])
code='from pathlib import Path;import hashlib,json;r=Path('+repr(str(source))+');print(json.dumps([dict(frame=f,sha256=hashlib.sha256((r/f"{f:06}/sparse/text/points3D.txt").read_bytes()).hexdigest()) for f in range(908,1269)]))'
records=json.loads(host.command('python3 -c '+shlex.quote(code),timeout=120))
atomic(BACKUP/'seed361_inventory.json',dict(records=records))
target=ROOT/'seed_points';target.mkdir(exist_ok=True)
process=subprocess.Popen(host.argv('tar -C '+shlex.quote(str(source))+' -czf - .'),stdout=subprocess.PIPE)
receiver=subprocess.run(['tar','-C',str(target),'-xzf','-'],stdin=process.stdout);process.stdout.close()
if receiver.returncode or process.wait():raise RuntimeError('Seed transfer failed')
for r in records:
 if sha(target/f'{r["frame"]:06}/sparse/text/points3D.txt')!=r['sha256']:raise ValueError('Seed checksum differs')
atomic(BACKUP/'seed361_complete.json',dict(time=time.time(),verified=len(records),records=records,path=str(target)))
print('SEED361_VERIFIED',len(records),flush=True)
