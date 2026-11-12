"""Preserve irreplaceable full-size masks and per-frame calibration, no source RGB."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,shlex,subprocess,sys,time
ROOT=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109')
BACKUP=ROOT.parent.parent/'emergency_workstations_20261111'
sys.path.insert(0,str(ROOT/'tools'))
from campaign import Host
from common import atomic,read,sha
DEST=ROOT/'source_archive/raid/lookcloser'
DEST.mkdir(parents=True,exist_ok=True)

def preserve(row):
    host=Host(row)
    code='''from pathlib import Path
import hashlib,json
base=Path('/raid/lookcloser');records=[]
for frame in range(FIRST,LAST+1):
 for p in (base/'masks8k'/f'{frame:06d}').glob('*.png'):
  records.append(dict(path=str(p.relative_to(base)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 for p in (base/'s3').glob(f'*/root/Marshawn_2_1/working/fullres_pq16/{frame:06d}/sparse/text/*.txt'):
  records.append(dict(path=str(p.relative_to(base)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(records))
'''.replace('FIRST',str(908 if row['name']=='ws07' else 1112)).replace('LAST',str(1111 if row['name']=='ws07' else 1268))
    records=json.loads(host.command('python3 -c '+shlex.quote(code),timeout=900))
    expected=(204 if row['name']=='ws07' else 157)*72
    masks=[r for r in records if r['path'].startswith('masks8k/')]
    if len(masks)!=expected:raise ValueError('Full-size mask coverage differs '+row['name'])
    atomic(BACKUP/('source_aux_'+row['name']+'_inventory.json'),dict(records=records))
    for start in range(0,len(records),864):
        batch=records[start:start+864]
        missing=[r for r in batch if not (DEST/r['path']).is_file() or sha(DEST/r['path'])!=r['sha256']]
        if missing:
            listing=BACKUP/f'source_aux_{row["name"]}_{start}.txt'
            listing.write_text(''.join(r['path']+'\n' for r in missing))
            with listing.open('rb') as incoming:
                process=subprocess.Popen(host.argv('tar -C /raid/lookcloser -czf - -T -'),stdin=incoming,stdout=subprocess.PIPE)
                receiver=subprocess.run(['tar','-C',str(DEST),'-xzf','-'],stdin=process.stdout)
                process.stdout.close()
                if receiver.returncode or process.wait():raise RuntimeError('Auxiliary source transfer failed')
        if any(sha(DEST/r['path'])!=r['sha256'] for r in batch):raise ValueError('Auxiliary source checksum differs')
        print('SOURCE_AUX_VERIFIED',row['name'],min(start+864,len(records)),len(records),flush=True)
    receipt=dict(time=time.time(),host=row['name'],verified=len(records),fullsize_masks=len(masks),source_rgb_copied=False,path=str(DEST))
    atomic(BACKUP/('source_aux_'+row['name']+'_complete.json'),receipt)
    return receipt
rows=[r for r in read(ROOT/'fleet.json')['hosts'] if r['name'] in ('ws07','ws08')]
with ThreadPoolExecutor(2) as pool:results=list(pool.map(preserve,rows))
atomic(BACKUP/'source_aux_complete.json',dict(time=time.time(),records=results))
