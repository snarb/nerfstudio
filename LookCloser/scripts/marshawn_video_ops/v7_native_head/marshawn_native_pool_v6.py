"""Opportunistic qualified native pool; field reservations take launch priority."""
from pathlib import Path
import copy,json,subprocess,shlex,time
import marshawn_sam3_assist as io

PROBE="""import json,fcntl,os,subprocess,csv
rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines(),skipinitialspace=True)); p=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines(),skipinitialspace=True));out={}
for r in rows:
 f=open('/tmp/lookcloser_tuning_gpu_%s_%s.lock'%(os.getuid(),r[0]),'a+');free=True
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:free=False
 out[r[0]]={'uuid':r[1],'lease_free':free,'processes':[int(x[1])for x in p if x and x[0]==r[1]]};f.close()
print(json.dumps(out))
"""

def field_policy(V):
 p=io.read(Path(V)/'operations/fleet10_active.json');state=io.read(p['state']);blocked=set();waiting=[]
 for frame,row in state.get('frames',{}).items():
  if row.get('host')is not None and row.get('gpu')is not None and state.get('jobs',{}).get(row.get('name'),{}).get('status')!='complete':blocked.add((row['host'],int(row['gpu'])))
 for path in Path(p['assignments']).glob('*.json'):
  row=io.read(path);f=state.get('frames',{}).get(str(row['frame']),{});j=state.get('jobs',{}).get(f.get('name'),{})
  if j.get('status')=='complete':continue
  if row.get('host')is not None and row.get('gpu')is not None:blocked.add((row['host'],int(row['gpu'])))
 for path in Path(p['demands']).glob('*.json'):
  row=io.read(path)
  if row.get('status')not in ('assigned','complete','retired','cancelled'):
   if row.get('preferred_host')is not None and row.get('preferred_gpu')is not None:blocked.add((row['preferred_host'],int(row['preferred_gpu'])))
   else:waiting.append(row['frame'])
 return blocked,waiting

def available(V,hosts,state=None):
 blocked,waiting=field_policy(V);result={};proof=dict(time=time.time(),blocked=sorted([list(x)for x in blocked]),waiting=waiting,hosts={})
 for key,host in hosts.items():
  argv=[host['python'],'-c',PROBE]
  if host.get('address'):argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host['address'],shlex.join(argv)]
  try:
   r=subprocess.run(argv,capture_output=True,text=True,timeout=25);r.check_returncode();snap=json.loads(r.stdout);proof['hosts'][key]=snap
  except Exception as e:proof['hosts'][key]={'error':str(e)};result[key]=[];continue
  own={int(j['gpu'])for j in (state or {}).get('jobs',{}).values()if j.get('host')==key and j.get('gpu')is not None and j.get('status')not in ('complete','failed','blocked','skipped','pending')}
  result[key]=[g for g in host['gpus']if(key,int(g))not in blocked and not waiting and (int(g)in own or (snap.get(str(g),{}).get('lease_free')and not snap[str(g)]['processes']))]
 return result,proof

def choose(allowed,loads):
 options=[(loads.get(h,0)/len(g),h)for h,g in allowed.items()if g]
 return min(options)[1]if options else None
