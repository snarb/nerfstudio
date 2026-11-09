#!/usr/bin/env python3
"""Read-only worker/ownership/GPU/OOM monitor for a multi-host frequency campaign."""
import argparse,json,time
from pathlib import Path

def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();r=a.campaign
 while True:
  status=read(r/'status.json');health=read(r/'health.json');maps=sorted(p for p in (r/'maps').glob('*.pt') if p.with_suffix('.json').exists());logs={p.name:p.read_text(errors='replace')[-30000:] for p in r.glob('*worker*.log')};states={}
  for worker in (status or {}).get('workers',[]):
   proc=Path('/proc')/str(worker['pid'])/'status';states[worker['name']]=proc.read_text().split('State:')[1].splitlines()[0].strip() if proc.exists() else 'exited'
  v=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),complete_pairs=len(maps),main_status=status,main_health=health,local_or_ssh_process_states=states,clever_first=read(r/'clever_relay/state.json'),clever_second=read(r/'clever_relay_second/state.json'),clever_first_lease=read(r/'clever_relay/lease.json'),clever_second_lease=read(r/'clever_relay_second/lease.json'),oom_evidence={n:'out of memory' in t.lower() for n,t in logs.items()},worker_exceptions={n:'Traceback' in t for n,t in logs.items()},newest_complete=[p.name for p in sorted(maps,key=lambda p:p.stat().st_mtime)[-4:]])
  with (r/'supervision_consolidated.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
  tmp=r/'supervision_current.json.tmp';tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(r/'supervision_current.json')
  if len(maps)==63:break
  time.sleep(30)
if __name__=='__main__':main()
