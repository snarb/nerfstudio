#!/usr/bin/env python3
"""Compact, recorded health check for the Marshawn campaign on authorized hosts."""
import argparse,datetime,json,subprocess,os,time
from pathlib import Path

def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=25)
 return {'returncode':p.returncode,'output':(p.stdout+p.stderr)[-12000:]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd'));p.add_argument('--campaign-dir',type=Path);a=p.parse_args();campaign=a.campaign_dir or a.root/'campaign'
 shell="ps -eo pid,ppid,etime,args | rg 'fit_marshawn_frequency_queue.py|relay_marshawn_clever_frequencies.py|run_marshawn_hd_campaign.py|ns-train|ns-eval' | rg -v 'rg |bash -c|ssh '; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"
 record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local':command(['bash','-c',shell]),'dev3':command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','ubuntu@dev3',shell]),'maps':len(list((a.root/'data/lookcloser_frequencies').glob('*.pt')))}
 logs=list(a.root.glob('frequency_queue*.log'))+list((campaign/'logs').glob('*.log'))+list(campaign.rglob('train_stdout.log'))
 record['logs']={str(f):{'tail':f.read_text(errors='replace').splitlines()[-2:],'error_evidence':[x for x in f.read_text(errors='replace').splitlines() if any(y in x.lower() for y in ['out of memory','cuda error','traceback','permissionerror'])][-4:]} for f in logs}
 record['campaign_root']=str(campaign)
 record['variants']={f.stem:json.loads(f.read_text()) for f in (campaign/'status').glob('*.json')}
 clever=a.root/'clever_frequency_relay/state.json'
 if clever.exists():
  record['clever']=json.loads(clever.read_text());record['clever']['age_seconds']=time.time()-record['clever'].get('time',0)
 with (a.root/'supervision.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 print(json.dumps({'utc':record['utc'],'maps':record['maps'],'host_status':{k:record[k]['returncode'] for k in ['local','dev3']},'variants':{k:v['status'] for k,v in record['variants'].items()}}))
if __name__=='__main__':main()
