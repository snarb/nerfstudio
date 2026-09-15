"""Lower priority only for exact, owned dev3 native-ending worker threads."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from parallel_cinematic_6k_endings import WORK
from render_cinematic_6k_output import REMOTE,REMOTE_PYTHON,REMOTE_ROOT


def remote_code():
    return '''import os,json
from pathlib import Path
from datetime import datetime,timezone
records=[]
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  if process.stat().st_uid!=os.getuid():continue
  argv=[x.decode() for x in (process/'cmdline').read_bytes().split(b'\\0') if x]
  if len(argv)!=5 or argv[:4]!=EXPECTED:continue
  if not argv[4].startswith(ROOT+'/'):continue
  frame=argv[4][len(ROOT)+1:]
  if not (frame.isdigit() and len(frame)==6 and int(frame)%2==1 and 1151<=int(frame)<=1195):continue
  threads=[]
  for task in (process/'task').iterdir():
   tid=int(task.name)
   try:
    before=os.getpriority(os.PRIO_PROCESS,tid)
    if before<10:os.setpriority(os.PRIO_PROCESS,tid,10)
    threads.append(dict(tid=tid,before=before,after=os.getpriority(os.PRIO_PROCESS,tid)))
   except ProcessLookupError:pass
  records.append(dict(pid=int(process.name),argv=argv,threads=threads))
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),ending_workers=records)))
'''.replace('EXPECTED',repr([REMOTE_PYTHON,REMOTE_ROOT+'/render_cinematic_6k_output.py','remote','--job'])).replace('ROOT',repr(REMOTE_ROOT))


def run(worker_pid):
    command=['ssh',REMOTE,REMOTE_PYTHON+' -c '+shlex.quote(remote_code())]
    while Path(f'/proc/{worker_pid}').exists():
        record=json.loads(subprocess.check_output(command,text=True))
        record['local_ending_controller_pid']=worker_pid
        with (WORK/'scheduling_checks.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        changed=[dict(pid=r['pid'],changed_threads=sum(t['before']!=t['after'] for t in r['threads']))
            for r in record['ending_workers'] if any(t['before']!=t['after'] for t in r['threads'])]
        if changed:print(json.dumps(dict(utc=record['utc'],changes=changed)),flush=True)
        time.sleep(2)
    print('ending controller terminal; priority watcher finished',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-pid',required=True,type=int);a=p.parse_args();run(a.worker_pid)
