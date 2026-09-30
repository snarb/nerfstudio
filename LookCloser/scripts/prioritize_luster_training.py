"""Pause only this campaign's local 2D GPU workers while a field job needs the GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil
from prepare_luster_video import write


def resume(workers):
    for item in workers:
        try:
            process=psutil.Process(item['pid'])
            if process.create_time()==item['created']:process.resume()
        except psutil.NoSuchProcess:pass


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--watchdog',action='store_true');args=p.parse_args()
    root=args.root.resolve();directory=root/'gpu_priority';directory.mkdir(exist_ok=True);status_path=directory/'status.json'
    if args.watchdog:
        while True:
            try:status=json.loads(status_path.read_text())
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
            if status.get('stopped'):return
            if not psutil.pid_exists(status['pid']) or time.time()-status['time']>60:
                resume(status['paused']);write(directory/'watchdog_recovery.json',dict(time=time.time(),status=status));return
            time.sleep(10)
    paused={};write(status_path,dict(pid=os.getpid(),time=time.time(),paused=[]))
    with (directory/'watchdog.log').open('a') as log:
        subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),'--watchdog'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    def stop(*unused):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    names={'run_luster_experiment.py','export_luster_selection.py','render_luster_video_frame.py','benchmark_luster_frequencies.py'}
    last=None
    try:
        while not (directory/'stop').exists():
            fields=[]
            for process in psutil.process_iter(['pid','cmdline']):
                command=process.info['cmdline'] or []
                if any(Path(arg).name in names for arg in command) and any(arg.startswith(str(root)) for arg in command):fields.append(process.pid)
            if fields:
                try:queue=json.loads((root/'frequency_queue/local.json').read_text());parent=(queue.get('progress') or {}).get('pid')
                except (FileNotFoundError,json.JSONDecodeError):parent=None
                if parent and psutil.pid_exists(parent):
                    parent_process=psutil.Process(parent);command=parent_process.cmdline()
                    if not any(Path(arg).name=='prepare_luster_frequencies.py' for arg in command) or not any(arg.startswith(str(root)) for arg in command):
                        raise RuntimeError('Frequency parent identity changed')
                    descendants={p.pid:p for p in parent_process.children(recursive=True)}
                    gpu={int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).splitlines() if x.strip().isdigit()}
                    for pid in gpu & descendants.keys():
                        process=descendants[pid];paused[pid]=dict(pid=pid,created=process.create_time());process.suspend()
            else:
                resume(paused.values());paused={}
            status=dict(pid=os.getpid(),time=time.time(),field_pids=fields,paused=list(paused.values()))
            write(status_path,status)
            key=(tuple(fields),tuple(paused))
            if key!=last:
                with (directory/'transitions.jsonl').open('a') as log:log.write(json.dumps(status)+'\n')
                last=key
            time.sleep(5)
    finally:
        resume(paused.values());write(status_path,dict(pid=os.getpid(),time=time.time(),paused=[],stopped=True))


if __name__=='__main__':main()
