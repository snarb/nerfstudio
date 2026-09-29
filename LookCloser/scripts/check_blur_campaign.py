"""Record controller/worker liveness, progress, GPU memory and OOM evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import psutil
from supervise_blur_campaign import consumed_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    processes = []
    for process in psutil.process_iter(['pid','ppid','cmdline','status']):
        command = process.info['cmdline'] or []
        if len(command)<2: continue
        script = Path(command[1]).name
        if script not in {'run_blur_experiment.py','supervise_blur_campaign.py'}: continue
        processes.append(process.info)
    runs = []
    for process in processes:
        if Path(process['cmdline'][1]).name!='run_blur_experiment.py':continue
        request = json.loads(Path(process['cmdline'][2]).read_text())
        output = Path(request['output'])
        progress = output/'progress.json'
        history = output/'history.json'
        last = json.loads(history.read_text())[-1] if history.exists() else None
        runs.append(dict(run=output.name,pid=process['pid'],controller=process['ppid'],
             controller_alive=any(p['pid']==process['ppid'] for p in processes),
             worker_status=process['status'],
             progress=json.loads(progress.read_text()) if progress.exists() else None,
             last_eval=None if last is None else {k:last[k] for k in ['step','eval_all_psnr','eval_all_ssim','eval_all_lpips']},
             oom='out of memory' in (output/'stdout.log').read_text(errors='replace').lower()))
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory',
                          '--format=csv,noheader'],capture_output=True,text=True,check=True)
    record = dict(time=time.time(),processes=processes,runs=runs,gpu=gpu.stdout.strip(),
                  charged_hours=consumed_seconds(args.root)/3600)
    with (args.root/'manual_checks.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    compact = [dict(run=r['run'],step=(r['progress'] or {}).get('step'),
                    eval_psnr=None if r['last_eval'] is None else round(r['last_eval']['eval_all_psnr'],3),
                    controller_alive=r['controller_alive'],oom=r['oom']) for r in runs]
    print(json.dumps(dict(runs=compact,gpu=record['gpu'],charged_hours=round(record['charged_hours'],3))))


if __name__ == '__main__': main()
