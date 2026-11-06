"""Preserve evaluated checkpoints on their host and fetch review artifacts durably."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)

def archive(branch,campaign,archive_name='semantic_actor_progress'):
    run=branch/'run';out=campaign/'independent_occupancy'/archive_name;out.mkdir(exist_ok=True)
    rows=[]
    for renders in sorted((run/'renders').glob('step-*')):
        step=int(renders.name.split('-')[1]);dest=out/renders.name;receipt=dest/'archive_receipt.json'
        if receipt.exists():rows.append(json.loads(receipt.read_text()));continue
        grid=run/'occupancy'/f'{renders.name}_grid.json';cp=run/'checkpoints'/f'{renders.name}.ckpt'
        if not all(p.exists()for p in [renders/'metrics.json',grid,cp]):continue
        if shutil.disk_usage(out).free<5*2**30:raise RuntimeError('Archive reserve below5GiB')
        dest.mkdir(exist_ok=True)
        try:os.link(cp,dest/'checkpoint.ckpt')
        except FileExistsError:pass
        record=json.loads(grid.read_text())
        if sha(dest/'checkpoint.ckpt')!=record['checkpoint_sha256']:raise ValueError('Checkpoint differs from actual grid receipt')
        shutil.copy2(grid,dest/grid.name);shutil.copy2(grid.with_suffix('.npz'),dest/grid.with_suffix('.npz').name)
        if sha(dest/grid.with_suffix('.npz').name)!=record['grid_archive_sha256']:raise ValueError('Grid archive differs from receipt')
        shutil.copy2(run/'worker/lookcloser/frozen/config.yml',dest/'config.yml')
        shutil.copytree(renders,dest/'renders',dirs_exist_ok=True)
        row=dict(step=step,checkpoint_sha256=record['checkpoint_sha256'],grid_sha256=record['grid_archive_sha256'],source_spec_sha256=sha(branch/'spec.json'),source_manifest_sha256=sha(branch/'source_manifest.json'),source_checkpoint=str(cp),archive=str(dest),immutable_hardlink=True,archived_at=time.time())
        write(receipt,row);rows.append(row)
    progress=json.loads((run/'progress.json').read_text());pid=int(progress['pid']);alive=Path(f'/proc/{pid}').exists()
    complete=(run/'complete.json').exists()
    state=dict(time=time.time(),archives=rows,progress=progress,producer_alive=alive,progress_age=time.time()-progress['time'],complete=complete)
    write(out/'state.json',state)
    return state

def watch(a):
    dest=a.local_campaign/'independent_occupancy'/a.archive_name;dest.mkdir(exist_ok=True)
    while True:
        try:
            command=[a.remote_python,a.remote_script,'--remote-once','--branch',a.remote_branch,'--campaign',a.remote_campaign,'--archive-name',a.archive_name]
            import shlex
            r=subprocess.run(['ssh',a.host,shlex.join(command)],capture_output=True,text=True,timeout=120,check=True)
            state=json.loads(r.stdout)
            for row in state['archives']:
                sub=dest/f"step-{row['step']:09d}"
                if not(sub/'fetch_complete.json').exists():
                    sub.mkdir(exist_ok=True)
                    subprocess.run(['rsync','-a','--exclude=checkpoint.ckpt',a.host+':'+row['archive']+'/',str(sub)+'/'],check=True,timeout=180,stdout=subprocess.DEVNULL)
                    npz=sub/f"step-{row['step']:09d}_grid.npz"
                    if sha(npz)!=row['grid_sha256']:raise ValueError('Fetched actual grid hash mismatch')
                    write(sub/'fetch_complete.json',dict(time=time.time(),checkpoint_retained_remote=True,grid_sha256=row['grid_sha256']))
            write(dest/'watcher_state.json',dict(pid=os.getpid(),**state))
            with(dest/'supervision.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),step=state['progress']['step'],producer_alive=state['producer_alive'],complete=state['complete'],archives=len(state['archives'])))+'\n')
            if state['complete']:return
        except Exception as error:
            with(dest/'errors.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),error=repr(error)))+'\n')
        time.sleep(60)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--remote-once',action='store_true');p.add_argument('--branch',type=Path);p.add_argument('--campaign',type=Path)
    for name in ['host','remote-python','remote-script','remote-branch','remote-campaign']:p.add_argument('--'+name)
    p.add_argument('--local-campaign',type=Path);p.add_argument('--archive-name',default='semantic_actor_progress');a=p.parse_args()
    if a.remote_once:print(json.dumps(archive(a.branch,a.campaign,a.archive_name)))
    else:watch(a)
