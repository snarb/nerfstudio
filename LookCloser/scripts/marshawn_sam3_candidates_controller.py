"""Durable source-qualified candidate-only queue, outside frozen training code."""
from pathlib import Path
import argparse,importlib,json,shlex,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import marshawn_sam3_assist as io
import marshawn_sam3_candidates as candidates


def execute(argv,**kwargs):return subprocess.run(argv,check=True,**kwargs)


def run_frame(a,frame):
    V=a.campaign.resolve();root=V/'operations/sam3_assist'/('candidates%06d'%frame);root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(V/'code/LookCloser/scripts'));fleet=importlib.import_module('marshawn_tuning_controller')
    held=fleet.lock(root/'controller.lock')
    if held is None:raise RuntimeError('Candidate controller already running for frame')
    cameras=sorted(f['physical_camera'] for f in io.read(V/'frames/001660/data/transforms.json')['frames'])
    if len(cameras)!=66 or len(set(cameras))!=66:raise ValueError('ALL66 physical cameras required')
    # Waiting for source publication consumes no GPU lease and survives Codex.
    while True:
        missing=[c for c in cameras if not (V/'sources'/str(frame)/c/'receipt.json').is_file()]
        io.write(root/'source_wait.json',dict(frame=frame,missing_cameras=missing,time=time.time()))
        if not missing:break
        time.sleep(10)
    remote=Path(a.remote_root)/'operations/sam3_assist'/('candidates%06d'%frame)
    code=Path(a.remote_root)/'code';opscode=Path(a.remote_root)/'operations/sam3_assist/code'
    host=dict(address=a.remote_host,python=a.remote_python,root=str(remote/'dispatch'),
        controller=str(code/'LookCloser/scripts/marshawn_tuning_controller.py'),gpus=a.gpus)
    execute(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.remote_host,shlex.join(['mkdir','-p',str(remote/'tasks'),str(opscode)])],timeout=30)
    # Helper publication is immutable; a view becomes eligible immediately after
    # its own inputs arrive, while later large native PNGs continue transferring.
    for module in (io,candidates):
        execute(['rsync','-rt',module.__file__,a.remote_host+':'+str(opscode)+'/'])
    state=io.read(root/'state.json') if (root/'state.json').exists() else dict(jobs={},cameras={})
    state.setdefault('poll_seconds',10 if state['jobs'] else 2)
    for camera in cameras:
        if camera not in state['cameras']:
            name='cand%06d_%s_a00'%(frame,camera);state['cameras'][camera]=dict(name=name,attempt=0,collected=False)
            state['jobs'][name]=dict(status='pending')
    def stage_views():
        for camera in cameras:
            task=root/'tasks'/camera;spec_path=task/'candidate_task.json'
            if spec_path.exists():
                spec=io.read(spec_path)
                original=V/'sources'/str(frame)/camera/'receipt.json'
                if spec['original_source_receipt_sha256']!=io.sha(original):raise ValueError('Immutable staged source changed')
                io.validate_task(spec,task)
            else:
                # A process can die before the first immutable task is sealed.
                # Only unsealed partial reference blobs may be discarded.
                blob=task/'reference.npz'
                if blob.exists():
                    import numpy as np
                    try:
                        with np.load(blob,allow_pickle=False) as z:assert set(z.files)=={'black','grey'}
                    except Exception:blob.unlink()
                candidates.pack(V,frame,camera,task,str(remote/'tasks'/camera),str(code),str(Path(a.remote_root)/'tools/sam3_overlay'))
            for attempt in range(8):
                try:
                    execute(['rsync','-rLtc',str(task)+'/',a.remote_host+':'+str(remote/'tasks'/camera)+'/'])
                    break
                except subprocess.CalledProcessError:
                    if attempt==7:raise
                    time.sleep(min(30,2**attempt))
            io.write(task/'stage_ready.json',dict(frame=frame,camera=camera,task_sha256=io.sha(spec_path)))
    pool=ThreadPoolExecutor(max_workers=1);staging=pool.submit(stage_views)
    io.write(root/'controller.json',dict(process=fleet.identity(),started=time.time()))
    while True:
        jobs=[]
        for camera,item in state['cameras'].items():
            if not (root/'tasks'/camera/'stage_ready.json').exists():continue
            jobs.append(dict(name=item['name'],host='silver',command=[a.remote_python,str(opscode/'marshawn_sam3_candidates.py'),
                'run','--spec',str(remote/'tasks'/camera/'candidate_task.json')],
                env={'SAM3_ASSIST_LEASE_RECEIPT':'{job_root}/receipt.json'},metrics_file=str(remote/'tasks'/camera/'operation_receipt.json')))
        fleet.tick(dict(schema_version=1,poll_seconds=state['poll_seconds'],hosts={'silver':host},jobs=jobs),state,root)
        if staging.done():staging.result()
        for camera,item in state['cameras'].items():
            job=state['jobs'][item['name']]
            if job['status']=='complete' and not item['collected']:
                task=root/'tasks'/camera;returned=task/'returned';(returned/'person_candidates').mkdir(parents=True,exist_ok=True)
                execute(['rsync','-rt',a.remote_host+':'+str(remote/'tasks'/camera/'operation_receipt.json'),str(returned)+'/'])
                execute(['rsync','-rt',a.remote_host+':'+str(remote/'tasks'/camera/'person_candidates')+'/',str(returned/'person_candidates')+'/'])
                candidates.collect(V,task);item['collected']=True
            elif job['status']=='failed':
                snap=state.get('host_snapshots',{}).get('silver',{});observed=snap.get('jobs',{}).get(item['name'])
                if observed and not observed.get('supervisor_alive') and not observed.get('worker_alive') and item['attempt']<3:
                    item['attempt']+=1;item['name']='cand%06d_%s_a%02d'%(frame,camera,item['attempt']);state['jobs'][item['name']]={'status':'pending'}
                elif item['attempt']>=3:raise RuntimeError('Candidate retry limit; source/identity artifacts retained')
        io.write(root/'state.json',state)
        remaining=[c for c,i in state['cameras'].items() if not i['collected']]
        io.write(root/'progress.json',dict(frame=frame,completed=66-len(remaining),remaining=remaining,time=time.time(),
            prepared_publication_allowed=False,native_mattes_published=False))
        if not remaining:
            target=V/'frames'/('%06d'%frame)/'person_candidates_sg'
            result=dict(frame=frame,cameras=cameras,records={c:io.sha(target/(c+'.json')) for c in cameras},
                source_bound_all66=True,prepared_publication_allowed=False,native_mattes_published=False,
                purpose='Current-frame person instances for crew bootstrap/tracking; identity hints unqualified')
            io.write(target/'batch_ready.json',result);io.write(root/'complete.json',result);return result
        time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--frames',nargs='+',type=int,required=True)
    p.add_argument('--gpus',nargs='+',type=int,default=[1,5,6,7]);p.add_argument('--remote-root',required=True)
    p.add_argument('--remote-host',default='ubuntu@silver-sable');p.add_argument('--remote-python',required=True);a=p.parse_args()
    for f in a.frames:print(json.dumps(run_frame(a,f)),flush=True)
