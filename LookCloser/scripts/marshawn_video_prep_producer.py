"""Durable chronological centralized video preparation and verified remote dispatch.

Only image inputs/segmentation IDs pass between frames. NeRF training, learned
occupancy and fresh frequency fitting run in separate frame-local consumers.
"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from prepare_marshawn_tuning import write,sha
from marshawn_video_prep_inputs import prototype


def command(args,stage,argv,env=None,pass_fds=()):
    log=args.root/'preparation_logs'/f'{args.frame:06d}_{stage}.log';log.parent.mkdir(exist_ok=True)
    state=dict(pid=os.getpid(),frame=args.frame,stage=stage,command=argv,started=time.time())
    write(args.root/'preparation_state.json',state)
    with log.open('a') as output:
        child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,pass_fds=pass_fds)
        state['worker_pid']=child.pid;write(args.root/'preparation_state.json',state)
        while child.poll() is None:
            state['checked_at']=time.time();write(args.root/'preparation_state.json',state);time.sleep(20)
        if child.returncode:raise RuntimeError(f'{stage} exited{child.returncode}; see{log}')
    write(args.root/'preparation_logs'/f'{args.frame:06d}_{stage}_done.json',dict(state,finished=time.time()))


def rebind_first_frame(root,box):
    data=prototype(root);ready=json.loads((data/'ready.json').read_text());prior=json.loads((data/'box.json').read_text());new=json.loads(box.read_text())
    if prior==new:return
    if prior['world_to_box']!=new['world_to_box']:raise ValueError('Changing rigid frame requires explicit camera reprojection, not silent rebind')
    if (root/'frames/001660/dispatch_ready.json').exists():raise ValueError('Cannot change already-dispatched1660 box')
    ready.pop('inventory',None);ready.pop('box_sha256',None);ready['aabb']=new['aabb'];ready['world_to_box']=new['world_to_box'];write(data/'box.json',new);write(data/'ready.json',ready)
    prepared=json.loads((data/'prepared_ready.json').read_text());prepared['box_sha256']=sha(data/'box.json');write(data/'prepared_ready.json',prepared)


def main(a):
    a.root.mkdir(parents=True,exist_ok=True);lease=open(a.root/'preparation_producer.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    release=json.loads(a.release.read_text());box=a.root/'common_box.json';inventory=a.root/'source_inventory.json';source=json.loads(inventory.read_text())
    if not release.get('authorized') or release['box_sha256']!=sha(box) or release['source_inventory_sha256']!=sha(inventory):raise ValueError('Preparation release/source/box not sealed')
    if source['raw_available_all'] is not True or len(source['cameras'])!=66:raise ValueError('Incomplete RAW source inventory')
    code=Path(__file__).resolve().parent;python=sys.executable;overlay=a.overlay
    for source_path,digest in release.get('source_files_sha256',{}).items():
        if sha(source_path)!=digest:raise ValueError('Frozen preparation source changed '+source_path)
    for name,digest in release['scripts_sha256'].items():
        if sha(code/name)!=digest:raise ValueError('Preparation script changed '+name)
    env=dict(os.environ,PYTHONPATH=str(overlay/'src')+':'+str(overlay/'deps')+':'+str(code),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=a.gpu_uuid)
    cameras=source['cameras'];rebind_first_frame(a.root,box)
    frames=[r['frame'] for r in source['frames']]
    if a.only_frame is not None:frames=[a.only_frame]
    for frame in frames:
        a.frame=frame;froot=a.root/'frames'/f'{frame:06d}'
        marker=froot/('central_prepared.json' if a.defer_dispatch else 'dispatch_ready.json')
        if marker.exists():continue
        attempt=0
        while not marker.exists():
            try:
                for source_path,digest in release.get('source_files_sha256',{}).items():
                    if sha(source_path)!=digest:raise ValueError('Frozen preparation source changed '+source_path)
                if frame!=1660:
                    standalone=a.root/'preparation_logs'/f'{frame}_standalone_launch.json'
                    if standalone.exists():
                        pid=int(json.loads(standalone.read_text())['pid'])
                        while True:
                            try:
                                owner=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
                                alive='marshawn_video_prep_one_frame.py' in owner and str(a.root) in owner
                            except OSError:alive=False
                            if not alive:break
                            write(a.root/'preparation_state.json',dict(pid=os.getpid(),frame=frame,stage='adopting_inflight_native_preparation',worker_pid=pid,checked_at=time.time()))
                            time.sleep(20)
                    native_done=a.root/'preparation_logs'/f'{frame}_native_complete.json'
                    native_ready=native_done.exists()
                    if native_ready:
                        for camera in cameras:
                            meta=froot/'native_masks'/f'{camera}.json'
                            if not meta.exists():native_ready=False;break
                            m=json.loads(meta.read_text())
                            if m.get('native_cleanup_version')!=1 or m.get('mask_code_sha256')!=sha(code/'marshawn_video_prep_masks.py') or m['archive_sha256']!=sha(froot/'native_masks'/f'{camera}.npz'):native_ready=False;break
                    if not native_ready:
                        command(a,'sources',[python,str(code/'marshawn_video_prep_sources.py'),'--frames',str(frame),'--cameras',*cameras,'--output',str(a.root/'sources'),'--raw-native','--workers',str(a.decode_workers)],env)
                        gpu_lease=open('/tmp/lookcloser_tuning_gpu_'+str(os.getuid())+'_0.lock','a+');fcntl.flock(gpu_lease,fcntl.LOCK_EX)
                        try:command(a,'masks',[python,str(code/'marshawn_video_prep_masks.py'),'--root',str(a.root),'--frame',str(frame),'--cameras',*cameras,'--checkpoint',str(overlay/'weights/sam3.pt')],env,pass_fds=(gpu_lease.fileno(),))
                        finally:fcntl.flock(gpu_lease,fcntl.LOCK_UN);gpu_lease.close()
                    command(a,'data',[python,str(code/'marshawn_video_prep_data.py'),'--root',str(a.root),'--frame',str(frame),'--box',str(box)],env)
                if a.defer_dispatch:
                    write(marker,dict(frame=frame,train=66,prepared_ready_sha256=sha(froot/'data/prepared_ready.json'),source_input_ready=True,dispatch_deferred=True,finished=time.time()))
                else:
                    command(a,'dispatch',[python,str(code/'marshawn_video_stage.py'),'push','--source-campaign',str(a.root),'--frame',str(frame),'--target',a.target,'--target-root',a.target_root,'--target-python',a.target_python,'--host-key',a.host_key],env)
                    # Stage helper owns the exact dispatch marker schema.
                    if not marker.is_file():raise RuntimeError('Verified staging did not publish dispatch marker')
            except Exception as error:
                attempt+=1;write(a.root/'preparation_error.json',dict(frame=frame,attempt=attempt,error=str(error),traceback=traceback.format_exc(),at=time.time(),policy='Retry60s; UNKNOWN visibility or missing source blocks publication'))
                if a.fail_fast:raise
                time.sleep(60)
    write(a.root/'preparation_complete.json',dict(frames=frames,finished=time.time(),all66_each_frame=True,no_model_optimizer_or_learned_grid_transfer=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','release','overlay']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['target','target-root','target-python','host-key']:p.add_argument('--'+name,required=True)
    p.add_argument('--gpu-uuid',default='GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46');p.add_argument('--decode-workers',type=int,default=2);p.add_argument('--only-frame',type=int);p.add_argument('--defer-dispatch',action='store_true');p.add_argument('--fail-fast',action='store_true');main(p.parse_args())
