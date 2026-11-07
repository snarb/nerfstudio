"""RGB-only prefetch/frequency work while semantic qualification is blocked.

No mask, admission, prepared_ready, frequency_ready or dispatch marker is made.
The six DEV3-owned cameras are excluded from every SG fit job.
"""
from pathlib import Path
import argparse,importlib,json,os,shlex,shutil,subprocess,sys,time
import marshawn_sam3_assist as io

RESERVED={'AA','AB','AC','AD','BA','BB'}


def execute(argv,**kwargs):return subprocess.run(argv,check=True,**kwargs)


def bundle(V,frame):
    out=V/'operations/early_rgb_sg'/('%06d'%frame)/'bundle';(out/'images').mkdir(parents=True,exist_ok=True)
    tf=io.read(V/'frames/001660/data/transforms.json');records=[]
    for row in tf['frames']:
        camera=row['physical_camera'];receipt=V/'sources'/str(frame)/camera/'receipt.json';r=io.read(receipt);image=Path(r['image'])
        if r['frame']!=frame or r['camera']!=camera or io.sha(image)!=r['image_sha256']:raise ValueError('Actual RGB source differs')
        relative='images/train_'+camera+'.png';target=out/relative
        if not target.exists():target.symlink_to(image)
        if io.sha(target)!=r['image_sha256']:raise ValueError('Existing staged RGB differs')
        row['file_path']=relative;row.pop('mask_path',None)
        records.append(dict(camera=camera,image=relative,image_sha256=r['image_sha256'],original_source_receipt_sha256=io.sha(receipt)))
    tf['train_filenames']=[r['file_path'] for r in tf['frames']]
    tf['val_filenames']=[r['file_path'] for r in tf['frames'] if r['physical_camera'] in ['DC','HC','KA']]
    tf['test_filenames']=list(tf['val_filenames']);tf['early_source_frame']=frame
    io.write(out/'transforms.json',tf);io.write(out/'ready.json',dict(frame=frame,train=66,stage='RGB_ONLY_UNTRAINABLE',prepared_publication_allowed=False))
    io.write(out/'early_rgb_sources.json',dict(frame=frame,records=records,prepared_publication_allowed=False))
    return out


def install(bundle,target,source_code):
    bundle=Path(bundle);target=Path(target);manifest=io.read(bundle/'early_rgb_sources.json');target.mkdir(parents=True,exist_ok=True)
    ready=io.read(target/'ready.json') if (target/'ready.json').exists() else {}
    immutable=bool(ready.get('records') or 'inventory' in ready or
        any((target/name).exists() for name in ['prepared_ready.json','frequency_ready.json']) or
        any((target.parent/name).exists() for name in ['dispatch_ready.json','spec.json','train/cold_provenance.json']))
    for row in manifest['records']:
        source=bundle/row['image'];destination=target/row['image']
        if io.sha(source)!=row['image_sha256']:raise ValueError('RGB bundle changed')
        if destination.exists():
            if io.sha(destination)!=row['image_sha256']:raise ValueError('Future data image already differs; never overwrite')
        else:
            if immutable:raise ValueError('Missing image in immutable prepared data; early RGB installer is read-only')
            destination.parent.mkdir(exist_ok=True)
            try:os.link(source,destination)
            except OSError:shutil.copy2(source,destination)
    if immutable:
        # A restarted prefetcher must never downgrade rich/sealed training data.
        # Even metadata/spec creation is forbidden in this branch.
        if not (target/'ready.json').is_file() or not (target/'transforms.json').is_file():raise ValueError('Immutable prepared metadata missing; refusing repair/overwrite')
        preprocess=Path(source_code)/'nerfstudio/scripts/lookcloser_preprocess.py'
        for row in manifest['records']:
            path=target/'frequency_specs'/(row['camera']+'.json')
            if not path.is_file():raise ValueError('Immutable data is missing a frequency spec; refusing creation')
            expected=dict(frame=manifest['frame'],camera=row['camera'],image=str(target/row['image']),image_sha256=row['image_sha256'],
                output=str(target/'lookcloser_frequencies'/('train_'+row['camera']+'.pt')),source_code=str(source_code),
                preprocess_source_sha256=io.sha(preprocess),allocator_GiB=4)
            if io.read(path)!=expected:raise ValueError('Existing sealed frequency spec differs')
        return dict(frame=manifest['frame'],images=66,specs=66,rich_ready_preserved=True,read_only_restart=True,prepared_publication_allowed=False)
    shutil.copy2(bundle/'transforms.json',target/'transforms.json');shutil.copy2(bundle/'ready.json',target/'ready.json')
    shutil.copy2(bundle/'early_rgb_sources.json',target/'early_rgb_sources.json')
    sys.path[:0]=[str(source_code),str(Path(source_code)/'LookCloser/scripts')]
    from marshawn_video_frequency_frame import specs
    # source_code is campaign/code; all paths exactly match final pipeline paths.
    paths=specs(Path(source_code).parent,manifest['frame'])
    result=dict(frame=manifest['frame'],images=66,specs=66,rich_ready_preserved=False,prepared_publication_allowed=False)
    io.write(bundle.parent/'installed.json',result);return result


def run(a):
    V=a.campaign.resolve();root=V/'operations/early_rgb_sg';root.mkdir(exist_ok=True)
    sys.path.insert(0,str(V/'code/LookCloser/scripts'));fleet=importlib.import_module('marshawn_tuning_controller');held=fleet.lock(root/'controller.lock')
    if held is None:raise RuntimeError('Early RGB frequency controller already running')
    R=Path(a.remote_root);remote_ops=R/'operations/early_rgb_sg';opscode=R/'operations/sam3_assist/code'
    host=dict(address=a.remote_host,python=a.remote_python,root=str(remote_ops/'dispatch'),controller=str(R/'code/LookCloser/scripts/marshawn_tuning_controller.py'),gpus=a.gpus)
    execute(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.remote_host,shlex.join(['mkdir','-p',str(remote_ops),str(opscode)])])
    execute(['rsync','-rt',__file__,a.remote_host+':'+str(opscode)+'/'])
    for frame in a.frames:
        b=bundle(V,frame);remote_bundle=remote_ops/('%06d'%frame)/'bundle'
        execute(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.remote_host,shlex.join(['mkdir','-p',str(remote_bundle)])])
        execute(['rsync','-rLt',str(b)+'/',a.remote_host+':'+str(remote_bundle)+'/'])
        execute(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.remote_host,shlex.join([a.remote_python,str(opscode/'marshawn_rgb_frequency_early.py'),
            'install','--bundle',str(remote_bundle),'--target',str(R/'frames'/('%06d'%frame)/'data'),'--source-code',str(R/'code')])])
    # Candidate capture/crew bootstrap has priority over optional map fits.
    while any(not (V/'frames'/('%06d'%f)/'person_candidates_sg/batch_ready.json').exists() for f in a.frames):
        io.write(root/'candidate_wait.json',dict(frames=a.frames,time=time.time(),GPU_leases_claimed=False));time.sleep(10)
    cameras=sorted(f['physical_camera'] for f in io.read(V/'frames/001660/data/transforms.json')['frames'])
    state=io.read(root/'state.json') if (root/'state.json').exists() else dict(jobs={},maps={})
    for frame in a.frames:
        for camera in cameras:
            if camera in RESERVED:continue
            key='%06d_%s'%(frame,camera)
            if key not in state['maps']:
                name='rgb%s_a00'%key;state['maps'][key]=dict(name=name,frame=frame,camera=camera,attempt=0);state['jobs'][name]={'status':'pending'}
    io.write(root/'controller.json',dict(process=fleet.identity(),started=time.time()))
    while True:
        jobs=[]
        for key,item in state['maps'].items():
            data=R/'frames'/('%06d'%item['frame'])/'data'
            jobs.append(dict(name=item['name'],host='silver',command=[a.remote_python,str(R/'code/LookCloser/scripts/marshawn_video_frequency.py'),
                '--spec',str(data/'frequency_specs'/(item['camera']+'.json'))],env={},
                progress_files=[str(data/'lookcloser_frequencies'/('train_'+item['camera']+'.progress.json'))]))
        fleet.tick(dict(schema_version=1,poll_seconds=5,hosts={'silver':host},jobs=jobs),state,root)
        for key,item in state['maps'].items():
            job=state['jobs'][item['name']]
            if job['status']=='failed':
                snap=state.get('host_snapshots',{}).get('silver',{});observed=snap.get('jobs',{}).get(item['name'])
                if observed and not observed.get('supervisor_alive') and not observed.get('worker_alive') and item['attempt']<12:
                    item['attempt']+=1;item['name']='rgb%s_a%02d'%(key,item['attempt']);state['jobs'][item['name']]={'status':'pending'}
        io.write(root/'state.json',state);statuses=[state['jobs'][i['name']]['status'] for i in state['maps'].values()]
        io.write(root/'progress.json',dict(frames=a.frames,counts={s:statuses.count(s) for s in set(statuses)},
            DEV3_reserved_cameras=sorted(RESERVED),prepared_publication_allowed=False,time=time.time()))
        if all(s=='complete' for s in statuses):io.write(root/'complete.json',dict(frames=a.frames,maps=len(statuses),prepared_publication_allowed=False));return
        time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','install']);p.add_argument('--campaign',type=Path)
    p.add_argument('--frames',nargs='+',type=int);p.add_argument('--gpus',nargs='+',type=int,default=[1,5,6,7]);p.add_argument('--remote-root')
    p.add_argument('--remote-host',default='ubuntu@silver-sable');p.add_argument('--remote-python');p.add_argument('--bundle',type=Path)
    p.add_argument('--target',type=Path);p.add_argument('--source-code',type=Path);a=p.parse_args()
    print(json.dumps(install(a.bundle,a.target,a.source_code))) if a.action=='install' else run(a)
