"""Idle-SG RGB frequency backfill with observed native/field priority.

Never writes ready/transforms/semantics/model/dispatch. Per-map specs live in
operations; only immutable same-frame RGB and frequency cache enter frame data.
"""
from pathlib import Path
import argparse,importlib,json,os,shlex,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
import marshawn_sam3_assist as io
import marshawn_rgb_frequency_early as early

SIX={'AA','AB','AC','AD','BA','BB'}


def priority_decision(producer,native,field_pending):
    """Policy independent of IO, exercised by the regression tests below."""
    if field_pending:return True,'eligible field model awaiting a GPU'
    if producer.get('stage') in {'data','native_visual_receipts'}:return True,'producer approaching next native phase'
    for row in native:
        if row['ready']:continue
        if not row['sg_complete']:return True,'native remote assignment/staging/queue active'
        if row['native_count']>=60:return True,'LOCAL remainder near completion; drain before next native'
    return False,'no current SG-native/field priority; LOCAL-only remainder may overlap backfill'


def eligible_gpus(gpus,native_priority,field_priority):
    if field_priority:return []
    if native_priority:return [g for g in gpus if g not in {1,5,6,7}]
    return list(gpus)


def priority(V):
    producer={}
    for name in ['producer_v4_state.json','producer_v3_state.json']:
        p=V/'operations'/name
        if p.exists():
            r=io.read(p)
            if r.get('checked_at',r.get('started',0))>producer.get('checked_at',producer.get('started',0)):producer=r
    native=[]
    for root in (V/'operations/sam3_assist').glob('full[0-9]*'):
        if not root.is_dir():continue
        frame=int(root.name[4:]);ready=(root/'mask_assist_ready.json').exists()
        if ready:continue
        identity=root/'coordinator_identity.json';watch=root/'watchdog_identity.json'
        # Existing directories without any active owner are still pending native
        # requests; yielding is preferable to starving a recovering writer.
        sg=V/'operations/sam3_assist'/('frame%06d'%frame)
        native.append(dict(frame=frame,ready=False,sg_complete=(sg/'assist_complete.json').exists(),
            native_count=len(list((V/'frames'/('%06d'%frame)/'native_masks').glob('[A-Z][A-Z].json')))))
    field=False
    p=V/'state.json'
    if p.exists():
        s=io.read(p)
        for frame,item in s.get('frames',{}).items():
            job=s.get('jobs',{}).get(item['name'],{})
            if job.get('status') in {'pending','dispatching','unknown'} and (V/'frames'/('%06d'%int(frame))/'dispatch_ready.json').exists():field=True
    yes,reason=priority_decision(producer,native,field)
    return dict(yield_new_maps=yes,reason=reason,producer=producer,native=native,field_pending=field,time=time.time())


def install(payload,target,code,opspecs):
    """Stricter than early install: no ready/transforms files are ever written."""
    payload=Path(payload);target=Path(target);code=Path(code);opspecs=Path(opspecs);m=io.read(payload/'early_rgb_sources.json')
    if len(m['records'])!=66:raise ValueError('ALL66 RGB manifest required')
    r=io.read(target/'ready.json') if (target/'ready.json').exists() else {}
    sealed=bool(r.get('records') or 'inventory' in r or any((target/p).exists() for p in ['prepared_ready.json','frequency_ready.json'])
        or (target.parent/'spec.json').exists() or (target.parent/'train/cold_provenance.json').exists())
    opspecs.mkdir(parents=True,exist_ok=True)
    for row in m['records']:
        source=payload/row['image'];dst=target/row['image']
        if io.sha(source)!=row['image_sha256']:raise ValueError('Source RGB payload changed')
        if dst.exists():
            if io.sha(dst)!=row['image_sha256']:raise ValueError('Prepared immutable RGB differs')
        else:
            if sealed:raise ValueError('Missing sealed image; refusing any repair')
            dst.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(source,dst)
            except FileExistsError:
                if io.sha(dst)!=row['image_sha256']:raise
        spec=dict(frame=m['frame'],camera=row['camera'],image=str(dst),image_sha256=row['image_sha256'],
            output=str(target/'lookcloser_frequencies'/('train_'+row['camera']+'.pt')),source_code=str(code),
            preprocess_source_sha256=io.sha(code/'nerfstudio/scripts/lookcloser_preprocess.py'),allocator_GiB=4)
        p=opspecs/(row['camera']+'.json')
        if p.exists() and io.read(p)!=spec:raise ValueError('Own-frame frequency spec changed')
        if not p.exists():io.write(p,spec)
    return dict(frame=m['frame'],images=66,ready_or_semantics_written=False,prepared_publication_allowed=False)


def configuration(V):
    root=V/'operations/future_rgb_sg';root.mkdir(exist_ok=True);scripts=V/'operations/sam3_assist/code'
    paths=[scripts/'marshawn_future_sg_frequency.py',scripts/'marshawn_rgb_frequency_early.py',scripts/'marshawn_sam3_assist.py',
        V/'code_seal.json',V/'code/LookCloser/scripts/marshawn_video_frequency.py',V/'code/nerfstudio/scripts/lookcloser_preprocess.py',
        V/'operations/future_sources/config.json',V/'operations/future_sources/ownership.json']
    c=dict(schema=1,root=str(V),frames=list(range(1675,2020,5)),gpus=[1,3,4,5,6,7],reserved_cameras=sorted(SIX),
        remote_host='ubuntu@silver-sable',remote_root='/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107',
        remote_python='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python',files_sha256={str(p):io.sha(p) for p in paths},
        priority='observe current producer/native requests and eligible main field jobs; never preempt existing fit',prepared_publication_allowed=False)
    p=root/'config.json'
    if p.exists() and io.read(p)!=c:raise ValueError('Future SG backfill config changed')
    io.write(p,c);return p


def guard(c):
    for p,d in c['files_sha256'].items():
        if io.sha(p)!=d:raise ValueError('Released future SG source changed: '+p)


def run(c,config):
    V=Path(c['root']);root=V/'operations/future_rgb_sg';sys.path.insert(0,str(V/'code/LookCloser/scripts'))
    fleet=importlib.import_module('marshawn_tuning_controller');held=fleet.lock(root/'controller.lock')
    if held is None:return
    guard(c);io.write(root/'controller_identity.json',fleet.identity());R=Path(c['remote_root']);op=R/'operations/future_rgb_sg';code=R/'operations/sam3_assist/code'
    def ssh(argv):return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',c['remote_host'],shlex.join(argv)],check=True,capture_output=True,text=True)
    ssh(['mkdir','-p',str(op),str(code)])
    subprocess.run(['rsync','-rt',__file__,c['remote_host']+':'+str(code)+'/'],check=True)
    def stage_frames():
        for frame in c['frames']:
            ready=V/'sources'/str(frame)/'source_ready.json';marker=root/('%06d'%frame)/'staged.json'
            while not ready.exists():time.sleep(10)
            if marker.exists():
                if io.read(marker)['source_ready_sha256']!=io.sha(ready):raise ValueError('Immutable source-ready changed')
                continue
            guard(c);s=io.read(ready)
            if s['frame']!=frame or s['source_verified_all66'] is not True:raise ValueError('Unqualified source')
            b=early.bundle(V,frame);dest=op/('%06d'%frame)/'payload';ssh(['mkdir','-p',str(dest)])
            subprocess.run(['rsync','-rLt',str(b)+'/',c['remote_host']+':'+str(dest)+'/'],check=True)
            ssh([c['remote_python'],str(code/'marshawn_future_sg_frequency.py'),'install','--payload',str(dest),'--target',str(R/'frames'/('%06d'%frame)/'data'),
                 '--code',str(R/'code'),'--opspecs',str(op/('%06d'%frame)/'specs')])
            io.write(marker,dict(frame=frame,source_ready_sha256=io.sha(ready),prepared_publication_allowed=False))
    pool=ThreadPoolExecutor(max_workers=1);staging=pool.submit(stage_frames)
    state=io.read(root/'state.json') if (root/'state.json').exists() else dict(jobs={},maps={})
    host=dict(address=c['remote_host'],python=c['remote_python'],root=str(op/'dispatch'),controller=str(R/'code/LookCloser/scripts/marshawn_tuning_controller.py'),gpus=c['gpus'])
    while True:
        guard(c)
        for frame in c['frames']:
            if not (root/('%06d'%frame)/'staged.json').exists():continue
            for camera in io.read(V/'sources'/str(frame)/'source_ready.json')['cameras']:
                if camera in SIX:continue
                key='%06d_%s'%(frame,camera)
                if key not in state['maps']:
                    name='future%s_a00'%key;state['maps'][key]=dict(frame=frame,camera=camera,name=name,attempt=0);state['jobs'][name]={'status':'pending'}
        if staging.done():staging.result()
        observed=priority(V);allowed=eligible_gpus(c['gpus'],observed['yield_new_maps'],observed['field_pending'])
        observed['eligible_map_gpus']=allowed;observed['native_priority_gpus']=[1,5,6,7]
        io.write(root/'priority.json',observed);jobs=[]
        for key,item in state['maps'].items():
            current=state['jobs'][item['name']]
            # Probe/adoption uses state independently of plan.jobs. Exclude jobs
            # pinned to native-priority slots even when UNKNOWN may become
            # PENDING during tick; unassigned jobs may use fair slots3/4.
            if not allowed or (current.get('gpu') is not None and current['gpu'] not in allowed):continue
            jobs.append(dict(name=item['name'],host='silver',command=[c['remote_python'],str(R/'code/LookCloser/scripts/marshawn_video_frequency.py'),
                '--spec',str(op/('%06d'%item['frame'])/'specs'/(item['camera']+'.json'))],env={}))
        tick_host=dict(host,gpus=allowed)
        fleet.tick(dict(schema_version=1,poll_seconds=5,hosts={'silver':tick_host},jobs=jobs),state,root)
        for key,item in state['maps'].items():
            job=state['jobs'][item['name']]
            if job['status']=='failed':
                snap=state.get('host_snapshots',{}).get('silver',{});proof=snap.get('jobs',{}).get(item['name'])
                if proof and not proof.get('supervisor_alive') and not proof.get('worker_alive') and item['attempt']<12:
                    item['attempt']+=1;item['name']='future%s_a%02d'%(key,item['attempt']);state['jobs'][item['name']]={'status':'pending'}
        io.write(root/'state.json',state);statuses=[state['jobs'][i['name']]['status'] for i in state['maps'].values()]
        io.write(root/'progress.json',dict(time=time.time(),queued_maps=len(statuses),counts={s:statuses.count(s) for s in set(statuses)},priority=observed,
            source_frames_available=len({i['frame'] for i in state['maps'].values()}),prepared_publication_allowed=False))
        if len(statuses)==69*60 and all(s=='complete' for s in statuses):io.write(root/'complete.json',dict(maps=len(statuses),time=time.time()));return
        time.sleep(5)


def watchdog(c,config):
    V=Path(c['root']);root=V/'operations/future_rgb_sg';sys.path.insert(0,str(V/'code/LookCloser/scripts'));fleet=importlib.import_module('marshawn_tuning_controller')
    held=fleet.lock(root/'watchdog.lock')
    if held is None:return
    io.write(root/'watchdog_identity.json',fleet.identity())
    while not (root/'complete.json').exists():
        guard(c);p=root/'controller_identity.json';old=io.read(p) if p.exists() else None
        if not fleet.alive(old):
            with (root/'controller.log').open('ab') as stream:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--config',str(config)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            fleet.append(root/'watchdog_checks.jsonl',dict(process=fleet.identity(child.pid),time=time.time()))
        time.sleep(15)


def selftest():
    assert not priority_decision({'stage':'native_assist'},[dict(ready=False,sg_complete=True,native_count=35)],False)[0]
    assert priority_decision({'stage':'native_assist'},[dict(ready=False,sg_complete=False,native_count=35)],False)[0]
    assert priority_decision({'stage':'native_assist'},[dict(ready=False,sg_complete=True,native_count=62)],False)[0]
    assert priority_decision({'stage':'data'},[],False)[0]
    assert priority_decision({},[],True)[0]
    assert not priority_decision({'stage':'crew_current_person_guard'},[],False)[0]
    assert eligible_gpus([1,3,4,5,6,7],True,False)==[3,4]
    assert eligible_gpus([1,3,4,5,6,7],False,True)==[]
    # Continuous native/guard/DATA cycles cannot starve RGB progress on3/4.
    progress={3:0,4:0};remaining={3:90,4:90}
    for cycle in range(10):
        for stage,duration in [('native_assist',120),('crew_current_person_guard',30),('data',150),('native_visual_receipts',30)]:
            needed=stage!='crew_current_person_guard'
            allowed=eligible_gpus([1,3,4,5,6,7],needed,False)
            for second in range(duration):
                for g in [3,4]:
                    assert g in allowed
                    remaining[g]-=1
                    if remaining[g]==0:progress[g]+=1;remaining[g]=90
    assert min(progress.values())>=30
    print('PASS: native4slots drain, field priority respected, continuous10native/DATA cycles stillcomplete>=30maps onEACHGPU3/4')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['configure','run','watchdog','install','selftest']);p.add_argument('--root',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--payload',type=Path);p.add_argument('--target',type=Path);p.add_argument('--code',type=Path);p.add_argument('--opspecs',type=Path);a=p.parse_args()
    if a.action=='selftest':selftest()
    elif a.action=='install':print(json.dumps(install(a.payload,a.target,a.code,a.opspecs)))
    elif a.action=='configure':print(configuration(a.root.resolve()))
    else:
        c=io.read(a.config);guard(c);run(c,a.config) if a.action=='run' else watchdog(c,a.config)
