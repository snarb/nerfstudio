"""Version3 RGB frequency broker: explicit native and MAT per-slot drains.

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
    if producer.get('native_frequency_drain_nonce'):return True,'explicit native partition drain nonce'
    for row in native:
        if row['ready']:continue
        if not row['sg_complete']:return True,'native remote assignment/staging/queue active'
        # A completed remote phase with only a LOCAL remainder needs no SG reservation.
    return False,'no current SG-native/field priority; LOCAL-only remainder may overlap backfill'


def eligible_gpus(gpus,native_priority,field_priority,mat_slots=()):
    if field_priority:return []
    allowed=[g for g in gpus if g not in {1,5,6,7}] if native_priority else list(gpus)
    return [g for g in allowed if g not in set(mat_slots)]


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
    for root in (V/'operations/sam3_assist/v5').glob('frame*/all66_controller'):
        if (root/'all66_review_ready.json').exists():continue
        state_path=root/'state.json'
        if not state_path.exists():continue
        state=io.read(state_path)
        config_path=root.parent/'all66_config.json'
        config=io.read(config_path) if config_path.exists() else {}
        ownership={row['camera']:row['host'] for row in config.get('records',[])}
        # LOCAL-only native remainder may overlap idle SG maps. Unknown
        # ownership remains reserved rather than assuming an idle queue.
        if any(not row.get('collected') and ownership.get(camera)!='olive'
               for camera,row in state.get('cameras',{}).items()):
            native.append(dict(frame=int(root.parent.name[5:]),ready=False,sg_complete=False,native_count=0,owner='V5 isolated queued or running native views'))
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


def material_requests(V,c):
    requests=[]
    for path in (V/'operations/advanced_inputs').glob('[0-9]*/mat_request.json'):
        request=io.read(path)
        if request.get('status')=='complete':continue
        if (request.get('kind')!='MAT_NATIVE_GPU_REQUEST_V1' or request.get('status')not in {'pending','running'}
            or request.get('raw_inputs_ready')is not True or not request.get('request_nonce')):
            raise ValueError('Unqualified MAT scheduling intent: '+str(path))
        if request.get('source_ops_sha256')not in c['mat_source_ops_sha256']:
            raise ValueError('MAT intent source is outside versioned release')
        frame=int(path.parent.name)
        if request['frame']!=frame or type(request.get('requested_slot'))is not int or request['requested_slot']not in list(range(8)):
            raise ValueError('MAT scheduling frame/GPU differs')
        alias=Path(request['raw_alias_receipt_path'])
        if not alias.resolve().is_relative_to(V.resolve()) or io.sha(alias)!=request['raw_alias_receipt_sha256']:
            raise ValueError('MAT native alias source proof changed')
        record=io.read(alias)
        if (record['frame']!=frame or len(record['records'])!=66
            or record.get('current_ALL66_native_aliases_verified')is not True
            or record.get('source_ops_sha256')!=request['source_ops_sha256']):
            raise ValueError('MAT requires ALL66 own-frame source aliases')
        requests.append(dict(frame=frame,gpu=request['requested_slot'],nonce=request['request_nonce'],status=request['status'],
                             request=str(path),request_sha256=io.sha(path),raw_alias_receipt_sha256=io.sha(alias)))
    return requests


def configuration(V):
    root=V/'operations/future_rgb_sg';root.mkdir(exist_ok=True);scripts=V/'operations/sam3_assist/code'
    mat_source=V/'operations/advanced_input_dispatcher_v2.py'
    if not mat_source.exists():raise ValueError('Versioned MAT intent publisher must be staged before brokerV3 release')
    mat_helper=V/'operations/advanced_mat_request.py'
    if not mat_helper.exists():raise ValueError('Versioned MAT source/capacity helper missing')
    paths=[scripts/'marshawn_future_sg_frequency_v3.py',scripts/'marshawn_rgb_frequency_early.py',scripts/'marshawn_sam3_assist.py',
        V/'code_seal.json',V/'code/LookCloser/scripts/marshawn_video_frequency.py',V/'code/nerfstudio/scripts/lookcloser_preprocess.py',
        V/'operations/future_sources/config.json',V/'operations/future_sources/ownership.json',mat_source,mat_helper]
    c=dict(schema=1,root=str(V),frames=list(range(1675,2020,5)),gpus=list(range(8)),reserved_cameras=sorted(SIX),
        remote_host='ubuntu@silver-sable',remote_root='/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107',
        remote_python='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python',mat_source_ops_sha256=[io.sha(mat_source)],files_sha256={str(p):io.sha(p) for p in paths},
        priority='observe current producer/native requests and eligible main field jobs; never preempt existing fit',prepared_publication_allowed=False)
    p=root/'config_v3.json'
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
            ssh([c['remote_python'],str(code/'marshawn_future_sg_frequency_v3.py'),'install','--payload',str(dest),'--target',str(R/'frames'/('%06d'%frame)/'data'),
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
        observed=priority(V);material=material_requests(V,c)
        mat_slots=sorted({r['gpu']for r in material});observed['material_requests']=material;observed['material_priority_gpus']=mat_slots
        allowed=eligible_gpus(c['gpus'],observed['yield_new_maps'],observed['field_pending'],mat_slots)
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
    assert not priority_decision({'stage':'data'},[],False)[0]
    assert not priority_decision({'stage':'native_visual_receipts'},[],False)[0]
    assert priority_decision({'native_frequency_drain_nonce':'nonce'},[],False)[0]
    assert priority_decision({},[dict(ready=False,sg_complete=False,native_count=0)],False)[0]
    assert not priority_decision({},[dict(ready=False,sg_complete=True,native_count=65)],False)[0]
    assert eligible_gpus(list(range(8)),True,False)==[0,2,3,4]
    assert eligible_gpus(list(range(8)),False,True)==[]
    # Fields occupy0..4 continuously. During each CPU phase5..7 can fit,
    # then an explicit nonce prevents any new map and all existing90s fits
    # finish before the native partition. Native inference owns5..7 next.
    fields=set(range(5));progress={5:0,6:0,7:0};remaining={5:None,6:None,7:None};drains=[]
    for cycle in range(10):
        for stage,duration in [('crew_current_person_guard',120),('data',150),('native_visual_receipts',30)]:
            needed=priority_decision({'stage':stage},[],False)[0]
            allowed=set(eligible_gpus(list(range(8)),needed,False))-fields
            assert allowed=={5,6,7}
            for second in range(duration):
                for g in allowed:
                    if remaining[g] is None:remaining[g]=90
                    remaining[g]-=1
                    if remaining[g]==0:progress[g]+=1;remaining[g]=None
        assert not ({5,6,7}&set(eligible_gpus(list(range(8)),True,False)))
        elapsed=0
        while any(v is not None for v in remaining.values()):
            elapsed+=1
            for g in remaining:
                if remaining[g] is not None:
                    remaining[g]-=1
                    if remaining[g]==0:progress[g]+=1;remaining[g]=None
        assert elapsed<=90;drains.append(elapsed)
    assert min(progress.values())>=30
    # MAT owns only its requested slot; all other free map slots advance.
    remaining={g:90 for g in [3,4,5,6,7]};completed={g:0 for g in remaining};mat_started=None
    for second in range(181):
        allowed=set(eligible_gpus(list(range(8)),False,False,[4]))-set(range(3))
        assert allowed=={3,5,6,7}
        for g in remaining:
            if remaining[g]is None and g in allowed:remaining[g]=90
            if remaining[g]is not None:
                remaining[g]-=1
                if remaining[g]==0:remaining[g]=None;completed[g]+=1
        if remaining[4]is None and mat_started is None:mat_started=second+1
    assert mat_started<=90 and completed[4]==1 and min(completed[g]for g in [3,5,6,7])>=2
    print(json.dumps(dict(MAT_pending_slot=4,MAT_global_lease_available_after_seconds=mat_started,other_slots_completions=completed)))
    # Once field models own0..4, a qualified own-map slot7 can instead drain
    # for MAT while maps5/6 continue. No field/NN slot is preempted.
    remaining={g:90 for g in [5,6,7]};completed={g:0 for g in remaining};mat_started=None
    for second in range(181):
        allowed=set(eligible_gpus(list(range(8)),False,False,[7]))-set(range(5))
        assert allowed=={5,6}
        for g in remaining:
            if remaining[g]is None and g in allowed:remaining[g]=90
            if remaining[g]is not None:
                remaining[g]-=1
                if remaining[g]==0:remaining[g]=None;completed[g]+=1
        if remaining[7]is None and mat_started is None:mat_started=second+1
    assert mat_started<=90 and completed[7]==1 and completed[5]>=2 and completed[6]>=2
    print(json.dumps(dict(field_gpus_held=list(range(5)),MAT_pending_slot=7,MAT_global_lease_available_after_seconds=mat_started,other_slots_completions=completed)))
    print(json.dumps(dict(passed=True,field_gpus_held=sorted(fields),cpu_phase_map_completions=progress,
                          maximum_natural_drain_seconds=max(drains),explicit_nonce_before_every_partition=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['configure','run','watchdog','install','selftest']);p.add_argument('--root',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--payload',type=Path);p.add_argument('--target',type=Path);p.add_argument('--code',type=Path);p.add_argument('--opspecs',type=Path);a=p.parse_args()
    if a.action=='selftest':selftest()
    elif a.action=='install':print(json.dumps(install(a.payload,a.target,a.code,a.opspecs)))
    elif a.action=='configure':print(configuration(a.root.resolve()))
    else:
        c=io.read(a.config);guard(c);run(c,a.config) if a.action=='run' else watchdog(c,a.config)
