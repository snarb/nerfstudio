"""All66 source-bound maps on transient free GPUs; no reserved six-camera role.

Existing Silver job names/specs/receipt roots are retained for safe adoption.
Version11 schedules all cameras in global-frame order after legacy migration.
Every newly assigned map has one durable host owner. Completed maps from shared
FSX are validated and published into the canonical Silver cache before use.
"""
from pathlib import Path
import argparse, copy, importlib, importlib.util, json, os, shlex, shutil, subprocess, sys, threading, time

V_DEFAULT = Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107')

def clever(c):
    sys.path.insert(0,str(Path(c['root'])/'operations/frequency13/code'))
    return importlib.import_module('marshawn_clever_frequency_backend')

def require_release(c):
    if c.get('authorized_main_jobs')is not True:raise ValueError('CandidateV13 cannot claim main jobs before ROOT release')
    release=json.loads(Path(c['ROOT_release']).read_text())
    if release.get('approved')is not True or release['config_sha256']!=__import__('hashlib').sha256(Path(c['config_path']).read_bytes()).hexdigest():raise ValueError('V13 ROOT release/config mismatch')


def imports(V):
    sys.path[:0] = [str(V / 'operations/sam3_assist/code'), str(V / 'code'), str(V / 'code/LookCloser/scripts')]
    return (importlib.import_module('marshawn_tuning_controller'), importlib.import_module('marshawn_sam3_assist'),
            importlib.import_module('marshawn_future_sg_frequency_v3'), importlib.import_module('marshawn_rgb_frequency_early'))


def configuration(V):
    fleet, io, _, _ = imports(V); old = io.read(V / 'operations/future_rgb_sg/config_v3.json')
    root = V / 'operations/frequency10'; root.mkdir(parents=True, exist_ok=True)
    R = Path(old['remote_root']); hosts = dict(
        silver=dict(address=old['remote_host'], python=old['remote_python'], root=str(R / 'operations/future_rgb_sg/dispatch'),
                    campaign=str(R), controller=str(R / 'code/LookCloser/scripts/marshawn_tuning_controller.py'), gpus=list(range(8))),
        olive=dict(python='/home/pavlo/.conda/envs/nerfstudio/bin/python', root=str(root / 'dispatch'), campaign=str(V),
                   controller=str(V / 'code/LookCloser/scripts/marshawn_tuning_controller.py'), gpus=[0]),
        dev3=dict(address='ubuntu@dev3', python='/home/ubuntu/anaconda3/envs/nerfstudio/bin/python', root=str(root / 'dispatch'),
                  campaign=str(V), controller=str(V / 'code/LookCloser/scripts/marshawn_tuning_controller.py'), gpus=[0]))
    files = [Path(__file__), V / 'code_seal.json', V / 'code/LookCloser/scripts/marshawn_video_frequency.py',
             V / 'code/nerfstudio/scripts/lookcloser_preprocess.py', V / 'operations/sam3_assist/code/marshawn_future_sg_frequency_v3.py',
             V / 'operations/sam3_assist/code/marshawn_rgb_frequency_early.py', V / 'operations/sam3_assist/code/marshawn_sam3_assist.py']
    sources = [V / 'operations/advanced_input_dispatcher_v2.py', V / 'operations/advanced_input_dispatcher_v3.py']
    files += [p for p in sources if p.exists()]
    c = dict(version=10, root=str(V), operations=str(root), frames=list(range(1675, 2020, 5)), hosts=hosts,
             legacy=old, mat_source_ops_sha256=[io.sha(p) for p in sources if p.exists()],
             source_code_sha256=io.sha(__file__), files_sha256={str(p): io.sha(p) for p in files},
             canonical_cache_host='silver', no_reserved_cameras=True, cold_map_fitter_unchanged=True)
    path = root / 'config.json'
    if path.exists() and io.read(path) != c:
        raise ValueError('Unified map configuration is immutable')
    io.write(path, c); return path


def guard(c, io):
    for path, digest in c['files_sha256'].items():
        if io.sha(path) != digest:
            raise ValueError('Unified map released code/input changed: ' + path)


def remote(c, command):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', c['hosts']['silver']['address'], shlex.join(command)],
                          check=True, capture_output=True, text=True, timeout=120)


def stage_frame(c, frame, io, early):
    V = Path(c['root']); R = Path(c['hosts']['silver']['campaign']); old_ops = V / 'operations/future_rgb_sg'
    source = V / 'sources' / str(frame) / 'source_ready.json'
    if not source.exists():
        return False
    marker = old_ops / ('%06d' % frame) / 'staged.json'
    if marker.exists():
        if io.read(marker)['source_ready_sha256'] != io.sha(source):
            raise ValueError('Staged source bytes changed')
        return True
    payload = early.bundle(V, frame); destination = R / 'operations/future_rgb_sg' / ('%06d' % frame) / 'payload'
    remote(c, ['mkdir', '-p', str(destination)])
    subprocess.run(['rsync', '-rLt', str(payload) + '/', c['hosts']['silver']['address'] + ':' + str(destination) + '/'], check=True)
    remote_helper = R / 'operations/frequency10/marshawn_frequency10.py'
    remote(c, [c['hosts']['silver']['python'], str(remote_helper), '_install', '--campaign', str(R), '--frame', str(frame), '--payload', str(destination)])
    io.write(marker, dict(frame=frame, source_ready_sha256=io.sha(source), prepared_publication_allowed=False))
    return True


def local_spec(c, frame, camera, io):
    V = Path(c['root']); ready = io.read(V / 'sources' / str(frame) / 'source_ready.json')
    row = next(r for r in ready['records'] if r['camera'] == camera); data = V / 'frames' / ('%06d' % frame) / 'data'
    image = data / 'images' / ('train_' + camera + '.png'); target = data / 'lookcloser_frequencies' / ('train_' + camera + '.pt')
    if io.sha(row['image']) != row['image_sha256'] or io.sha(row['receipt']) != row['receipt_sha256']:
        raise ValueError('Own-frame RGB/source binding changed')
    if image.exists():
        if io.sha(image) != row['image_sha256']:
            raise ValueError('Existing frame RGB differs')
    else:
        if (data / 'prepared_ready.json').exists() or (data.parent / 'spec.json').exists():
            raise ValueError('Missing sealed RGB cannot be repaired by a map scheduler')
        image.parent.mkdir(parents=True, exist_ok=True); image.symlink_to(Path(row['image']))
    target.parent.mkdir(parents=True, exist_ok=True)
    for path in [target.parent, V / 'operations/frequency10/dispatch', V / 'operations/frequency10/specs' / ('%06d' % frame)]:
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(['setfacl', '-m', 'u:1000:rwx,d:u:1000:rwx,d:u:1001:rwx', str(path)], check=True, capture_output=True)
    spec = dict(frame=frame, camera=camera, image=str(image), image_sha256=row['image_sha256'], output=str(target),
                source_code=str(V / 'code'), preprocess_source_sha256=io.sha(V / 'code/nerfstudio/scripts/lookcloser_preprocess.py'), allocator_GiB=4)
    path = V / 'operations/frequency10/specs' / ('%06d' % frame) / (camera + '.json')
    if path.exists() and io.read(path) != spec:
        raise ValueError('Own-frame map spec changed')
    if not path.exists():
        io.write(path, spec)
    return path


def validator(V):
    path = V / 'code/LookCloser/scripts/marshawn_video_frequency.py'
    spec = importlib.util.spec_from_file_location('_unified_map_validation', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def publish_local_map(c, frame, camera, io):
    V = Path(c['root']); R = Path(c['hosts']['silver']['campaign']); path = local_spec(c, frame, camera, io)
    spec = io.read(path); check = validator(V); check.validate_spec(spec)
    if not check.validate_existing(spec):
        return None
    remote_data = R / 'frames' / ('%06d' % frame) / 'data/lookcloser_frequencies'
    remote(c, ['mkdir', '-p', str(remote_data)])
    # Logical ownership and retired legacy services exclude a same-map writer.
    # An already valid canonical cache is retained rather than overwritten.
    files = [spec['output'], str(Path(spec['output']).with_suffix('.json'))]
    subprocess.run(['rsync', '-rt', '--ignore-existing'] + files + [c['hosts']['silver']['address'] + ':' + str(remote_data) + '/'], check=True)
    remote_spec = R / 'operations/future_rgb_sg' / ('%06d' % frame) / 'specs' / (camera + '.json')
    code = ("import importlib.util,json,os;os.environ['CUDA_VISIBLE_DEVICES']='';"
            "p=%r;s=importlib.util.spec_from_file_location('f',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "a=json.load(open(%r));m.validate_spec(a);assert m.validate_existing(a);"
            "print(json.dumps(dict(map_sha256=m.sha(a['output']),metadata_sha256=m.sha(__import__('pathlib').Path(a['output']).with_suffix('.json')))))") % (
                str(R / 'code/LookCloser/scripts/marshawn_video_frequency.py'), str(remote_spec))
    verified = json.loads(remote(c, [c['hosts']['silver']['python'], '-c', code]).stdout)
    result = dict(frame=frame, camera=camera, LOCAL_map_sha256=io.sha(spec['output']),
                  LOCAL_metadata_sha256=io.sha(Path(spec['output']).with_suffix('.json')), canonical_SG=verified,
                  source_image_sha256=spec['image_sha256'], same_frame_only=True)
    io.write(V / 'operations/frequency10/published' / ('%06d' % frame) / (camera + '.json'), result)
    return result


def field_reservations(V, io):
    pointer = V / 'operations/fleet10_active.json'; reservations = set(); demand = False
    if not pointer.exists():
        return reservations, demand
    active = io.read(pointer)
    if active.get('hard_max_iterations')!=140000 or io.sha(active['plan'])!=active['plan_sha256']:
        raise ValueError('Unified prerequisites require the qualified140k field ledger')
    state = io.read(active['state'])
    for row in state['frames'].values():
        if row.get('host') and state['jobs'][row['name']]['status'] != 'complete':
            reservations.add((row['host'], row.get('gpu')))
    for path in Path(active['assignments']).glob('*.json'):
        row = io.read(path); logical = state['frames'].get(str(row['frame']), {})
        if state['jobs'].get(logical.get('name'), {}).get('status') != 'complete':
            reservations.add((row['host'], row['gpu']))
    for path in Path(active['demands']).glob('*.json'):
        row = io.read(path)
        if row.get('ready_for_training') and row['status'] == 'awaiting_idle_training_slot':
            if row.get('preferred_host')is not None and row.get('preferred_gpu')is not None:
                reservations.add((row['preferred_host'],row['preferred_gpu']))
            else:demand = True
    return reservations, demand


def allowed_slots(c, V, legacy, io):
    reserved, demand = field_reservations(V, io)
    result = {h: [g for g in row['gpus'] if (h, g) not in reserved] for h, row in c['hosts'].items()}
    if demand:
        # Clever is frequency-only, outside the ten field slots.
        result={h:(gs if h=='clever'else[])for h,gs in result.items()}
    # Legacy native requests remain supported until the opportunistic native
    # factory publishes its own source-bound host/GPU requests.
    old = legacy.priority(V)
    producer=old.get('producer',{});frame=producer.get('frame');finished=False
    if producer.get('native_frequency_drain_nonce')and frame is not None:
        done=V/'operations/sam3_assist/v5'/('frame%06d'%frame)/'all66_controller/all66_review_ready.json'
        if done.exists()and done.stat().st_mtime>producer.get('checked_at',0):
            summary=io.read(done)
            finished=summary.get('frame')==frame and len(summary.get('cameras',{}))==66
    if finished:
        old=dict(old,yield_new_maps=False,completed_legacy_nonce_ignored=True)
    if old['yield_new_maps']:
        result['silver'] = [g for g in result['silver'] if g not in {1, 5, 6, 7}]
    materials = legacy.material_requests(V, c)
    for row in materials:
        result['silver'] = [g for g in result['silver'] if g != row['gpu']]
    invalid_requests=[]
    for path in (V / 'operations/native10_requests').glob('*.json'):
        try:
            request=io.read(path)
            slots=request.get('host_gpus') if isinstance(request,dict) else None
            if not isinstance(slots,dict) or not slots or any(h not in c['hosts'] or not isinstance(gs,list) or any(type(g)is not int or g not in c['hosts'][h]['gpus'] for g in gs)for h,gs in slots.items()):
                raise ValueError('Not a valid native GPU request: host_gpus missing or invalid')
        except (ValueError,TypeError,KeyError) as error:
            invalid_requests.append(dict(path=str(path),reason=str(error)))
            continue
        if request.get('status') == 'complete':
            continue
        for host, slots in request['host_gpus'].items():
            result[host] = [g for g in result.get(host, []) if g not in slots]
    storage=clever(c).storage_ok(c)if 'clever'in result else{}
    if 'clever'in result and not storage.get('eligible'):result['clever']=[]
    return result, dict(qualified_field_waiting=bool(demand), reservations=sorted(reserved), material_requests=materials, legacy_native=old, invalid_native_requests=invalid_requests,clever_storage=storage)


def precopy_six(c):
    """Copy only already-complete immutable batches while old fitters run."""
    V=Path(c['root']);fleet,io,_,early=imports(V);R=Path(c['hosts']['silver']['campaign']);check=validator(V);result=[]
    for frame in c['frames']:
        completed=V/'operations/future_frequency'/('%06d'%frame)/'six_maps_verified.json'
        if not completed.exists():continue
        proof=io.read(completed);receipt=V/'operations/frequency10/imports'/('%06d.json'%frame)
        if receipt.exists()and io.read(receipt)['legacy_batch_sha256']==io.sha(completed):continue
        stage_frame(c,frame,io,early);records={};files=[]
        for camera in ['AA','AB','AC','AD','BA','BB']:
            spec=io.read(local_spec(c,frame,camera,io));check.validate_spec(spec)
            if not check.validate_existing(spec):raise ValueError('Completed legacy six map invalid')
            records[camera]=dict(LOCAL_map_sha256=io.sha(spec['output']),LOCAL_metadata_sha256=io.sha(Path(spec['output']).with_suffix('.json')))
            files += [spec['output'],str(Path(spec['output']).with_suffix('.json'))]
        target=R/'frames'/('%06d'%frame)/'data/lookcloser_frequencies';remote(c,['mkdir','-p',str(target)])
        subprocess.run(['rsync','-rt','--ignore-existing']+files+[c['hosts']['silver']['address']+':'+str(target)+'/'],check=True)
        code=("from pathlib import Path;import importlib.util,json,os;os.environ['CUDA_VISIBLE_DEVICES']='';"
              "p=%r;s=importlib.util.spec_from_file_location('f',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);r={};"
              "exec(\"for camera in ['AA','AB','AC','AD','BA','BB']:\\n a=json.load(open(str(Path(%r)/(camera+'.json'))));m.validate_spec(a);assert m.validate_existing(a);r[camera]=dict(map_sha256=m.sha(a['output']),metadata_sha256=m.sha(Path(a['output']).with_suffix('.json')))\");"
              "print(json.dumps(r))")%(str(R/'code/LookCloser/scripts/marshawn_video_frequency.py'),str(R/'operations/future_rgb_sg'/('%06d'%frame)/'specs'))
        canonical=json.loads(remote(c,[c['hosts']['silver']['python'],'-c',code]).stdout)
        for camera in records:records[camera]['canonical_SG']=canonical[camera]
        io.write(receipt,dict(frame=frame,records=records,legacy_batch_sha256=io.sha(completed),same_frame_only=True,no_unstarted_task_reassigned=True,time=time.time()))
        result.append(frame)
    return result


def map_job(c,item):
    if item.get('host')=='clever':return clever(c).map_job(c,item)
    host_key=item['host'];host=c['hosts'][host_key];campaign=Path(host['campaign'])
    spec=item.get('spec')or str(campaign/'operations/future_rgb_sg'/('%06d'%item['frame'])/'specs'/(item['camera']+'.json'))
    return dict(name=item['name'],host=host_key,command=[host['python'],str(campaign/'code/LookCloser/scripts/marshawn_video_frequency.py'),'--spec',spec],env={})


def preflight(c):
    V=Path(c['root']);fleet,io,_,_=imports(V);old=io.read(V/'operations/future_rgb_sg/state.json');proof={}
    for key,item in old['maps'].items():
        job=old['jobs'][item['name']]
        if not job.get('spec_sha256'):continue
        current=dict(item,host='silver');new=map_job(c,current);spec=fleet.materialize(new,c['hosts']['silver'],job['gpu'])
        spec['poll_seconds']=5;spec['env']['TUNING_DEPENDENCIES_JSON']='{}'
        if fleet.digest(spec)!=job['spec_sha256']:raise ValueError('Preserved scientific map specification changed: '+item['name'])
        if job['status']in {'running','unknown','dispatching'}:proof[item['name']]=dict(spec_sha256=job['spec_sha256'],worker=job.get('worker'),supervisor=job.get('supervisor'))
    result=dict(preserved_live_workers=proof,all66_cameras=True,no_reserved_six=True,hosts={k:v['gpus']for k,v in c['hosts'].items()},no_GPU_started=True,time=time.time())
    io.write(Path(c['operations'])/'preflight.json',result);return result


def adopt(c):
    V = Path(c['root']); fleet, io, _, early = imports(V); root = Path(c['operations'])
    retirement = io.read(V / 'operations/future_frequency/exclusive_service_retired.json')
    if not retirement['parent_and_child_proved_dead'] or not retirement['global_lease_free']:
        raise ValueError('Exclusive DEV3 six-map ownership has not retired')
    for path in [V / 'operations/future_rgb_sg/controller_identity.json', V / 'operations/future_rgb_sg/watchdog_identity.json']:
        if path.exists() and fleet.alive(io.read(path)):
            raise ValueError('Legacy Silver map CPU controller must retire before adoption')
    if (root / 'state.json').exists():
        return
    old = io.read(V / 'operations/future_rgb_sg/state.json'); state = copy.deepcopy(old)
    host = c['hosts']['silver']; reply = fleet.host_call(host, ['_probe', '--root', host['root']]); reply.check_returncode()
    remote_state = json.loads(reply.stdout)
    for key, item in state['maps'].items():
        job = state['jobs'][item['name']]; observed = remote_state['jobs'].get(item['name'])
        if observed:
            if job.get('spec_sha256') and observed['spec_sha256'] != job['spec_sha256']:
                raise ValueError('Existing Silver map job specification differs')
            job.update(observed); job['host'] = 'silver'; item['host'] = 'silver'
        elif job.get('host'):
            item['host'] = job['host']
        else:
            item['host'] = None
    # Validate and publish every available legacy-six artifact before claiming
    # any missing six-camera task. There is no full-video completion barrier.
    imported = []
    for frame in c['frames']:
        if not (V / 'sources' / str(frame) / 'source_ready.json').exists():
            continue
        # Import the existing ready prefix only. Future RGB transfer belongs
        # to the asynchronous stager, never a global adoption barrier.
        if not (V/'operations/future_rgb_sg'/('%06d'%frame)/'staged.json').exists():
            continue
        cameras = io.read(V / 'sources' / str(frame) / 'source_ready.json')['cameras']
        for camera in cameras:
            key = '%06d_%s' % (frame, camera)
            if key not in state['maps']:
                name = 'future%s_a00' % key; state['maps'][key] = dict(frame=frame, camera=camera, name=name, attempt=0, host=None)
                state['jobs'][name] = dict(status='pending')
            if camera not in {'AA', 'AB', 'AC', 'AD', 'BA', 'BB'}:
                continue
            output = V / 'frames' / ('%06d' % frame) / 'data/lookcloser_frequencies' / ('train_' + camera + '.pt')
            if not output.exists() or not output.with_suffix('.json').exists():
                continue
            batch_import=V/'operations/frequency10/imports'/('%06d.json'%frame)
            imported_batch=io.read(batch_import)if batch_import.exists()else{}
            imported_record=imported_batch.get('records',{}).get(camera)
            if imported_record and io.sha(output)==imported_record['LOCAL_map_sha256']and io.sha(output.with_suffix('.json'))==imported_record['LOCAL_metadata_sha256']:
                result=dict(frame=frame,camera=camera,preverified_batch_import=str(batch_import),**imported_record)
            else:result = publish_local_map(c, frame, camera, io)
            if result:
                item = state['maps'][key]; job = state['jobs'][item['name']]
                if job.get('host') or job['status'] not in {'pending', 'complete'}:
                    raise ValueError('Legacy-six map has another live or unresolved owner')
                item.update(published=True, publication=result, imported_legacy_six=True)
                job['status'] = 'complete'; imported.append(key)
    state['unified_version'] = 10; state['imported_legacy_six'] = imported
    io.write(root / 'state.json', state); io.write(root / 'adoption.json', dict(imported_legacy_six=imported, same_Silver_job_namespace=True,
             exclusive_six_retirement_sha256=io.sha(V / 'operations/future_frequency/exclusive_service_retired.json'), time=time.time()))


def ordered_map_keys(state):
    return sorted(state['maps'],key=lambda key:(state['maps'][key]['frame'],state['maps'][key]['camera']))


def run(c):
    require_release(c)
    V = Path(c['root']); root = Path(c['operations']); fleet, io, legacy, early = imports(V)
    held = fleet.lock(root / 'controller.lock')
    if held is None:
        return
    guard(c, io); state = io.read(root / 'state.json'); io.write(root / 'controller_identity.json', fleet.identity())
    staging_errors=[]
    def stager():
        try:
            for frame in c['frames']:
                while not (V / 'sources' / str(frame) / 'source_ready.json').exists():
                    time.sleep(5)
                stage_frame(c, frame, io, early)
        except BaseException as error:
            staging_errors.append(str(error))
            io.write(root/'staging_failure.json',dict(error=str(error),time=time.time()))
    thread = threading.Thread(target=stager, daemon=True); thread.start()
    while True:
        guard(c, io)
        if staging_errors:raise RuntimeError('Source-only staging failed: '+staging_errors[-1])
        for frame in c['frames']:
            if not (V / 'operations/future_rgb_sg' / ('%06d' % frame) / 'staged.json').exists():
                continue
            for camera in io.read(V / 'sources' / str(frame) / 'source_ready.json')['cameras']:
                key = '%06d_%s' % (frame, camera)
                if key not in state['maps']:
                    name = 'future%s_a00' % key; state['maps'][key] = dict(frame=frame, camera=camera, name=name, attempt=0, host=None)
                    state['jobs'][name] = dict(status='pending')
        allowed, priority = allowed_slots(c, V, legacy, io); priority.update(allowed_gpus=allowed, time=time.time())
        io.write(root / 'priority.json', priority)
        # Compatibility view for the released explicit native drain/MAT ack.
        compatibility = dict(priority, eligible_map_gpus=allowed['silver'], producer=priority.get('legacy_native', {}).get('producer', {}))
        io.write(V / 'operations/future_rgb_sg/priority.json', compatibility)
        jobs = []; available=[]
        busy={(j.get('host'),j.get('gpu'))for j in state['jobs'].values()if j['status']not in {'complete','failed','skipped','blocked','pending'}}
        for host_key,snapshot in state.get('host_snapshots',{}).items():
            if snapshot.get('error'):continue
            for gpu in allowed.get(host_key,[]):
                d=snapshot.get('gpus',{}).get(str(gpu),{})
                if d and not d['processes']and d.get('used_mib',999999)<1024 and(host_key,gpu)not in busy:
                    available.append((host_key!='silver',gpu,host_key))
        available.sort()
        for key in ordered_map_keys(state):
            item = state['maps'][key]
            current = state['jobs'][item['name']]
            if current['status'] == 'complete' and not item.get('published'):
                if item.get('host')=='clever':
                    try:item['publication']=clever(c).finish(c,item,current)
                    except Exception as error:
                        item['publication_error']=str(error);allowed['clever']=[]
                        io.write(root/'clever_publication_wait.json',dict(frame=item['frame'],camera=item['camera'],error=str(error),time=time.time()));continue
                if item.get('host') in {'olive', 'dev3'}:
                    item['publication'] = publish_local_map(c, item['frame'], item['camera'], io)
                    if not item['publication']:raise ValueError('Completed local map lacks a verified canonical publication')
                item['published'] = True
            if item.get('published'):
                continue
            if current.get('host'):
                item['host']=current['host'];item['pinned_host']=current['host']
            elif not item.get('pinned_host'):
                item['host']=None;item.pop('spec',None)
            if item.get('host') is None:
                # Assign only after a fresh physical probe; empty slots are
                # transient, while each scientific map keeps one durable host.
                if not available:
                    continue
                _, _, item['host'] = available.pop(0)
                if item['host']=='clever':
                    # Durable logical owner precedes every remote stage sideeffect.
                    item['pinned_host']='clever';io.write(root/'state.json',state)
                elif item['host'] != 'silver':
                    item['spec'] = str(local_spec(c, item['frame'], item['camera'], io))
                io.write(root / 'state.json', state)
            host_key = item['host']
            if host_key=='clever'and not item.get('spec')and allowed.get('clever'):
                try:
                    item['spec']=clever(c).stage(c,item['frame'],item['camera']);item.pop('staging_error',None);io.write(root/'state.json',state)
                except Exception as error:
                    item['staging_error']=str(error);allowed['clever']=[];io.write(root/'state.json',state)
                    io.write(root/'clever_staging_wait.json',dict(frame=item['frame'],camera=item['camera'],error=str(error),time=time.time()));continue
            if not allowed.get(host_key) or (current.get('gpu') is not None and current['gpu'] not in allowed[host_key]):
                continue
            jobs.append(map_job(c,item))
        tick_hosts = {h: dict(row, gpus=allowed[h]) for h, row in c['hosts'].items()}
        fleet.tick(dict(schema_version=1, poll_seconds=5, hosts=tick_hosts, jobs=jobs), state, root)
        for item in state['maps'].values():
            job = state['jobs'][item['name']]
            if job.get('host'):
                item['host']=job['host'];item['pinned_host']=job['host']
            if job['status'] == 'failed':
                proof = state.get('host_snapshots', {}).get(item['host'], {}).get('jobs', {}).get(item['name'])
                if proof and not proof.get('supervisor_alive') and not proof.get('worker_alive') and item['attempt'] < 12:
                    item['attempt'] += 1; item['name'] = 'future%06d_%s_a%02d' % (item['frame'], item['camera'], item['attempt'])
                    state['jobs'][item['name']] = dict(status='pending')
        io.write(root / 'state.json', state); statuses = [state['jobs'][r['name']]['status'] for r in state['maps'].values()]
        io.write(root / 'progress.json', dict(time=time.time(), maps=len(statuses), counts={s: statuses.count(s) for s in set(statuses)},
                 all66_cameras=True, reserved_six=False, priority=priority, prepared_publication_allowed=False))
        if len(statuses) == len(c['frames']) * 66 and all(state['maps'][key].get('published') for key in state['maps']):
            io.write(root / 'complete.json', dict(maps=len(statuses), time=time.time())); return
        time.sleep(5)


def watchdog(c, path):
    require_release(c)
    V = Path(c['root']); root = Path(c['operations']); fleet, io, _, _ = imports(V); held = fleet.lock(root / 'watchdog.lock')
    if held is None:
        return
    io.write(root / 'watchdog_identity.json', fleet.identity())
    while not (root / 'complete.json').exists():
        guard(c, io); identity = io.read(root / 'controller_identity.json') if (root / 'controller_identity.json').exists() else None
        if not fleet.alive(identity):
            with (root / 'controller.log').open('ab') as stream:
                child = subprocess.Popen([sys.executable, __file__, 'run', '--config', str(path)], stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            fleet.append(root / 'watchdog_checks.jsonl', dict(time=time.time(), started=fleet.identity(child.pid)))
        time.sleep(15)



def configuration(V):
    """Create an unapproved candidate only; never mutate activeV12 configuration."""
    fleet,io,_,_=imports(V);c=copy.deepcopy(io.read(V/'operations/frequency10/config_v12.json'))
    C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');base=Path('/home/ubuntu/lookcloser_private/marshawn_frequency_canary_20261108');R=Path(c['hosts']['silver']['campaign']);ops=V/'operations/frequency13';helper=ops/'code/marshawn_frequency_remote_store.py'
    c.update(version=13,release_revision=13,authorized_main_jobs=False,config_path=str(V/'operations/frequency10/config_v13.json'),ROOT_release=str(ops/'root_release.json'),source_code_sha256=io.sha(__file__))
    c['hosts']['clever']=dict(address='ubuntu@clever-shadow',python='/home/ubuntu/sam3env/bin/python',root=str(base/'operations/frequency13/dispatch'),campaign=str(base),controller=str(base/'code/LookCloser/scripts/marshawn_tuning_controller.py'),gpus=[0],role='frequency_only_nonshared')
    c['clever_backend']=dict(private_root=str(base),transactions=str(ops/'transactions'),archive_root=str(R/'operations/clever_frequency_archive_v13'),Clever_helper=str(base/'operations/frequency13/helpers/marshawn_frequency_remote_store.py'),SG_helper=str(R/'operations/clever_frequency_archive_v13/helpers/marshawn_frequency_remote_store.py'),helper_sha256=io.sha(helper),worker_sha256=io.sha(V/'code/LookCloser/scripts/marshawn_video_frequency.py'),preprocess_source_sha256=io.sha(V/'code/nerfstudio/scripts/lookcloser_preprocess.py'),bundle_manifest_sha256='1c0667c8446666c2b2c39636b624de2e705af9eed0cb9b11735965cfe9e3b290',GPU_uuid='GPU-a8fe37e8-8d05-09ff-6359-9fc613d3d63f',max_private_bytes=48*2**30,min_free_bytes=32*2**30,namespace='frequency13',qualification_only=False)
    for path in [Path(__file__),helper,ops/'code/marshawn_clever_frequency_backend.py',C/'operations/clever_frequency_canary_20261108/canary_review.json',C/'operations/frequency_variability_control_20261108/comparison.json']:
        c['files_sha256'][str(path)]=io.sha(path)
    path=ops/'config_v13_candidate.json'
    if path.exists()and io.read(path)!=c:raise ValueError('Candidate changed; preserve/version oldcandidate before resealing')
    io.write(path,c);return path

def preflight(c):
    """Exact existingV12 materialization proof, including Olive/DEV3 workers."""
    V=Path(c['root']);fleet,io,_,_=imports(V);previous=io.read(V/'operations/frequency10/config_v12.json');state=io.read(Path(c['operations'])/'state.json')
    source=importlib.util.spec_from_file_location('_v12_materialization_reference',V/'operations/frequency10/code/marshawn_frequency12.py');old=importlib.util.module_from_spec(source);source.loader.exec_module(old)
    checked=0;live={}
    for item in state['maps'].values():
        job=state['jobs'][item['name']];host=job.get('host')or item.get('host')
        if not job.get('spec_sha256')or host not in previous['hosts']:continue
        row=dict(item,host=host)
        a=fleet.materialize(old.map_job(previous,row),previous['hosts'][host],job['gpu']);b=fleet.materialize(map_job(c,row),c['hosts'][host],job['gpu'])
        for value in [a,b]:value['poll_seconds']=5;value['env']['TUNING_DEPENDENCIES_JSON']='{}'
        if a!=b or fleet.digest(b)!=job['spec_sha256']:raise ValueError('ExistingV12 scientific job spec changed '+item['name'])
        checked+=1
        if job['status']not in ['complete','failed','skipped','blocked']:live[item['name']]=dict(host=host,gpu=job['gpu'],status=job['status'],worker=job.get('worker'),supervisor=job.get('supervisor'),spec_sha256=job['spec_sha256'])
    result=dict(kind='READ_ONLY_V12_TO_V13_EXACT_SPEC_ADOPTION_PREFLIGHT',checked_existing_jobs=checked,preserved_live=live,host_configuration_unchanged=all(c['hosts'][k]==v for k,v in previous['hosts'].items()),no_GPU_or_mainclaim_started=True,time=time.time())
    io.write(V/'operations/frequency13/qualification/preflight.json',result);return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['configure','preflight','run','watchdog'])
    p.add_argument('--campaign', type=Path, default=V_DEFAULT); p.add_argument('--config', type=Path); p.add_argument('--frame', type=int); p.add_argument('--payload', type=Path)
    a = p.parse_args()
    if a.action == 'configure':
        print(configuration(a.campaign))
    elif a.action == '_install':
        V = a.campaign; _, _, legacy, _ = imports(V)
        print(json.dumps(legacy.install(a.payload, V / 'frames' / ('%06d' % a.frame) / 'data', V / 'code', V / 'operations/future_rgb_sg' / ('%06d' % a.frame) / 'specs')))
    else:
        c = json.loads(a.config.read_text()); _, io, _, _ = imports(Path(c['root'])); guard(c, io)
        print(json.dumps(preflight(c)))if a.action=='preflight'else print(json.dumps(precopy_six(c)))if a.action=='precopy-six'else adopt(c) if a.action == 'adopt' else run(c) if a.action == 'run' else watchdog(c, a.config)
