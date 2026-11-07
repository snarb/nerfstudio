"""Optional external per-camera SAM3 assist. Frozen campaign science is read-only.

Run only after assist_release.json binds the reviewed bitwise canary/helper.
Writes assignment.json before staging; caller can run canonical LOCAL cameras
in parallel. After this returns, canonical LOCAL ALL66 safely skips completed
views and handles listed fallback cameras, publishing its usual batch receipt.
"""
from pathlib import Path
import argparse,importlib,json,math,os,shlex,subprocess,sys,time
import shutil
import marshawn_sam3_assist as assist


def command(argv,**kwargs):return subprocess.run(argv,check=True,**kwargs)


def remote(host,argv):
    return command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,shlex.join(argv)],capture_output=True,text=True,timeout=120)


def native_queue_inactive(state,now,timeout):
    """No LOCAL fallback while multi-wave work or UNKNOWN owners remain."""
    active={'starting','running','dispatching','unknown','orphaned'}
    healthy=False
    for item in state['cameras'].values():
        job=state['jobs'][item['name']]
        if job['status'] in active or (job['status'] not in {'complete','failed','skipped','blocked'} and (job.get('supervisor_alive') or job.get('worker_alive'))):healthy=True
        if job['status']=='complete' and not item.get('collected'):healthy=True
    if healthy:
        state['native_inactive_since']=None;return False
    if state.get('native_inactive_since') is None:state['native_inactive_since']=now
    return now-state['native_inactive_since']>=timeout


def can_fallback(job,queue_inactive):
    return bool(queue_inactive and job['status']=='pending' and not job.get('host'))


def verify_return(task):
    import numpy as np
    spec=assist.read(task/'task.json');returned=task/'returned';proof=assist.read(returned/'operation_receipt.json')
    required={'task_sha256':assist.sha(task/'task.json'),'canonical_mask_sha256':spec['canonical_mask_sha256'],
        'cleanup_sha256':spec['cleanup_sha256'],'adapter_sha256':spec['adapter_sha256'],'checkpoint_sha256':spec['checkpoint_sha256'],
        'original_source_receipt_sha256':spec['original_source_receipt_sha256'],
        'portable_source_receipt_sha256':spec['portable_source_receipt_sha256'],
        'raw_native_image_sha256':spec['raw_native_image_sha256'],'reference_blob_sha256':spec['reference']['blob_sha256'],
        'identity_reference_frame':spec['reference']['frame'],'identity_reference_sha256':spec['reference']['archive_sha256']}
    if any(proof.get(k)!=v for k,v in required.items()):raise ValueError('Remote result input/code mapping differs')
    camera=spec['camera'];masks=returned/'native_masks';record=assist.read(masks/(camera+'.json'))
    if assist.sha(masks/(camera+'.npz'))!=proof['archive_sha256'] or assist.sha(masks/(camera+'.json'))!=proof['remote_mask_receipt_sha256']:
        raise ValueError('Remote result artifact SHA differs')
    for key,value in dict(frame=spec['frame'],camera=camera,source_receipt_sha256=spec['portable_source_receipt_sha256'],
        raw_native_image_sha256=spec['raw_native_image_sha256'],mask_code_sha256=spec['canonical_mask_sha256'],
        checkpoint_sha256=spec['checkpoint_sha256'],identity_reference_frame=spec['reference']['frame'],
        identity_reference_sha256=spec['reference']['archive_sha256'],native_cleanup_version=1).items():
        if record.get(key)!=value:raise ValueError('Canonical result receipt differs: '+key)
    with np.load(masks/(camera+'.npz'),allow_pickle=False) as z:
        if set(z.files)!={'black','grey','source_ineligible','unknown','crop_xyxy'}:raise ValueError('Mask output schema differs')
        for key in z.files:
            if assist.array_sha(z[key])!=proof['arrays_sha256'][key]:raise ValueError('Returned array SHA differs')
            if key!='crop_xyxy' and (z[key].dtype!=np.bool_ or z[key].shape!=(4320,8192)):raise ValueError('Native raster/dtype differs')
    if not proof.get('person_candidates'):raise ValueError('Fresh current-frame person candidates required before crew review')
    candidates=returned/'person_candidates';candidate_record=assist.read(candidates/(camera+'.json'))
    if assist.sha(candidates/(camera+'.npz'))!=proof['person_candidates']['archive_sha256'] or assist.sha(candidates/(camera+'.json'))!=proof['person_candidates']['receipt_sha256']:
        raise ValueError('Current-frame person candidate artifact changed')
    for key,value in dict(frame=spec['frame'],camera=camera,source_receipt_sha256=spec['portable_source_receipt_sha256'],
        original_source_receipt_sha256=spec['original_source_receipt_sha256'],raw_native_image_sha256=spec['raw_native_image_sha256'],
        checkpoint_sha256=spec['checkpoint_sha256'],canonical_mask_sha256=spec['canonical_mask_sha256']).items():
        if candidate_record.get(key)!=value:raise ValueError('Current-frame candidates source differs')
    with np.load(candidates/(camera+'.npz'),allow_pickle=False) as z:
        if z['masks'].dtype!=np.bool_ or z['masks'].shape[1:]!=(4320,8192) or assist.array_sha(z['masks'])!=candidate_record['masks_sha256']:
            raise ValueError('Current-frame candidate tensor changed')
    gpu=proof['gpu']
    if gpu['actual_physical_uuid']!=gpu['expected_physical_uuid'] or gpu['visible_device_count']!=1:raise ValueError('GPU identity proof differs')
    return proof


def publish_candidates(campaign,task):
    """Candidate-only publication never modifies an existing canonical mask."""
    proof=verify_return(task);spec=assist.read(task/'task.json');camera=spec['camera']
    source=task/'returned/person_candidates';target=Path(campaign)/'frames'/('%06d'%spec['frame'])/'person_candidates';target.mkdir(exist_ok=True)
    if (target/(camera+'.json')).exists():
        existing=assist.read(target/(camera+'.json'))
        if existing['archive_sha256']!=proof['person_candidates']['archive_sha256'] or assist.sha(target/(camera+'.npz'))!=existing['archive_sha256']:
            raise ValueError('Existing current-frame person candidates differ; never overwrite')
        return
    record=assist.read(source/(camera+'.json'));provenance=target/'assist_provenance';provenance.mkdir(exist_ok=True)
    shutil.copy2(source/(camera+'.json'),provenance/(camera+'.remote.json'))
    shutil.copy2(task/'returned/operation_receipt.json',provenance/(camera+'.operation.json'))
    shutil.copy2(source/(camera+'.npz'),target/(camera+'.npz'))
    assist.write(target/(camera+'.json'),dict(record,source_receipt_sha256=spec['original_source_receipt_sha256'],
        assist_mapping=dict(remote_candidate_receipt_sha256=proof['person_candidates']['receipt_sha256'],
            operation_receipt_sha256=assist.sha(task/'returned/operation_receipt.json'),
            portable_source_receipt_sha256=spec['portable_source_receipt_sha256'],
            original_source_receipt_sha256=spec['original_source_receipt_sha256'])))


def run(args):
    V=args.campaign.resolve();base=V/'operations/sam3_assist';root=base/('frame%06d'%args.frame);root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(V/'code/LookCloser/scripts'));fleet=importlib.import_module('marshawn_tuning_controller')
    held=fleet.lock(root/'controller.lock')
    if held is None:raise RuntimeError('This frame assist controller is already active')
    release=assist.read(args.release)
    if release.get('approved') is not True or release.get('adapter_sha256')!=assist.sha(assist.__file__):raise ValueError('Reviewed canary release required')
    if release.get('canonical_mask_sha256')!=assist.sha(V/'code/LookCloser/scripts/marshawn_video_prep_masks.py'):raise ValueError('Canonical worker differs from reviewed canary')
    remote_base=Path(args.remote_root)/'operations/sam3_assist';remote_frame=remote_base/('frame%06d'%args.frame)
    host=dict(address=args.remote_host,python=args.remote_python,root=str(remote_frame/'dispatch'),
        controller=str(Path(args.remote_root)/'code/LookCloser/scripts/marshawn_tuning_controller.py'),gpus=args.gpus)
    assignment=root/'assignment.json'
    if assignment.exists():selected=assist.read(assignment)
    else:
        native=V/'frames'/('%06d'%args.frame)/'native_masks'
        if list(native.glob('*.json')) or list(native.glob('*.npz')):raise ValueError('Fresh partition required before any LOCAL mask worker starts')
        probe=fleet.host_call(host,['_probe','--root',host['root']]);probe.check_returncode();snapshot=json.loads(probe.stdout)
        idle=[g for g in args.gpus if not snapshot['gpus'][str(g)]['processes'] and snapshot['gpus'][str(g)]['used_mib']<512]
        local_count=args.local_count if args.local_count is not None else math.ceil(len(args.cameras)/(len(idle)+1))
        if not idle:local_count=len(args.cameras)
        local={args.cameras[int(i*len(args.cameras)/local_count)] for i in range(local_count)} if local_count else set()
        selected=dict(frame=args.frame,local_cameras=sorted(local),remote_cameras=[c for c in args.cameras if c not in local],
            idle_sg_gpus_at_partition=idle,all_cameras=args.cameras,
            canonical_followup='After assist completes, run canonical LOCAL ALL66; it verifies/skips finished views and computes fallbacks')
        assist.write(assignment,selected)
    if selected['frame']!=args.frame or selected['all_cameras']!=args.cameras:raise ValueError('Immutable camera partition changed')
    remote(host['address'],['mkdir','-p',str(remote_frame/'tasks'),str(remote_base/'code')])
    helper=base/'code/marshawn_sam3_assist.py'
    if assist.sha(helper)!=release['adapter_sha256']:raise ValueError('Operational helper snapshot differs')
    command(['rsync','-rt',str(helper),args.remote_host+':'+str(remote_base/'code')+'/'])
    for camera in selected['remote_cameras']:
        task=root/'tasks'/camera
        assist.pack(V,args.frame,camera,task,str(remote_frame/'tasks'/camera),str(Path(args.remote_root)/'code'),
                    str(Path(args.remote_root)/'tools/sam3_overlay'))
    command(['rsync','-rLt',str(root/'tasks')+'/',args.remote_host+':'+str(remote_frame/'tasks')+'/']) if selected['remote_cameras'] else None
    state=assist.read(root/'state.json') if (root/'state.json').exists() else dict(jobs={},cameras={})
    for camera in selected['remote_cameras']:
        if camera not in state['cameras']:
            name='f%06d_%s_a00'%(args.frame,camera);state['cameras'][camera]=dict(name=name,attempt=0,collected=False)
            state['jobs'][name]=dict(status='pending')
    start=time.time();assist.write(root/'controller.json',dict(process=fleet.identity(),started=start))
    while True:
        jobs=[]
        for camera,item in state['cameras'].items():
            if item.get('fallback'):continue
            jobs.append(dict(name=item['name'],host='silver',command=[args.remote_python,str(remote_base/'code/marshawn_sam3_assist.py'),
                'run','--spec',str(remote_frame/'tasks'/camera/'task.json')],env={'SAM3_ASSIST_LEASE_RECEIPT':'{job_root}/receipt.json'},
                metrics_file=str(remote_frame/'tasks'/camera/'operation_receipt.json')))
        fleet.tick(dict(schema_version=1,poll_seconds=10,hosts={'silver':host},jobs=jobs),state,root)
        inactive=native_queue_inactive(state,time.time(),args.max_pending_seconds)
        for camera,item in state['cameras'].items():
            job=state['jobs'][item['name']]
            if job['status']=='complete' and not item['collected']:
                task=root/'tasks'/camera;returned=task/'returned';(returned/'native_masks').mkdir(parents=True,exist_ok=True)
                command(['rsync','-rt',args.remote_host+':'+str(remote_frame/'tasks'/camera/'operation_receipt.json'),str(returned)+'/'])
                command(['rsync','-rt',args.remote_host+':'+str(remote_frame/'tasks'/camera/'campaign/frames'/('%06d'%args.frame)/'native_masks')+'/',str(returned/'native_masks')+'/'])
                (returned/'person_candidates').mkdir(exist_ok=True)
                command(['rsync','-rt',args.remote_host+':'+str(remote_frame/'tasks'/camera/'campaign/frames'/('%06d'%args.frame)/'person_candidates')+'/',str(returned/'person_candidates')+'/'])
                proof=verify_return(task)
                local=V/'frames'/('%06d'%args.frame)/'native_masks'/(camera+'.json')
                if local.exists():
                    comparison=assist.compare(V,task)
                    if not comparison['all_equal']:raise ValueError('Existing LOCAL output differs; never overwrite')
                    publish_candidates(V,task)
                else:assist.collect(V,task)
                item['collected']=True;item['archive_sha256']=proof['archive_sha256']
            elif job['status']=='failed':
                snapshot=state.get('host_snapshots',{}).get('silver',{});observed=snapshot.get('jobs',{}).get(item['name'])
                if observed and not observed.get('supervisor_alive') and not observed.get('worker_alive') and item['attempt']<3:
                    item['attempt']+=1;item['name']='f%06d_%s_a%02d'%(args.frame,camera,item['attempt']);state['jobs'][item['name']]={'status':'pending'}
                elif item['attempt']>=3:raise RuntimeError('Mask assist exhausted retries; retain outputs for review')
            elif can_fallback(job,inactive):
                item['fallback']=True;job['status']='skipped'
        assist.write(root/'state.json',state)
        pending=[c for c,i in state['cameras'].items() if not i['collected'] and not i.get('fallback')]
        summary=dict(frame=args.frame,remote_complete=[c for c,i in state['cameras'].items() if i['collected']],
            local_cameras=selected['local_cameras'],local_fallback_cameras=[c for c,i in state['cameras'].items() if i.get('fallback')],
            pending=pending,time=time.time())
        summary['publication_status']='canonical native masks only; mandatory crew_guard qualification before prepared_ready publication'
        summary['prepared_publication_allowed']=False
        summary['stage']='CANONICAL_TRANSPORT_COMPLETE_CREW_GUARD_REQUIRED' if not pending else 'CANONICAL_TRANSPORT_RUNNING'
        assist.write(root/'progress.json',summary)
        if not pending:assist.write(root/'assist_complete.json',summary);return summary
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--frame',type=int,required=True)
    p.add_argument('--cameras',nargs='+',required=True);p.add_argument('--gpus',nargs='+',type=int,default=list(range(1,8)))
    p.add_argument('--local-count',type=int);p.add_argument('--remote-root',required=True);p.add_argument('--remote-host',default='ubuntu@silver-sable')
    p.add_argument('--remote-python',required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--max-pending-seconds',type=float,default=60)
    a=p.parse_args();print(json.dumps(run(a)))
