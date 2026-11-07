"""One durable V3 call: source-ready -> LOCAL+idle-SG masks/candidates ALL66.

Crew/fragment/reference qualification, DATA, MAT/admission and dispatch remain
separate mandatory stages. This service never publishes prepared/dispatch.
"""
from pathlib import Path
import argparse,importlib,json,os,shlex,subprocess,sys,time
import marshawn_sam3_assist as io


def guard_release(V):
    release=io.read(V/'operations/sam3_assist/full_release.json')
    if release.get('approved_transport_math') is not True or release.get('prepared_publication_allowed') is not False:raise ValueError('Operational full-mask release required')
    for path,digest in release['files_sha256'].items():
        if io.sha(path)!=digest:raise ValueError('Released full-mask helper changed: '+path)


def fleet_for(V):
    sys.path.insert(0,str(V/'code/LookCloser/scripts'));return importlib.import_module('marshawn_tuning_controller')


def start(argv,log):
    with Path(log).open('ab') as stream:
        return subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def validate_ready(V,frame,marker):
    result=io.read(marker);source=V/'sources'/str(frame)/'source_ready.json';ready=io.read(source)
    if result['frame']!=frame or result['source_ready_sha256']!=io.sha(source) or result['cameras']!=ready['cameras']:raise ValueError('Completed mask source binding changed')
    native=V/'frames'/('%06d'%frame)/'native_masks';people=native.parent/'person_candidates';rows={r['camera']:r for r in ready['records']}
    for camera in ready['cameras']:
        m=io.read(native/(camera+'.json'));p=io.read(people/(camera+'.json'));r=rows[camera]
        if m['source_receipt_sha256']!=r['receipt_sha256'] or p['source_receipt_sha256']!=r['receipt_sha256']:raise ValueError('Completed output source changed')
        if io.sha(native/(camera+'.npz'))!=m['archive_sha256'] or io.sha(people/(camera+'.npz'))!=p['archive_sha256']:raise ValueError('Completed native/candidate bytes changed')
        if io.sha(native/(camera+'.json'))!=result['records'][camera]:
            # V3 legitimately applies its qualified crew/fragment guard after
            # this canonical stage. Adopt that work, never regenerate over it.
            guard=m.get('crew_guard',{})
            if not guard.get('receipt_path') or io.sha(native.parent/guard['receipt_path'])!=guard.get('receipt_sha256'):
                raise ValueError('Canonical output changed without qualified crew receipt')
    return result


def props_inputs(V,frame,cameras):
    """Retain the sibling MAT worker's exact portable candidate-task contract."""
    import marshawn_sam3_candidates as candidates
    S='/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107';remote=Path(S)/'operations/sam3_assist'/('candidates%06d'%frame)
    local=V/'operations/sam3_assist'/('candidates%06d'%frame);tasks=local/'tasks'
    for camera in cameras:candidates.pack(V,frame,camera,tasks/camera,str(remote/'tasks'/camera),str(Path(S)/'code'),str(Path(S)/'tools/sam3_overlay'))
    subprocess.run(['ssh','-o','BatchMode=yes','ubuntu@silver-sable',shlex.join(['mkdir','-p',str(remote/'tasks')])],check=True)
    subprocess.run(['rsync','-rt','--exclude=raw_native.png','--exclude=returned/',str(tasks)+'/','ubuntu@silver-sable:'+str(remote/'tasks')+'/'],check=True)
    script="""from pathlib import Path
import sys,json,os
R=Path(sys.argv[1]);F=Path(sys.argv[2]);missing=[]
for camera in json.loads(sys.argv[3]):
 dst=R/'tasks'/camera/'raw_native.png';src=F/'tasks'/camera/'raw_native.png'
 if not dst.exists():
  if src.exists():os.link(src,dst)
  else:missing.append(camera)
print(json.dumps(missing))"""
    command=['/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python','-c',script,str(remote),str(Path(S)/'operations/sam3_assist'/('frame%06d'%frame)),json.dumps(cameras)]
    probe=subprocess.run(['ssh','-o','BatchMode=yes','ubuntu@silver-sable',shlex.join(command)],capture_output=True,text=True,check=True)
    for camera in json.loads(probe.stdout):
        subprocess.run(['rsync','-rLt',str(tasks/camera/'raw_native.png'),'ubuntu@silver-sable:'+str(remote/'tasks'/camera)+'/'],check=True)
    verify="""from pathlib import Path
import sys,json,hashlib
R=Path(sys.argv[1]);records={}
for camera in json.loads(sys.argv[2]):
 p=R/'tasks'/camera;s=json.loads((p/'candidate_task.json').read_text());h=hashlib.sha256()
 with (p/'raw_native.png').open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==s['raw_native_image_sha256'];records[camera]=h.hexdigest()
(R/'portable_inputs_ready.json').write_text(json.dumps({'frame':int(sys.argv[3]),'records':records,'camera_count':66,'prepared_publication_allowed':False},indent=2))
print('portable_native_inputs66_verified')"""
    command=['/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python','-c',verify,str(remote),json.dumps(cameras),str(frame)]
    subprocess.run(['ssh','-o','BatchMode=yes','ubuntu@silver-sable',shlex.join(command)],check=True)


def local_job(V,root,frame,cameras,name):
    fleet=fleet_for(V);code=V/'operations/sam3_assist/code';python='/home/pavlo/.conda/envs/nerfstudio/bin/python'
    host=dict(python=python,root=str(root/name),controller=str(V/'code/LookCloser/scripts/marshawn_tuning_controller.py'),gpus=[0])
    job=dict(name=name,host='olive',command=[python,str(code/'marshawn_sam3_local_observer.py'),'--campaign',str(V),'--frame',str(frame),'--cameras']+cameras,
        env={'LOCAL_SAM3_LEASE_RECEIPT':'{job_root}/receipt.json'})
    plan=dict(schema_version=1,poll_seconds=5,hosts={'olive':host},jobs=[job]);manifest=root/(name+'.json')
    if manifest.exists() and io.read(manifest)!=plan:raise ValueError('LOCAL camera assignment changed')
    io.write(manifest,plan);jobroot=root/name
    subprocess.run([python,str(V/'code/LookCloser/scripts/marshawn_tuning_controller.py'),'launch','--root',str(jobroot),'--manifest',str(manifest)],check=True,capture_output=True)
    while True:
        state=fleet.read(jobroot/'state.json',{}).get('jobs',{}).get(name,{})
        if state.get('status')=='complete':return
        if state.get('status')=='failed':raise RuntimeError('LOCAL native-mask worker failed; source/candidate artifacts retained')
        time.sleep(5)


def coordinate(V,frame):
    V=V.resolve();root=V/'operations/sam3_assist'/('full%06d'%frame);root.mkdir(parents=True,exist_ok=True);fleet=fleet_for(V)
    guard_release(V)
    held=fleet.lock(root/'coordinator.lock')
    if held is None:return
    io.write(root/'coordinator_identity.json',fleet.identity())
    source=V/'sources'/str(frame)/'source_ready.json'
    while not source.exists():io.write(root/'progress.json',dict(frame=frame,stage='waiting_source_ready',time=time.time()));time.sleep(10)
    ready=io.read(source);cameras=ready['cameras']
    if ready['frame']!=frame or len(cameras)!=66 or ready['source_verified_all66'] is not True:raise ValueError('Unqualified ALL66 source')
    code=V/'operations/sam3_assist/code';assignment=V/'operations/sam3_assist'/('frame%06d'%frame)/'assignment.json'
    sg_complete=assignment.parent/'assist_complete.json';sg_identity=root/'sg_identity.json'
    if not sg_complete.exists():
        old=io.read(sg_identity) if sg_identity.exists() else None
        if not fleet.alive(old):
            argv=[sys.executable,str(code/'marshawn_sam3_assist_controller.py'),'--campaign',str(V),'--frame',str(frame),'--cameras']+cameras+[
                '--gpus','1','5','6','7','--remote-root','/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107',
                '--remote-python','/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python',
                '--release',str(V/'operations/sam3_assist/transport_release.json')]
            p=start(argv,root/'SG.log');io.write(sg_identity,fleet.identity(p.pid))
    while not assignment.exists():
        if sg_identity.exists() and not fleet.alive(io.read(sg_identity)):raise RuntimeError('SG assignment failed before publication')
        time.sleep(2)
    partition=io.read(assignment);local=partition['local_cameras']
    if local:local_job(V,root,frame,local,'local_primary_%06d'%frame)
    while not sg_complete.exists():
        if sg_identity.exists() and not fleet.alive(io.read(sg_identity)):raise RuntimeError('SG controller stopped; outer watchdog will adopt/restart')
        time.sleep(5)
    # Compute only missing views after remote workers finish; already verified
    # canonical files are retained by the native worker's own same-view checks.
    missing=[];native=V/'frames'/('%06d'%frame)/'native_masks';people=native.parent/'person_candidates'
    for camera in cameras:
        if not (native/(camera+'.json')).exists() or not (people/(camera+'.json')).exists():missing.append(camera)
    if missing:local_job(V,root,frame,missing,'local_fallback_%06d'%frame)
    records={};candidate_records={}
    source_rows={r['camera']:r for r in ready['records']}
    for camera in cameras:
        m=io.read(native/(camera+'.json'));p=io.read(people/(camera+'.json'));r=source_rows[camera]
        if m['frame']!=frame or m['camera']!=camera or m['source_receipt_sha256']!=r['receipt_sha256'] or p['source_receipt_sha256']!=r['receipt_sha256']:
            raise ValueError('ALL66 source mapping differs')
        if io.sha(native/(camera+'.npz'))!=m['archive_sha256'] or io.sha(people/(camera+'.npz'))!=p['archive_sha256']:raise ValueError('Native mask/candidate changed')
        records[camera]=io.sha(native/(camera+'.json'));candidate_records[camera]=io.sha(people/(camera+'.json'))
    result=dict(frame=frame,cameras=cameras,records=records,person_candidate_records=candidate_records,
        source_ready_sha256=io.sha(source),prepared_publication_allowed=False,
        next_required_stage='Current crew/fragment/reference qualification then frozen DATA; MAT/admission before dispatch')
    props_inputs(V,frame,cameras)
    io.write(native/'batch_ready.json',result);io.write(root/'mask_assist_ready.json',result)


def watchdog(V,frame):
    fleet=fleet_for(V);root=V/'operations/sam3_assist'/('full%06d'%frame);root.mkdir(parents=True,exist_ok=True)
    guard_release(V)
    held=fleet.lock(root/'watchdog.lock')
    if held is None:return
    io.write(root/'watchdog_identity.json',fleet.identity())
    while not (root/'mask_assist_ready.json').exists():
        p=root/'coordinator_identity.json';old=io.read(p) if p.exists() else None
        if not fleet.alive(old):
            child=start([sys.executable,str(Path(__file__).resolve()),'coordinate','--campaign',str(V),'--frame',str(frame)],root/'coordinator.log')
            fleet.append(root/'watchdog_checks.jsonl',dict(action='coordinator_started',process=fleet.identity(child.pid),time=time.time()))
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',choices=['run','coordinate','watchdog'],default='run')
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args();V=a.campaign.resolve()
    if a.action=='coordinate':coordinate(V,a.frame)
    elif a.action=='watchdog':watchdog(V,a.frame)
    else:
        guard_release(V)
        fleet=fleet_for(V);root=V/'operations/sam3_assist'/('full%06d'%a.frame);root.mkdir(parents=True,exist_ok=True)
        identity=root/'watchdog_identity.json';old=io.read(identity) if identity.exists() else None
        if not fleet.alive(old):start([sys.executable,str(Path(__file__).resolve()),'watchdog','--campaign',str(V),'--frame',str(a.frame)],root/'watchdog.log')
        while not (root/'mask_assist_ready.json').exists():time.sleep(5)
        print(json.dumps(validate_ready(V,a.frame,root/'mask_assist_ready.json')))
