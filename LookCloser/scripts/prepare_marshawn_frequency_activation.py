"""Stage126 frozen HD frequency fits with paired camera factors on the same host/GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main(campaign,root,remote_root,factory_receipt):
    if root.exists():raise ValueError('Preserve prepared generations')
    source=campaign/'frequency_variants_HD63_prepared_v1';ready=read(campaign/'data/ready.json');transforms=read(campaign/'data/transforms.json')
    original=read(source/'preparation.json')
    for relative,digest in original['inventory'].items():
        if sha(source/relative)!=digest:raise ValueError('Prepared input changed: '+relative)
    expected={'worker.py':'13a3d723cb5462eeadead9df5862cd57eba37c5b351a5e9d417f0a5be524d91e','preprocess.py':'729f10f26b24b97f56adcc3c62a6722a78c4ee5d59d5b0b8dd963fde6561ffad','gpu_isolation.py':'42a2155aefdeae4ceeca441a28c417450ef49dee41b3a6cae6beb767dc53069b'}
    for name,digest in expected.items():assert sha(source/name)==digest
    assert sha(factory_receipt)=='d7bbc11107d78defb366ed6135c6d2cf0410323335c8bd926f45588fb0c6a777'
    records=sorted((r for r in ready['records']if r['camera']not in {'DC','HC','KA'}),key=lambda r:r['camera'])
    assert len(records)==63 and {r['image']for r in records}==set(transforms['train_filenames'])
    root.mkdir();(root/'science').mkdir();(root/'transport').mkdir();(root/'scripts').mkdir();(root/'evidence').mkdir()
    for name in expected:shutil.copy2(source/name,root/'science'/name)
    scripts=Path(__file__).resolve().parent
    controller=scripts/'marshawn_tuning_controller.py';assert sha(controller)=='2e7a1be53794882ffc3c0a6a82fff3cc24070b36fbf7593e65771772e0dadf39'
    shutil.copy2(controller,root/'transport/controller.py')
    for name in ['prepare_marshawn_frequency_activation.py','launch_marshawn_frequency_activation.py','validate_marshawn_frequency_activation.py','inspect_marshawn_frequency_runtime.py','reconcile_marshawn_frequency_activation.py','check_marshawn_frequency_activation_paths.py','bind_marshawn_frequency_activation.py']:
        shutil.copy2(scripts/name,root/'scripts'/name)
    shutil.copy2(factory_receipt,root/'evidence/factory_complete.json')
    remote_campaign=remote_root.parent
    host_roots={'local':root,'silver':remote_root};data_roots={'local':campaign/'data','silver':remote_campaign/'data'}
    python={'local':'/home/pavlo/.conda/envs/nerfstudio/bin/python','silver':'/home/ubuntu/lookcloser/envs/marshawn_leader/bin/python'}
    code={h:p/'semantic_actor_candidate_v2_20261106/code'for h,p in [('local',campaign),('silver',remote_campaign)]}
    hosts={h:dict(root=str(p/'host'),python=python[h],controller=str(p/'transport/controller.py'),gpus=[0]if h=='local'else[4,7],**({'address':'ubuntu@silver-sable'}if h=='silver'else{}))for h,p in host_roots.items()}
    jobs=[];inventory=[];changes=[]
    for index,row in enumerate(records):
        camera=row['camera'];host,gpu=[('local',0),('silver',4),('silver',7)][index%3];package=host_roots[host];data=data_roots[host]
        assert sha(campaign/'data'/row['image'])==row['rgb_sha256']
        for variant in ['lr_decay','threshold093']:
            old=read(source/'specs'/(camera+'_'+variant+'.json'))
            name='freqHD63_v1_'+camera+'_'+variant;relative='specs/'+camera+'_'+variant+'.json';inputs={str(package/'science'/n):h for n,h in expected.items()}
            inputs.update({str(data/n):sha(campaign/'data'/n)for n in ['ready.json','transforms.json',row['image']]})
            spec=dict(old,data=str(data),output=str(package/'host/jobs'/name/'run'),preprocess=str(package/'science/preprocess.py'),gpu_isolation=str(package/'science/gpu_isolation.py'),inputs_sha256=inputs,images={camera:str(data/row['image'])})
            assert spec['shape']==[1012,1920]and spec['steps_per_level']==1000 and spec['variants']==[variant]and spec['worker_sha256']==expected['worker.py']and spec['controller_owned_lease']
            scientific={k:v for k,v in old.items()if k not in ['data','output','preprocess','gpu_isolation','inputs_sha256','images']}
            assert scientific=={k:v for k,v in spec.items()if k in scientific}
            write(root/relative,spec)
            inventory.append(dict(name=name,camera=camera,variant=variant,host=host,gpu=gpu,spec=str(package/relative),spec_relative=relative,spec_sha256=sha(root/relative),original_prepared_spec_sha256=sha(source/'specs'/(camera+'_'+variant+'.json')),source_image_sha256=row['rgb_sha256']))
            changes.append(dict(camera=camera,variant=variant,changed_fields=[k for k in spec if spec[k]!=old[k]],scientific_values_unchanged=True))
            jobs.append(dict(name=name,host=host,gpu=gpu,dependencies=[],command=[python[host],str(package/'science/worker.py'),'--plan',str(package/relative),'--pinned-worker','--lease-receipt','{job_root}/receipt.json'],cwd=str(code[host]/'LookCloser'),env=dict(PYTHONPATH=str(code[host]),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),progress_files=['{job_root}/run/progress.json'],metrics_file='{job_root}/run/complete.json'))
    assert len(jobs)==126 and len({(j['camera'],j['variant'])for j in inventory})==126
    write(root/'controller_plan.json',dict(schema_version=1,poll_seconds=10,hosts=hosts,jobs=jobs));write(root/'jobs.json',inventory);write(root/'path_only_changes.json',changes)
    write(root/'scope.json',dict(prepared_only=True,map_fits_authorized=False,NeRF_training_authorized=False,assembly_authorized=False,READY_writes=False,jobs=126,cameras=63,variants={'lr_decay':dict(SSIM=.95,LR_initial=.01,LR_final_per_level=.001),'threshold093':dict(SSIM=.93,LR_initial=.01,LR_final_per_level=None)},shape=[1012,1920],steps_per_level=1000,levels=16,updates_per_fit=16000,allocator_cap_GiB=4,expected_fit_seconds=76,expected_three_lane_wall_minutes='roughly55–70 including dispatch/import overhead; not a timeout guarantee',source_prepared=str(source),source_preparation_sha256=sha(source/'preparation.json'),science_sha256=expected,controller_sha256=sha(controller),factory_receipt_sha256=sha(factory_receipt),same_camera_factor_host_and_gpu_paired=True,host_job_counts={h:sum(j['host']==h for j in inventory)for h in hosts},failure_policy='Failed fit receipts/output directories are immutable. No automatic retry/replacement; a separately reviewed retry generation may later fill a missing camera/factor. SSH UNKNOWN never implies stopped. Controller resume preserves durable claims and completed prefix.',assembly_policy='After all126 fits, separate CPU validation and source/visual review. No dataset assembly, READY mutation or NeRF jobs in this activation.'))
    print(json.dumps(dict(root=str(root),jobs=len(jobs),launched=False,requires_staging_bindings_runtime_and_root_release=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['campaign','root','remote-root','factory-receipt']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.campaign,a.root,a.remote_root,a.factory_receipt)
