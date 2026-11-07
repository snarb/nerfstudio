"""Seal the staged frequency-fit paths and protected sources; no GPU execution."""
import argparse,hashlib,importlib.util,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    if path.exists():raise ValueError('Preserve existing seal: '+str(path))
    path.write_text(json.dumps(value,indent=2)+'\n')


def main(root,campaign,remote_root):
    plan=read(root/'controller_plan.json');jobs=read(root/'jobs.json');scope=read(root/'scope.json')
    spec=importlib.util.spec_from_file_location('frozen_dispatcher',root/'transport/controller.py');controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller);controller.validate(plan)
    assert len(plan['jobs'])==len(jobs)==126 and scope['jobs']==126
    assert plan['hosts']['local']['gpus']==[0]and plan['hosts']['silver']['gpus']==[4,7]
    assert all(not j['dependencies']and '--pinned-worker'in j['command']and '--lease-receipt'in j['command']for j in plan['jobs'])
    ready=read(campaign/'data/ready.json');train={r['camera']:r for r in ready['records']if r['camera']not in {'DC','HC','KA'}}
    assert len(train)==63 and {(j['camera'],j['variant'])for j in jobs}=={(c,v)for c in train for v in ['lr_decay','threshold093']}
    for camera in train:
        pair=[j for j in jobs if j['camera']==camera];assert len({(j['host'],j['gpu'])for j in pair})==1
    for host,count in [('local',42),('silver',84)]:
        proof=read(root/'evidence'/('paths_'+host+'.json'));assert proof['passed']and proof['jobs']==count and not proof['Torch_imported']and not proof['GPU_jobs_launched']
    contexts={h:read(root/'evidence'/('runtime_'+h+'.json'))for h in ['local','silver']}
    assert all(v['torch']=='2.7.1+cu128'and v['CUDA_build']=='12.8'and not v['CUDA_initialized']for v in contexts.values())
    def package_hashes(context,module):return {Path(p).name:h for p,h in context['packages'][module]['files_sha256'].items()}
    assert package_hashes(contexts['local'],'pytorch_msssim')==package_hashes(contexts['silver'],'pytorch_msssim')
    roots={'local':root,'silver':remote_root};campaigns={'local':campaign,'silver':remote_root.parent};bindings={h:{}for h in roots}
    package_files=[p for p in root.rglob('*')if p.is_file()and '__pycache__'not in p.parts]
    for host,package in roots.items():
        for p in package_files:bindings[host][str(package/p.relative_to(root))]=sha(p)
    code_parent=campaign/'semantic_actor_candidate_v2_20261106';code_manifest=read(code_parent/'source_manifest.json')
    protected={str(Path('data')/n):h for n,h in ready['inventory']['files'].items()}
    protected.update({'data/ready.json':sha(campaign/'data/ready.json')})
    for p in (campaign/'data/lookcloser_frequencies').glob('*'):
        if p.is_file():protected[str(p.relative_to(campaign))]=sha(p)
    prepared=campaign/'frequency_variants_HD63_prepared_v1'
    for p in prepared.rglob('*'):
        if p.is_file()and '__pycache__'not in p.parts:protected[str(p.relative_to(campaign))]=sha(p)
    for rel,h in code_manifest['files_sha256'].items():protected[str(Path('semantic_actor_candidate_v2_20261106/code')/rel)]=h
    protected['semantic_actor_candidate_v2_20261106/source_manifest.json']=sha(code_parent/'source_manifest.json')
    for relative,h in protected.items():
        if sha(campaign/relative)!=h:raise ValueError('Protected source mismatch '+relative)
        for host,c in campaigns.items():bindings[host][str(c/relative)]=h
    write(root/'bindings.json',dict(hosts=bindings,science_original_unchanged=True,protected_data_and_existing_frequency_maps=True,source_code_manifest_sha256=sha(code_parent/'source_manifest.json')))
    GPU_UUIDs={host:{str(gpu):next(d['uuid']for d in contexts[host]['GPUs']if d['index']==gpu)for gpu in cfg['gpus']}for host,cfg in plan['hosts'].items()}
    activation=dict(kind='HD63_TWO_FREQUENCY_FACTORS_MAP_FITS_ONLY',jobs=126,cameras=63,controller_plan_sha256=sha(root/'controller_plan.json'),bindings_sha256=sha(root/'bindings.json'),controller_sha256=sha(root/'transport/controller.py'),GPU_UUIDs=GPU_UUIDs,NeRF_training_authorized=False,assembly_authorized=False,READY_writes=False,root_release_required=True,host_file_bindings={h:len(v)for h,v in bindings.items()},worker_sha256=scope['science_sha256']['worker.py'],preprocess_sha256=scope['science_sha256']['preprocess.py'],factory_receipt_sha256=scope['factory_receipt_sha256'],same_camera_paired_host_gpu=True,failed_outputs_and_completed_prefix_immutable=True)
    write(root/'activation.json',activation)
    write(root/'root_release.template.json',dict(reviewer='root',approved_map_fits=False,activation_sha256=sha(root/'activation.json'),scope='Only126 HD1000x16 frequency fits; no NeRF/data assembly/READY or map promotion.'))
    print(json.dumps(dict(activation_sha256=sha(root/'activation.json'),bindings=activation['host_file_bindings'],GPU_UUIDs=GPU_UUIDs,launched=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','campaign','remote-root']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();main(a.root,a.campaign,a.remote_root)
