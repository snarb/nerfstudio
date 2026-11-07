"""CPU-only executable path aliases; canonical local/source bytes stay intact."""
import copy
import json
from pathlib import Path

from prepare_marshawn_lc_semantic_capture import CAMPAIGN,sha,write
from stage_marshawn_lc_head_silver import SC,ssh,UPLOAD,INVENTORY

NAME='frozen_LC_semantic_head_silver_alias_v1'


def relocate(value):
    if isinstance(value,str) and value.startswith(str(CAMPAIGN)+'/'):
        return str(SC/Path(value).relative_to(CAMPAIGN))
    if isinstance(value,list):return [relocate(x)for x in value]
    if isinstance(value,dict):return {relocate(k):relocate(v)for k,v in value.items()}
    return value


def differences(a,b,path=''):
    if isinstance(a,dict)and isinstance(b,dict):
        assert a.keys()==b.keys()
        return sum((differences(a[k],b[k],path+'.'+k)for k in a),[])
    if isinstance(a,list)and isinstance(b,list):
        assert len(a)==len(b)
        return sum((differences(x,y,path+f'[{i}]')for i,(x,y)in enumerate(zip(a,b))),[])
    return []if a==b else[dict(key=path,before=a,after=b)]


def main():
    root=CAMPAIGN/NAME;root.mkdir(exist_ok=False);(root/'specs').mkdir()
    remote=SC/NAME;head=CAMPAIGN/'frozen_LC_semantic_head_plan_v1'
    source_template=json.loads((head/'release_TEMPLATE.json').read_text())
    contract=json.loads((head/'head_input_contract.json').read_text());alias=relocate(contract)
    write(root/'head_input_contract.json',alias)
    capture=json.loads((head/'capture_adapter_receipt.json').read_text());alias_capture=relocate(capture)
    alias_capture['manifest_sha256']=sha(root/'head_input_contract.json');write(root/'capture_adapter_receipt.json',alias_capture)
    labels=json.loads((head/'label_qualification.json').read_text());alias_labels=relocate(labels)
    alias_labels['capture_receipt_sha256']=sha(root/'capture_adapter_receipt.json');write(root/'label_qualification.json',alias_labels)
    assert alias['LC_capture_receipt_sha256']=='fb8b875d8c7f3676c3dfbd187b4c76c3ee40ec8a3b62a940fdf33bb97d99073d'
    assert alias_capture['model_digest']==contract['source_pipeline_digest']
    assert [r.get('archive_sha256')for r in alias_capture['records']]==[r.get('archive_sha256')for r in capture['records']]
    specs={};source_specs={};diffs={}
    for prior in ['099','001']:
        name='prior'+prior+'_fresh2064.json';base=json.loads((head/'specs'/name).read_text());derived=relocate(base)
        derived.update(root_release=str(remote/'root_release.json'),manifest=str(remote/'head_input_contract.json'),
            capture_receipt=str(remote/'capture_adapter_receipt.json'),label_qualification=str(remote/'label_qualification.json'),
            output=str(remote/'runs'/('prior'+prior+'_fresh2064')))
        for key in set(base)-{'root_release','manifest','capture_receipt','label_qualification','output','camera_history_reference'}:
            assert base[key]==derived[key],key
        write(root/'specs'/name,derived);specs[str(remote/'specs'/name)]=sha(root/'specs'/name)
        source_specs[str(head/'specs'/name)]=sha(head/'specs'/name);diffs[name]=differences(base,derived)
    # Every changed non-hash value is exactly an executable path relocation.
    for name,base,derived,hash_keys in [('head_input_contract.json',contract,alias,set()),
            ('capture_adapter_receipt.json',capture,alias_capture,{'.manifest_sha256'}),
            ('label_qualification.json',labels,alias_labels,{'.capture_receipt_sha256'})]:
        diff=differences(base,derived)
        for d in diff:
            if d['key']in hash_keys:continue
            assert isinstance(d['before'],str)and d['after']==relocate(d['before']),d
        diffs[name]=diff
    provenance=dict(kind='EXPLICIT_CPU_EXECUTION_PATH_ALIAS_EQUIVALENCE',source_local_head=str(head),source_head_remote_bytes=str(SC/head.relative_to(CAMPAIGN)),
        canonical_capture_receipt_sha256=contract['LC_capture_receipt_sha256'],source_pipeline_digest=contract['source_pipeline_digest'],
        source_contract_sha256=sha(head/'head_input_contract.json'),alias_contract_sha256=sha(root/'head_input_contract.json'),
        source_adapter_sha256=sha(head/'capture_adapter_receipt.json'),alias_adapter_sha256=sha(root/'capture_adapter_receipt.json'),
        source_label_qualification_sha256=sha(head/'label_qualification.json'),alias_label_qualification_sha256=sha(root/'label_qualification.json'),
        source_specs_sha256=source_specs,alias_specs_sha256=specs,scientific_numeric_objective_seed_prior_LR_grid_camera_packets_unchanged=True,
        original63_membership_69888_rows_136_capacity_identities_two_UNKNOWN_retained=True,
        all_cache_arrays_and_source_metadata_byte_identical=True,canonical_JSONs_NOT_rewritten=True,
        required_digest_reference_repairs=['aliasadapter.manifest_sha256 -> aliascontract SHA','aliaslabels.capture_receipt_sha256 -> aliasadapter SHA'],
        original_CP_hash='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641',
        source_CP_alias=str(SC/'frozen_LC_semantic_capture_plan_v2/local_source/checkpoint.ckpt'),
        source_CP_original=str(SC/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin/checkpoint.ckpt'),
        remote_python=str(SC/'python'),remote_torch_version='2.7.1+cu128',remote_CUDA_build='12.8',
        path_differences=diffs,GPU_started=False,training_authorized=False)
    write(root/'canonical_equivalence.json',provenance)
    release=copy.deepcopy(source_template)
    for key in release:
        if isinstance(release[key],bool):release[key]=False
    release['bound_files']={relocate(k):v for k,v in source_template['bound_files'].items()}
    for p in [root/'head_input_contract.json',root/'capture_adapter_receipt.json',root/'label_qualification.json',root/'canonical_equivalence.json',*list((root/'specs').glob('*.json'))]:
        release['bound_files'][str(remote/p.relative_to(root))]=sha(p)
    release['label_qualification_sha256']=sha(root/'label_qualification.json');release['spec_sha256']=specs
    controller=SC/'frozen_LC_semantic_capture_plan_v4/code/marshawn_tuning_controller.py'
    release['bound_files'][str(controller)]=sha(CAMPAIGN/'frozen_LC_semantic_capture_plan_v4/code/marshawn_tuning_controller.py')
    for query in alias_labels['unknown_queries']:
        release['bound_files'][query['qualification']]=query['qualification_sha256']
    worker=SC/'frozen_LC_semantic_head_plan_v1/code/probe_marshawn_LC_frozen_semantic_head.py'
    jobs=[]
    for prior in ['099','001']:
        job='LC_prior'+prior+'_fresh2064'
        jobs.append(dict(name=job,host='silver',command=[str(SC/'python'),str(worker),'--spec',str(remote/'specs'/('prior'+prior+'_fresh2064.json')),'--execute'],
            cwd=str(worker.parent),env=dict(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json'),
            dependencies=[],poll_seconds=5,progress_files=[str(remote/'runs'/('prior'+prior+'_fresh2064')/'progress.json')],metrics_file=str(remote/'runs'/('prior'+prior+'_fresh2064')/'complete.json')))
    plan=dict(schema_version=1,poll_seconds=15,hosts=dict(silver=dict(address='ubuntu@silver-sable',python=str(SC/'python'),controller=str(controller),root=str(remote/'queue'),gpus=[0,1])),jobs=jobs)
    write(root/'controller_plan.json',plan)
    release['bound_files'][str(remote/'controller_plan.json')]=sha(root/'controller_plan.json')
    write(root/'root_review_TEMPLATE.json',release)
    files=[p for p in root.rglob('*')if p.is_file()]
    inventory=[]
    for p in files:
        target=remote/p.relative_to(root);h=sha(p)
        with p.open('rb')as f:ssh(UPLOAD,[target,h],stdin=f)
        inventory.append(dict(local=str(p),remote=str(target),sha256=h,bytes=p.stat().st_size))
    write(root/'alias_inventory.json',inventory)
    with(root/'alias_inventory.json').open('rb')as f:status=json.loads(ssh(INVENTORY,stdin=f).stdout)
    assert all(x['state']=='verified'for x in status)
    receipt=dict(passed=True,local_review_mirror=str(root),remote_execution_root=str(remote),remote_python=str(SC/'python'),
        source_CP_alias=provenance['source_CP_alias'],canonical_capture_receipt_sha256=provenance['canonical_capture_receipt_sha256'],
        canonical_equivalence_sha256=sha(root/'canonical_equivalence.json'),root_review_template_sha256=sha(root/'root_review_TEMPLATE.json'),
        controller_plan_sha256=sha(root/'controller_plan.json'),alias_files_verified=len(status),source_byte_stage_receipt=str(CAMPAIGN/'frozen_LC_semantic_head_silver_stage_v1/stage_receipt.json'),
        all_permissions_false=True,GPU_started=False,controller_started=False,local_contracts_modified=False,missing_inputs=[])
    write(root/'alias_stage_receipt.json',receipt)
    with(root/'alias_stage_receipt.json').open('rb')as f:ssh(UPLOAD,[remote/'alias_stage_receipt.json',sha(root/'alias_stage_receipt.json')],stdin=f)
    print(json.dumps(receipt))


if __name__=='__main__':main()
