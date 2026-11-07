"""Cold quarter-step/.1 with frozen4K sources; unchanged scientific worker."""
import argparse,copy,hashlib,json,os,sys
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main(a):
    p=json.loads(a.policy.read_text());spec_path=Path(p['spec']);spec=json.loads(spec_path.read_text())
    assert sha(spec_path)==p['spec_sha256']and sha(__file__)==p['factory_sha256']
    for name,digest in p['bound_files'].items():
        if sha(name)!=digest:raise ValueError('Bound artifact changed: '+name)
    science=Path(p['science_root']);inventory=json.loads(Path(p['science_manifest']).read_text())['files_sha256']
    assert len(inventory)==1495
    for name,digest in inventory.items():
        if sha(science/name)!=digest:raise ValueError('Scientific source changed: '+name)
    if a.dry_run:os.environ['CUDA_VISIBLE_DEVICES']=''
    else:
        from marshawn_gpu_isolation import require_controller_lease
        if a.lease_receipt is None:raise ValueError('Actual parent global GPU lease required')
        lease=require_controller_lease(a.lease_receipt)
        available=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
        if available<20*2**30:raise ValueError('At least20GiB availableRAM required before4K cache')
    sys.path[:0]=[str(science),str(science/'LookCloser/scripts'),str(Path(p['legacy_helpers']))]
    import train_marshawn_semantic_admission as base
    import marshawn_semantic_4k_production_support as support
    assert Path(base.__file__).resolve()==(science/'LookCloser/scripts/train_marshawn_semantic_admission.py').resolve()
    assert Path(support.__file__).resolve()==(Path(p['legacy_helpers'])/'marshawn_semantic_4k_production_support.py').resolve()
    original_build=base.build_config;original_validate=base.validate_spec
    original_validate(spec)
    support.CONTRACT=support.validate_contract(json.loads(Path(p['data_validation_contract']).read_text()))
    ready=support.validate_data(Path(spec['data']))
    def build(value,resume=None):
        if value!=spec:raise ValueError('Only exact sealed quarter4K spec accepted')
        cfg=original_build(value,resume)
        cfg.pipeline.model.semantic_data=p['geometry_data']
        return cfg
    base.build_config=build
    base.validate_data=support.validate_data
    if a.dry_run:
        import numpy as np,torch,yaml
        from marshawn_semantic_admission import load_support
        from marshawn_semantic_4k_derivation import parser_check
        cfg=build(spec);parent=json.loads(Path(p['localized_parent_spec']).read_text());original_validate(parent)
        reference=original_build(parent)
        actual_parser=parser_check(science/'LookCloser/scripts/train_marshawn_semantic_admission.py',p['geometry_data'],spec['data'])
        gate=spec['semantic_admission'];bounds=np.array(ready['aabb'],dtype=np.float32)
        mask,receipt=load_support(gate['manifest'],gate['manifest_sha256'],cfg.pipeline.model.semantic_data,bounds,(128,128,128))
        assert receipt['mapped_bool_sha256']==p['mapped_bool_sha256']
        def normal(c,s):
            c=copy.deepcopy(c)
            assert c.output_dir==Path(s['output'])and c.pipeline.datamanager.data==Path(s['data'])and c.pipeline.datamanager.dataparser.data==Path(s['data'])
            assert c.pipeline.model.semantic_data==p['geometry_data']
            c.output_dir=Path('<OUTPUT>');c.pipeline.datamanager.data=Path('<DATA>');c.pipeline.datamanager.dataparser.data=Path('<DATA>')
            return yaml.dump(c)
        left,right=normal(reference,parent),normal(cfg,spec)
        assert left==right,'Unexpected model/optimizer/schedule difference'
        assert cfg.load_checkpoint is None and not torch.cuda.is_initialized()
        result=dict(passed=True,CPU_only=True,training_launched=False,science_files_unchanged=1495,
            policy_sha256=sha(a.policy),factory_sha256=sha(__file__),spec_sha256=sha(spec_path),
            actual4K_data_validator_passed=True,actual_CPU_parser=actual_parser,actual_admission_load=receipt,
            admission_cells=int(mask.size),normalized_config_equal=True,
            normalized_config_sha256=hashlib.sha256(right.encode()).hexdigest(),
            only_config_differences=['datamanager.data','dataparser.data','output_dir'],
            geometry_data=p['geometry_data'],TRAIN_shape=[63,2024,3840],EVAL_shape=[3,1012,1920],
            cold_seed=42,checkpoint_ancestor=None,mandatory_end=106316,iteration_limit=None,
            inherited_4K_cache_proof=p['inherited_cache_proof'],
            pure_resolution_causality=False,bundle='Actual4K RGB/native confidence/frozen4000step maps; no refits')
        if a.output is None or a.output.exists():raise ValueError('New proof output required')
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    from run_marshawn_semantic_release import verify
    release=verify(spec_path)
    if release.get('external_factory_policy_sha256')!=sha(a.policy):raise ValueError('Exact quarter4K root release required')
    import torch
    from marshawn_gpu_isolation import verify_cuda
    gpu=verify_cuda(lease['expected_physical_uuid'],torch)
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
    name='external_factory_resume.json'if(out/'latest.json').exists()else'external_factory_cold.json'
    (out/name).write_text(json.dumps(dict(policy_sha256=sha(a.policy),factory_sha256=sha(__file__),GPU=gpu,
        lease=lease,source_grid_unchanged=True,geometry_data=p['geometry_data'],RGB_data=spec['data'],allocator_cap_GiB=10),indent=2)+'\n')
    base.train(spec_path)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--lease-receipt',type=Path)
    main(p.parse_args())
