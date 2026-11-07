"""Prepare a cold quarter-step/.1 trial on existing frozen4K data/maps, CPU only."""
import argparse,copy,hashlib,json,os,shutil,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')

def main(a):
    if a.output.exists():raise ValueError('Preserve prior candidate')
    C=a.recipe.parent;remote='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
    localize=lambda d:json.loads(json.dumps(d).replace(remote,str(C)))
    manifest=json.loads((a.recipe/'source_manifest.json').read_text())
    assert len(manifest['files_sha256'])==1495
    for n,h in manifest['files_sha256'].items():assert sha(a.recipe/'code'/n)==h
    a.output.mkdir();shutil.copytree(a.recipe/'code',a.output/'code',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(a.recipe/'source_manifest.json',a.output/'source_manifest.json')
    (a.output/'legacy').mkdir()
    for name in ['marshawn_semantic_4k_derivation.py','marshawn_semantic_4k_production_support.py']:
        shutil.copy2(a.source4k/'code/LookCloser/scripts'/name,a.output/'legacy'/name)
    factory=Path(__file__).with_name('train_marshawn_quarter_4k.py');isolation=Path(__file__).with_name('marshawn_gpu_isolation.py')
    for p in [factory,isolation]:shutil.copy2(p,a.output/p.name)
    sys.path[:0]=[str(a.output/'code'),str(a.output/'code/LookCloser/scripts'),str(a.output/'legacy')]
    from marshawn_semantic_4k_derivation import inspect
    oldcontract=json.loads((a.source4k/'contract.json').read_text());contract=localize(oldcontract)
    for n,h in contract['artifacts'].items():assert sha(n)==h,n
    actual=inspect(C/'data',a.source4k/'data',a.source4k/'source_inventory.json',contract['semantic_manifest'])
    expected=localize(json.loads((a.source4k/'derivation_receipt.json').read_text()))
    for key in ['source_inventory_sha256','geometry_source_hashes','semantic_manifest_sha256','support','splits','eval9bytes_and_crops_identical','native_RAW_widths_preserved']:
        assert actual[key]==expected[key],key
    write(a.output/'data_derivation.json',actual)
    contract['derivation_receipt']=str(a.output/'data_derivation.json')
    write(a.output/'data_validation_contract.json',contract)
    parent=localize(json.loads((a.recipe/'spec.json').read_text()))
    assert parent['coarse_step']==.0015625 and parent['distortion_final']==.1
    write(a.output/'localized_parent_spec.json',parent)
    spec=copy.deepcopy(parent);spec['data']=str(a.source4k/'data');spec['output']=str(a.output/'run')
    spec['semantic_admission']['release_receipt']=str(a.output/'root_release.json')
    write(a.output/'spec.json',spec)
    bounds={str(p):sha(p)for p in [a.output/'data_derivation.json',a.output/'data_validation_contract.json',a.output/'localized_parent_spec.json',a.output/'source_manifest.json',a.output/isolation.name]}
    bounds.update(contract['artifacts'])
    for n,h in json.loads((a.source4k/'source_inventory.json').read_text()).items():bounds[str(a.source4k/'data'/n)]=h
    for n,h in json.loads((a.source4k/'map_inventory.json').read_text()).items():bounds[str(a.source4k/n)]=h
    for n,h in json.loads(Path(spec['semantic_admission']['manifest']).read_text())['files_sha256'].items():bounds[str(Path(spec['semantic_admission']['manifest']).parent/n)]=h
    policy=dict(scope='COLD_QUARTER_STEP_DISTORTION01_FROZEN4K_BUNDLE',spec=str(a.output/'spec.json'),spec_sha256=sha(a.output/'spec.json'),
        factory_sha256=sha(factory),science_root=str(a.output/'code'),science_manifest=str(a.output/'source_manifest.json'),
        legacy_helpers=str(a.output/'legacy'),geometry_data=str(C/'data'),data_validation_contract=str(a.output/'data_validation_contract.json'),
        localized_parent_spec=str(a.output/'localized_parent_spec.json'),mapped_bool_sha256=actual['support']['mapped_bool_sha256'],
        inherited_cache_proof=dict(path=str(a.source4k/'cpu_complete.json'),sha256=sha(a.source4k/'cpu_complete.json'),peak_RSS_GiB=17.51175308227539),
        inherited_4K_technical_proof=dict(path=str(a.source4k/'technical_proof/complete.json'),sha256=sha(a.source4k/'technical_proof/complete.json')),
        comparisons=dict(HD_quarter01='Same structural recipe;4KRGB/native-confidence/4000map bundle versusHD1000 maps, not pure resolution.',
                         old4K='Same frozen4K inputs, but coarse.00625→.0015625 AND distortion.01→.1; not one-factor attribution.'),
        no_new_frequency_fits=True,training_authorized=False,bound_files=bounds)
    for p in (a.output/'legacy').iterdir():policy['bound_files'][str(p)]=sha(p)
    write(a.output/'policy.json',policy)
    release=dict(approved_for_training=False,spec_sha256=sha(a.output/'spec.json'),semantic_manifest_sha256=spec['semantic_admission']['manifest_sha256'],
        native_gpu_paths_passed=True,all_ones_identity_passed=True,adam_resume_passed=True,source_positive_support_passed=True,
        proof_attribution='Inherited unchanged quarter.1 mechanism and frozen4K actual-cache/32resume/1024frequency-update proofs; new CPU validation required; no newGPUproof claimed.',
        external_factory_policy_sha256=sha(a.output/'policy.json'),artifacts_sha256={'policy.json':sha(a.output/'policy.json'),factory.name:sha(factory),isolation.name:sha(isolation)})
    write(a.output/'root_release.template.json',release)
    controller=C/'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'
    plan=dict(schema_version=1,poll_seconds=15,hosts=dict(local=dict(python=a.python,controller=str(controller),root=str(a.output/'host'),gpus=[0])),
        jobs=[dict(name='quarter01_frozen4K',host='local',gpu=0,dependencies=[],command=[a.python,str(a.output/factory.name),'--policy',str(a.output/'policy.json'),'--lease-receipt','{job_root}/receipt.json'],cwd=str(a.output/'code/LookCloser'),
            env=dict(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),progress_files=[str(a.output/'run/progress.json'),str(a.output/'run/latest.json')],metrics_file=str(a.output/'run/complete.json'))])
    write(a.output/'controller_plan.json',plan)
    review=dict(files_sha256=dict(bounds,**{str(a.output/'code'/n):h for n,h in manifest['files_sha256'].items()}),source_files_unchanged=1495,launch_authorized=False)
    for p in [a.output/'policy.json',a.output/'controller_plan.json',a.output/'spec.json',a.output/factory.name]:review['files_sha256'][str(p)]=sha(p)
    write(a.output/'review_manifest.json',review)
    print(json.dumps(dict(root=str(a.output),policy_sha256=sha(a.output/'policy.json'),bindings=len(review['files_sha256']))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['recipe','source4k','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--python',default='/home/pavlo/.conda/envs/nerfstudio/bin/python');main(p.parse_args())
