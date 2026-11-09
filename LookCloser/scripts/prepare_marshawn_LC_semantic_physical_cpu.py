"""Seal CPU-only LC physical/qualification proposals after actual capture binding."""
import argparse
import ast
import difflib
import json
from pathlib import Path
import shutil
from prepare_marshawn_frozen_semantic_cpu import sha,write


def scientific_loader_statements(text):
    module=ast.parse(text)
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef)and n.name in ['run','load_source'])
    statements=function.body
    start=next(i for i,n in enumerate(statements)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='cp'for t in n.targets))
    end=next(i for i,n in enumerate(statements)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='rng_before'for t in n.targets))
    return ast.dump(ast.Module(body=statements[start:end+1],type_ignores=[]),include_attributes=False)


def prepare(fit,output):
    if output.exists():raise FileExistsError('Use new immutable namespace')
    contract_path=fit/'head_input_contract.json';contract=json.loads(contract_path.read_text())
    plan=Path(contract['LC_capture_plan']);m=json.loads(plan.read_text());source_root=plan.parent
    source_worker=source_root/'code/run_marshawn_lc_semantic_capture_v3.py'
    if sha(source_worker)!='5c601707f722d1f6f82c43fba0b50e1e7caf0f6db455c9fa1fc8633356b9bbef':raise ValueError('Unreviewed source loader')
    if sha(source_root/'code/marshawn_lc_semantic_capture.py')!='e5dbb1eba7045e4fea14112afe59992452c67c7bd03d5e84cf20c637e55db976':raise ValueError('Use final FP16/FP32-qualified observer')
    own_loader=Path(__file__).with_name('marshawn_LC_frozen_source.py')
    if scientific_loader_statements(source_worker.read_text())!=scientific_loader_statements(own_loader.read_text()):
        raise ValueError('Scientific source restoration statements must remain AST-identical')
    release=json.loads((fit/'release_TEMPLATE.json').read_text());bound=dict(release['bound_files'])
    for p,digest in bound.items():
        if sha(p)!=digest:raise ValueError('Fit proposal input changed: '+p)
    output.mkdir(parents=True);(output/'code').mkdir();(output/'specs').mkdir();(output/'controllers').mkdir()
    names=['marshawn_LC_frozen_source.py','marshawn_LC_semantic_physical_path.py','render_marshawn_LC_semantic_physical.py',
        'audit_marshawn_LC_semantic_train_distributions.py','evaluate_marshawn_LC_semantic_physical.py',
        'compare_marshawn_LC_semantic_priors_cpu.py','marshawn_frozen_semantic_head.py','marshawn_frozen_semantic_selectors.py',
        'prepare_marshawn_frozen_semantic_cpu.py','run_marshawn_frozen_semantic_probe.py','launch_marshawn_LC_semantic_stages.py']
    for name in names:
        target=output/'code'/name;shutil.copyfile(Path(__file__).with_name(name),target);bound[str(target)]=sha(target)
    for name in ['marshawn_gpu_isolation.py','marshawn_tuning_metrics.py']:
        target=output/'code'/name
        source_helper=Path(m['branch'])/'code/LookCloser/scripts'/name if name=='marshawn_tuning_metrics.py' else Path(m['helpers'])/name
        shutil.copyfile(source_helper,target);bound[str(target)]=sha(target)
    for name in ['marshawn_tuning_controller.py']:
        target=output/'code'/name;shutil.copyfile(source_root/'code'/name,target);bound[str(target)]=sha(target)
    for name in m['code_sha256']:
        target=source_root/'code'/name;bound[str(target)]=sha(target)
    source_release=source_root/'root_release.json'
    bound[str(source_release)]=sha(source_release)
    campaign=Path(m['campaign'])
    common=dict(root_release=str(output/'root_release.json'),LC_source_plan=str(plan),LC_source_release=str(source_release),
        LC_source_worker_name=source_worker.name,LC_capture_receipt_sha256=contract['LC_capture_receipt_sha256'],
        label_qualification=str(fit/'label_qualification.json'),chunk=128,max_seconds=3600,
        no_NGP_head_transfer=True,world_to_box=m['world_to_box'])
    specs={}
    q=dict(common,phase='uncapped_qualification',path_mode='uncapped0',output=str(output/'uncapped_qualification'))
    specs['uncapped_qualification']=q
    capacity136=campaign/'frozen_native_semantic_capacity_source_review_CPU_v1/capacity136.csv'
    bound[str(capacity136)]=sha(capacity136)
    source_metrics=Path(m['source_archive'])/'gate_metrics.json';bound[str(source_metrics)]=sha(source_metrics)
    ready=campaign/'data/ready.json'
    regions=campaign/'native_semantic_grid_20000/corrected_v2/review/region_crops.json'
    for p in [ready,regions]:bound[str(p)]=sha(p)
    for name,prior in [('prior099',.99),('prior001',.01)]:
        head_spec=fit/'specs'/(name+'_fresh2064.json')
        if not head_spec.exists():raise ValueError('Actual fresh2064 head proposal missing')
        bound[str(head_spec)]=sha(head_spec)
        root=output/name
        specs[name+'_TRAIN']=dict(root_release=str(output/'root_release.json'),manifest=str(contract_path),
            capture_receipt=str(fit/'capture_adapter_receipt.json'),label_qualification=str(fit/'label_qualification.json'),
            head_completion=str(fit/'runs'/(name+'_fresh2064')/'complete.json'),expected_head_iteration=2064,
            head_training_spec_sha256=sha(head_spec),expected_initial_probability=prior,
            original_native_capacity136_csv=str(capacity136),torch_version=m['torch_version'],cuda_version=m['cuda_version'],
            output=str(root/'TRAIN_distributions'))
        specs[name+'_render']=dict(common,phase='first2064_capped_native_path_physical_diagnostic',path_mode='native1024',
            head_completion=str(fit/'runs'/(name+'_fresh2064')/'complete.json'),expected_initial_probability=prior,
            head_training_spec_sha256=sha(head_spec),output=str(root/'renders'))
        specs[name+'_metrics']=dict(root_release=str(output/'root_release.json'),renders=str(root/'renders'),ready=str(ready),
            regions=str(regions),native_baseline_evaluation=str(source_metrics),output=str(root/'visual_gate'))
    specs['paired_CPU_review']=dict(root_release=str(output/'root_release.json'),prior99=str(output/'prior099'),prior01=str(output/'prior001'),
        prior99_TRAIN=str(output/'prior099/TRAIN_distributions/distributions.json'),prior01_TRAIN=str(output/'prior001/TRAIN_distributions/distributions.json'),
        ready=str(ready),regions=str(regions),output=str(output/'paired_review'))
    spec_hashes={}
    for name,value in specs.items():
        p=output/'specs'/(name+'.json');write(p,value);spec_hashes[str(p)]=sha(p)
    controller=output/'code/marshawn_tuning_controller.py';python=m['python']
    def job(name,worker,spec,dependencies,progress,metrics):
        return dict(name=name,host='olive',command=[python,str(worker),'--spec',str(spec),'--execute'],
            cwd=str(worker.parent),env=dict(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',EXPECTED_GPU_UUID=m['expected_physical_UUID'],LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json'),
            poll_seconds=5,dependencies=dependencies,progress_files=[str(progress)],metrics_file=str(metrics))
    def stage(name,jobs):
        p=output/'controllers'/(name+'.json')
        write(p,dict(schema_version=1,poll_seconds=15,hosts=dict(olive=dict(python=python,controller=str(controller),
            root=str(output/'queue'/name),gpus=[0])),jobs=jobs));return p
    technical=[]
    for name,deps in [('technical32',[]),('technical_resume32',['technical32']),('technical64',['technical_resume32'])]:
        technical.append(job(name,fit/'code/probe_marshawn_LC_frozen_semantic_head.py',fit/'specs'/(name+'.json'),deps,
            fit/'runs'/name/'progress.json',fit/'runs'/name/'complete.json'))
    comparison=dict(technical[-1]);comparison.update(name='technical_CPU_comparison',dependencies=['technical64'],
        command=[python,str(fit/'code/compare_marshawn_frozen_semantic_roundtrip_cpu.py'),
            '--first32',str(fit/'runs/technical32'),'--resumed64',str(fit/'runs/technical_resume32'),
            '--full64',str(fit/'runs/technical64'),'--output',str(output/'technical_CPU_comparison')],
        progress_files=[],metrics_file=str(output/'technical_CPU_comparison/comparison.json'))
    technical.append(comparison)
    stages={'technical':stage('technical',technical)}
    stages['uncapped_qualification']=stage('uncapped_qualification',[job('uncapped_source_qualification',
        output/'code/render_marshawn_LC_semantic_physical.py',output/'specs/uncapped_qualification.json',[],
        output/'uncapped_qualification/progress.json',output/'uncapped_qualification/complete.json')])
    production=[];previous=[]
    for name in ['prior099_fresh2064','prior001_fresh2064']:
        production.append(job(name,fit/'code/probe_marshawn_LC_frozen_semantic_head.py',fit/'specs'/(name+'.json'),previous,
            fit/'runs'/name/'progress.json',fit/'runs'/name/'complete.json'));previous=[name]
    for phase,worker,subdir,completion_file in [('TRAIN','audit_marshawn_LC_semantic_train_distributions.py','TRAIN_distributions','distributions.json'),
            ('render','render_marshawn_LC_semantic_physical.py','renders','complete.json'),
            ('metrics','evaluate_marshawn_LC_semantic_physical.py','visual_gate','metrics.json')]:
        for prior in ['prior099','prior001']:
            name=prior+'_'+phase
            production.append(job(name,output/'code'/worker,output/'specs'/(name+'.json'),previous,
                output/prior/subdir/'progress.json',output/prior/subdir/completion_file));previous=[name]
    production.append(job('paired_CPU_review',output/'code/compare_marshawn_LC_semantic_priors_cpu.py',
        output/'specs/paired_CPU_review.json',previous,output/'paired_review/progress.json',output/'paired_review/comparison.json'))
    stages['first2064_gate']=stage('first2064_gate',production)
    stage_hashes={k:sha(p)for k,p in stages.items()}
    write(output/'release_TEMPLATE.json',dict(reviewer='root',LC_uncapped_qualification_authorized=False,
        LC_train_distribution_audit_authorized=False,LC_physical_render_authorized=False,
        LC_physical_metrics_authorized=False,LC_paired_prior_comparison_authorized=False,
        LC_controller_technical_authorized=False,LC_controller_first2064_authorized=False,
        production_after_visualgate=False,spec_sha256=spec_hashes,bound_files=bound,controller_plan_sha256=stage_hashes))
    original=(source_root/'code/marshawn_lc_semantic_capture.py').read_text().splitlines(True)
    modified=(output/'code/marshawn_LC_semantic_physical_path.py').read_text().splitlines(True)
    (output/'physical_observer_vs_v4.diff').write_text(''.join(difflib.unified_diff(original,modified,fromfile='sealed_v4_capture',tofile='independent_physical_path')))
    write(output/'CPU_readiness.json',dict(CPU_only=True,CUDA_started=False,optimizer_started=False,
        actual_LC_capture_binding=contract['LC_capture_receipt_sha256'],source_loader_scientific_AST_identical=True,
        python=python,fit_release=str(fit/'root_release.json'),controller_plans=stage_hashes,
        source_worker_sha256=sha(source_worker),source_observer_sha256=sha(source_root/'code/marshawn_lc_semantic_capture.py'),
        new_physical_observer_diff='Only declared cap parameter/guard/flag; v4 product dtype math preserved.',
        native1024='Physical recomposition on source-capped intervals only; no opacity earlystop. Tail and phase limitations explicit.',
        uncapped0='Separate zero-update qualification. Allocation/phase changes measured; strict fullimage RGB/alpha native1024 comparison. No automatic progression.',
        contributor_traces={'HC':[[875,602]],'DC':[[1097,572]],'KA':[[1075,270],[961,485]]},
        original_NGP136_query_identities_retained=True,new_LC_capacity_failures_separate=True,
        automatic_promotion=False,release_template_all_permissions_false=True))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fit-plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.fit_plan,args.output)
