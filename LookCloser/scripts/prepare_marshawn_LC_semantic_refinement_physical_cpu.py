"""Stage olive0 typed physical evaluation; finalize only after genuine2064 heads exist."""
import argparse,json,shutil
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha,write
from marshawn_LC_semantic_refinement_contract import BASE_SHA,ARMS

def prepare(campaign,output,finalize):
    prior=campaign/'frozen_LC_semantic_LA_two_UNKNOWN_physical_plan_v1'
    fit=campaign/'frozen_LC_semantic_matched_refinement_plan_v1'
    if not finalize:
        output.mkdir(parents=True,exist_ok=False);(output/'specs').mkdir()
        shutil.copytree(prior/'code',output/'code',ignore=shutil.ignore_patterns('__pycache__'))
        for name in ['render_marshawn_LC_semantic_refinement_physical.py','marshawn_LC_semantic_refinement_physical.py',
            'marshawn_LC_semantic_refinement_contract.py','marshawn_semantic_residual_head.py',
            'audit_marshawn_LC_semantic_refinement_train_distributions.py','compare_marshawn_LC_semantic_refinement_physical_cpu.py']:
            shutil.copyfile(Path(__file__).with_name(name),output/'code'/name)
    elif (output/'root_release.json').exists():
        raise ValueError('Never mutate a physically released namespace')
    original_release=json.loads((prior/'release_TEMPLATE.json').read_text())
    bound={p:d for p,d in original_release['bound_files'].items()if not p.startswith(str(prior))}
    base=campaign/'frozen_LC_semantic_head_archives/LA_four_UNKNOWN_20261107_v1/runs/prior001_fresh2064/head.pt'
    assert sha(base)==BASE_SHA;bound[str(base)]=BASE_SHA
    bound.update({str(p):sha(p)for p in (output/'code').iterdir()if p.is_file()})
    specs={};jobs=[];pending=[];python='/home/pavlo/.conda/envs/nerfstudio/bin/python'
    env=dict(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',
        EXPECTED_GPU_UUID='GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46',LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json')
    previous=None
    for arm in ARMS:
        trained_spec=fit/'specs'/(arm+'_fresh2064.json');bound[str(trained_spec)]=sha(trained_spec)
        completion=fit/'runs'/(arm+'_fresh2064')/'complete.json';head=completion.with_name('head.pt')
        adapter=output/(arm+'_local_completion_adapter.json')
        if completion.exists()and head.exists():
            doc=json.loads(completion.read_text())
            assert doc['actual_head_updates']==doc['iteration']==2064 and sha(head)==doc['checkpoint_sha256']
            assert doc['head_type']==ARMS[arm]and doc['head_schema_version']==1
            doc.update(checkpoint=str(head),completion_adapter_kind='EXPLICIT_PATH_ONLY_ADAPTER_OF_GENUINE_TYPED_REFINEMENT_COMPLETION',
                original_completion=str(completion),original_completion_sha256=sha(completion))
            write(adapter,doc)
            for p in [completion,head,adapter]:bound[str(p)]=sha(p)
        else:pending.append(arm)
        typed=dict(expected_head_type=ARMS[arm],refinement_training_spec=str(trained_spec),refinement_base_checkpoint=str(base),
            head_training_spec_sha256=sha(trained_spec),head_completion=str(adapter))
        for phase,oldname,script,suffix,progress,result in [
            ('render','four_UNKNOWN_render','render_marshawn_LC_semantic_refinement_physical.py','renders','progress.json','complete.json'),
            ('metrics','four_UNKNOWN_metrics','evaluate_marshawn_LC_semantic_physical.py','visual_gate','progress.json','metrics.json'),
            ('TRAIN','four_UNKNOWN_TRAIN','audit_marshawn_LC_semantic_refinement_train_distributions.py','TRAIN_distributions','progress.json','distributions.json')]:
            spec=json.loads((prior/'specs'/(oldname+'.json')).read_text())
            spec.update(root_release=str(output/'root_release.json'),output=str(output/arm/suffix))
            if phase!='metrics':spec.update(typed)
            else:spec['renders']=str(output/arm/'renders')
            sp=output/'specs'/(arm+'_'+phase+'.json');write(sp,spec);specs[str(sp)]=sha(sp)
            name='refinement_'+arm+'_'+phase
            jobs.append(dict(name=name,host='olive',command=[python,str(output/'code'/script),'--spec',str(sp),'--execute'],
                cwd=str(output/'code'),env=env,poll_seconds=5,dependencies=[previous]if previous else[],
                progress_files=[str(output/arm/suffix/progress)],metrics_file=str(output/arm/suffix/result)))
            previous=name
    comparison=dict(root_release=str(output/'root_release.json'),base_control=str(prior/'prior001'),
        coarse_refinement=str(output/'coarse128'),residual_refinement=str(output/'residual256'),
        ready=str(campaign/'data/ready.json'),regions=str(campaign/'native_semantic_grid_20000/corrected_v2/review/region_crops.json'),
        base_checkpoint_sha256=BASE_SHA,output=str(output/'matched_review'))
    sp=output/'specs/comparison.json';write(sp,comparison);specs[str(sp)]=sha(sp)
    jobs.append(dict(name='refinement_matched_CPU_comparison',host='olive',command=[python,str(output/'code/compare_marshawn_LC_semantic_refinement_physical_cpu.py'),'--spec',str(sp),'--execute'],
        cwd=str(output/'code'),env=dict(env,CUDA_VISIBLE_DEVICES=''),poll_seconds=5,dependencies=[previous],progress_files=[],metrics_file=str(output/'matched_review/comparison.json')))
    plan=dict(schema_version=1,poll_seconds=15,hosts=dict(olive=dict(python=python,controller=str(output/'code/marshawn_tuning_controller.py'),root=str(output/'queue'),gpus=[0])),jobs=jobs)
    write(output/'controller_plan.json',plan)
    release=dict(reviewer=None,LC_physical_render_authorized=False,LC_physical_metrics_authorized=False,
        LC_train_distribution_audit_authorized=False,LC_refinement_comparison_authorized=False,
        LC_uncapped_qualification_authorized=False,optimizer_authorized=False,production_after_visualgate=False,
        spec_sha256=specs,controller_plan_sha256=sha(output/'controller_plan.json'),bound_files=bound,
        pending_genuine_head_artifacts=pending)
    write(output/'release_TEMPLATE.json',release)
    write(output/'CPU_readiness.json',dict(CPU_only=True,CUDA_started=False,authorized=False,physical_UUID=env['EXPECTED_GPU_UUID'],
        pending_genuine_head_artifacts=pending,release_ready=not pending,job_count=len(jobs),render_views=['DC','HC','KA'],detail_regions=9,
        reuse_original7937_physical_control=True,typed_constructor_required=True,source_recipe_and_selectors_unchanged=True,
        completion_adapters_explicit_path_only=True,semantic_labels_and_native_arrays_unchanged=True))
    print(json.dumps(dict(output=str(output),pending=pending,CUDA_started=False,authorization=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--finalize-existing',action='store_true');a=p.parse_args();prepare(a.campaign,a.output,a.finalize_existing)
