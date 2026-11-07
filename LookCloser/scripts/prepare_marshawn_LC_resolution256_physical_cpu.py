"""CPU-only physical plan for the completed matched FOUR-UNKNOWN256 head."""
import argparse,json,shutil
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha,write


def prepare(c,output):
    if output.exists():raise FileExistsError('Use an independent immutable namespace')
    experiment=c/'frozen_LC_semantic_resolution256_fourUNK_plan_v1'
    checkpoint=experiment/'runs/prior001_fresh2064/head.pt'
    completion=checkpoint.with_name('complete.json')
    audit=experiment/'CPU_four_UNKNOWN_resolution_comparison.json'
    report=json.loads(audit.read_text())
    if (sha(checkpoint)!='befa50aa2b178938fc2e918b8d78d3e59c7c39699d63ccc45663fb5851ce9d21' or
        report['candidate256_head_sha256']!=sha(checkpoint) or not report['exact_camera_history_control_match'] or
        report['qualified_UNKNOWN_count']!=4 or report['actual_fresh_Adam_steps']!=2064):
        raise ValueError('Completed matched four-UNKNOWN256 head and actual CPU audit required')
    base=c/'frozen_LC_semantic_LA_two_UNKNOWN_physical_plan_v1'
    release=json.loads((base/'release_TEMPLATE.json').read_text());bound=dict(release['bound_files'])
    for path,digest in bound.items():
        if sha(path)!=digest:raise ValueError('Frozen source changed: '+path)
    output.mkdir(parents=True);(output/'code').mkdir();(output/'specs').mkdir()
    for source in (base/'code').glob('*.py'):
        target=output/'code'/source.name;shutil.copyfile(source,target);bound[str(target)]=sha(target)
    # These two guarded constructors are the only GPU-runtime changes.
    helper=output/'code/marshawn_LC_semantic_resolution_contract.py'
    shutil.copyfile(experiment/'code'/helper.name,helper);bound[str(helper)]=sha(helper)
    renderer=output/'code/render_marshawn_LC_semantic_physical.py'
    before=renderer.read_text();needle='        head=FrozenActorGrid(m[\'bounds\'],np.eye(4),resolution=128,initial_probability=initial_probability).cuda()'
    assert before.count(needle)==1
    after=before.replace('from prepare_marshawn_frozen_semantic_cpu import sha, write','from prepare_marshawn_frozen_semantic_cpu import sha, write\nfrom marshawn_LC_semantic_resolution_contract import physical_node_resolution')
    after=after.replace(needle,"        resolution=physical_node_resolution(trained,completion,dict(spec,expected_head_bounds=m['bounds']))\n        head=FrozenActorGrid(m['bounds'],np.eye(4),resolution=resolution,initial_probability=initial_probability).cuda()")
    renderer.write_text(after);bound[str(renderer)]=sha(renderer)
    train=output/'code/audit_marshawn_LC_semantic_train_distributions.py'
    before=train.read_text();needle="    head = FrozenActorGrid(m['bounds'], m['native_world_to_box'], 128, initial_probability=initial_probability).cuda()"
    assert before.count(needle)==1
    after=before.replace('from prepare_marshawn_frozen_semantic_cpu import sha, write','from prepare_marshawn_frozen_semantic_cpu import sha, write\nfrom marshawn_LC_semantic_resolution_contract import physical_node_resolution')
    after=after.replace(needle,"    resolution = physical_node_resolution(trained, completion, dict(s, expected_head_bounds=m['bounds']))\n    head = FrozenActorGrid(m['bounds'], m['native_world_to_box'], resolution, initial_probability=initial_probability).cuda()")
    train.write_text(after);bound[str(train)]=sha(train)
    labels=c/'frozen_LC_semantic_head_archives/LA_four_UNKNOWN_20261107_v1/label_qualification.json'
    specification=experiment/'specs/prior001_fresh2064.json'
    for p in [checkpoint,completion,audit,labels,specification,experiment/'root_release.json',experiment/'root_technical_acceptance.json']:
        if p.exists():bound[str(p)]=sha(p)
    original=json.loads(completion.read_text())
    adapter=dict(original,kind='DERIVED_LOCAL_FOUR_UNKNOWN256_CHECKPOINT_LOCATOR',checkpoint=str(checkpoint),
        original_remote_checkpoint=original['checkpoint'],original_completion_sha256=sha(completion),
        actual_label_qualification_sha256=sha(labels),scientific_checkpoint_bytes_unchanged=True)
    locator=output/'local_completion_adapter.json';write(locator,adapter);bound[str(locator)]=sha(locator)
    hashes={};specs={}
    for phase in ['render','metrics','TRAIN']:
        item=json.loads((base/'specs'/('four_UNKNOWN_'+phase+'.json')).read_text())
        item.update(root_release=str(output/'root_release.json'),output=str(output/'prior001'/dict(render='renders',metrics='visual_gate',TRAIN='TRAIN_distributions')[phase]))
        if phase in ['render','TRAIN']:
            item.update(head_completion=str(locator),label_qualification=str(labels),head_training_spec_sha256=sha(specification),expected_grid_resolution=256)
        else:item['renders']=str(output/'prior001/renders')
        path=output/'specs'/('resolution256_'+phase+'.json');write(path,item);hashes[str(path)]=sha(path);specs[phase]=path
    original_plan=json.loads((base/'controller_plan.json').read_text());host=dict(original_plan['hosts']['olive'],controller=str(output/'code/marshawn_tuning_controller.py'),root=str(output/'queue'))
    jobs=[];dependencies=[]
    for i,(phase,worker,folder,done) in enumerate([('render','render_marshawn_LC_semantic_physical.py','renders','complete.json'),('metrics','evaluate_marshawn_LC_semantic_physical.py','visual_gate','metrics.json'),('TRAIN','audit_marshawn_LC_semantic_train_distributions.py','TRAIN_distributions','distributions.json')]):
        name='resolution256_'+phase;job=dict(original_plan['jobs'][i]);job.update(name=name,command=[host['python'],str(output/'code'/worker),'--spec',str(specs[phase]),'--execute'],cwd=str(output/'code'),dependencies=dependencies,progress_files=[str(output/'prior001'/folder/'progress.json')],metrics_file=str(output/'prior001'/folder/done));jobs.append(job);dependencies=[name]
    plan=output/'controller_plan.json';write(plan,dict(schema_version=1,poll_seconds=15,hosts=dict(olive=host),jobs=jobs))
    write(output/'release_TEMPLATE.json',dict(reviewer='root',LC_physical_render_authorized=False,LC_physical_metrics_authorized=False,LC_train_distribution_audit_authorized=False,LC_uncapped_qualification_authorized=False,optimizer_authorized=False,production_after_visualgate=False,spec_sha256=hashes,controller_plan_sha256=sha(plan),bound_files=bound))
    write(output/'CPU_readiness.json',dict(CPU_only=True,CUDA_started=False,optimizer_created=False,new_head_sha256=sha(checkpoint),new_label_qualification_sha256=sha(labels),new_training_spec_sha256=sha(specification),control128_head_sha256=report['control128_head_sha256'],scientific_factor='FOUR_UNKNOWN128->FOUR_UNKNOWN256 independent XYZ logit nodes only',source_capture_Fb8_and_pipeline_c83_unchanged=True,GPU_runtime_changed_files=[renderer.name,train.name,helper.name],render_unchanged_except_guarded_constructor=True,TRAIN_unchanged_except_guarded_constructor=True,source_masks_RGB_sigma_sampling_reference_tolerances_selectors_unchanged=True,source_legacy_cap1024_only=True,exact4_same_batch_trace_pixels_preserved=True,actual_109point_ge05_count=106,actual_109point_ge01_count=109,blacksole6862_probability=.26195964217185974,physical_shoe_preservation_unqualified=True,all_permissions_false=True,automatic_promotion=False,comparison='Reuse completed matched4UNK128 physical outputs in a separate CPU comparison after this threejobgate; no dependency on still-running output bytes.'))
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.campaign,a.output)
