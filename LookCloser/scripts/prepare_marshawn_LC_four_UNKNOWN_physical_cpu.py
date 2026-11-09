"""Bind the completed four-UNKNOWN head to unchanged P3 native physical code."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_marshawn_frozen_semantic_cpu import sha,write


def prepare(c,output):
    if output.exists():raise FileExistsError('Use a new immutable physical namespace')
    p3=c/'frozen_LC_semantic_physical_plan_v3';archive=c/'frozen_LC_semantic_head_archives/LA_four_UNKNOWN_20261107_v1'
    checkpoint=archive/'runs/prior001_fresh2064/head.pt';completion=archive/'runs/prior001_fresh2064/complete.json'
    labels=archive/'label_qualification.json';specification=archive/'spec.json';audit=archive/'CPU_four_UNKNOWN_control_audit.json'
    if sha(checkpoint)!='7937ab3df744254ea4b4087f8efda9384888e3bccbf833719bc45945b8a81beb' or not json.loads(audit.read_text())['exact_camera_history_control_match']:
        raise ValueError('Actual completed and CPU-verified four-UNKNOWN head required')
    policy=json.loads((p3/'release_TEMPLATE.json').read_text());bound=dict(policy['bound_files'])
    for p,digest in bound.items():
        if sha(p)!=digest:raise ValueError('Frozen control input changed: '+p)
    output.mkdir(parents=True);(output/'code').mkdir();(output/'specs').mkdir()
    for p in (p3/'code').glob('*.py'):
        target=output/'code'/p.name;shutil.copyfile(p,target);bound[str(target)]=sha(target)
    extra=output/'code/compare_marshawn_LC_label_overlays_cpu.py';shutil.copyfile(Path(__file__).with_name(extra.name),extra);bound[str(extra)]=sha(extra)
    for p in [checkpoint,completion,labels,specification,audit,archive/'root_release.json']:
        bound[str(p)]=sha(p)
    original=json.loads(completion.read_text())
    adapter=dict(original,kind='DERIVED_LOCAL_FOUR_UNKNOWN_CHECKPOINT_LOCATOR',checkpoint=str(checkpoint),
        original_remote_checkpoint=original['checkpoint'],original_completion_sha256=sha(completion),
        actual_label_qualification_sha256=sha(labels),scientific_checkpoint_bytes_unchanged=True)
    locator=output/'local_completion_adapter.json';write(locator,adapter);bound[str(locator)]=sha(locator)
    specs={}
    for phase in ['render','metrics','TRAIN']:
        item=json.loads((p3/'specs'/('prior001_'+phase+'.json')).read_text());item.update(root_release=str(output/'root_release.json'),output=str(output/'prior001'/({'render':'renders','metrics':'visual_gate','TRAIN':'TRAIN_distributions'}[phase])))
        if phase in ['render','TRAIN']:item.update(head_completion=str(locator),label_qualification=str(labels),head_training_spec_sha256=sha(specification))
        else:item['renders']=str(output/'prior001/renders')
        specs['four_UNKNOWN_'+phase]=item
    control=p3/'prior001'
    for path in [control/'renders/complete.json',control/'visual_gate/metrics.json']:
        bound[str(path)]=sha(path)
    for path in (control/'renders').glob('*/*.png'):bound[str(path)]=sha(path)
    for path in (control/'renders').glob('*/physical_and_partition.npz'):bound[str(path)]=sha(path)
    specs['label_overlay_CPU_comparison']=dict(root_release=str(output/'root_release.json'),control=str(control),candidate=str(output/'prior001'),
        ready=str(c/'data/ready.json'),regions=str(c/'native_semantic_grid_20000/corrected_v2/review/region_crops.json'),
        qualified_override_sha256='e3dc58f2bb6293c1551dc3da76286224e34a92bba97bbe70b00d1c0f241cc687',output=str(output/'label_overlay_review'))
    hashes={}
    for name,value in specs.items():
        path=output/'specs'/(name+'.json');write(path,value);hashes[str(path)]=sha(path)
    original_plan=json.loads((p3/'controllers/native_first_visual_gate.json').read_text());host=dict(original_plan['hosts']['olive'],controller=str(output/'code/marshawn_tuning_controller.py'),root=str(output/'queue'))
    jobs=[];previous=[]
    for name,worker,folder,done in [('four_UNKNOWN_render','render_marshawn_LC_semantic_physical.py','prior001/renders','complete.json'),
            ('four_UNKNOWN_metrics','evaluate_marshawn_LC_semantic_physical.py','prior001/visual_gate','metrics.json'),
            ('four_UNKNOWN_TRAIN','audit_marshawn_LC_semantic_train_distributions.py','prior001/TRAIN_distributions','distributions.json'),
            ('label_overlay_CPU_comparison','compare_marshawn_LC_label_overlays_cpu.py','label_overlay_review','comparison.json')]:
        job=dict(original_plan['jobs'][0]);job.update(name=name,command=[host['python'],str(output/'code'/worker),'--spec',str(output/'specs'/(name+'.json')),'--execute'],
            cwd=str(output/'code'),dependencies=previous,progress_files=[str(output/folder/'progress.json')],metrics_file=str(output/folder/done));jobs.append(job);previous=[name]
    plan=output/'controller_plan.json';write(plan,dict(schema_version=1,poll_seconds=15,hosts=dict(olive=host),jobs=jobs))
    write(output/'release_TEMPLATE.json',dict(reviewer='root',LC_physical_render_authorized=False,LC_physical_metrics_authorized=False,
        LC_train_distribution_audit_authorized=False,LC_paired_prior_comparison_authorized=False,LC_uncapped_qualification_authorized=False,
        optimizer_authorized=False,production_after_visualgate=False,spec_sha256=hashes,controller_plan_sha256=sha(plan),bound_files=bound))
    write(output/'CPU_readiness.json',dict(CPU_only=True,CUDA_started=False,optimizer_created=False,new_head_sha256=sha(checkpoint),
        new_label_qualification_sha256=sha(labels),new_training_spec_sha256=sha(specification),control_head_reused='bdb2cc0d8138e4c22c61b4d411227b272b95ceac9e8295d474eecc4a84aa98bc',
        source_capture_Fb8_and_pipeline_c83_unchanged=True,all_P3_GPU_runtime_code_byte_identical=True,
        source_masks_RGB_sigma_field_sampling_reference_tolerance_selectors_unchanged=True,source_legacy_cap1024_only=True,
        all4_actual_same_batch_trace_pixels_preserved=True,localcompletion_adapter_explicit=True,original_SCmetadata_unchanged=True,
        original_verified_control_not_rerendered=True,onlytwoadditional_TRAIN_confidences_changed_infit=True,
        exact4jobDAG=True,all_permissions_false=True,automatic_promotion=False))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.campaign,args.output)
