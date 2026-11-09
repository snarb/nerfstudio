"""Fit-only four-UNKNOWN overlay; existing cache/contracts/code stay unchanged."""
import argparse
import json
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha,write


def prepare(c,sc,output):
    if output.exists():raise FileExistsError('New controlled experiment namespace required')
    alias=c/'frozen_LC_semantic_head_silver_alias_v1';remote_alias=sc/alias.name;remote=sc/output.name
    qualification=c/'native_actor_bag_projection_review_CPU_v1/root_LA_two_UNKNOWN_qualification.json'
    qualification_sha='e3dc58f2bb6293c1551dc3da76286224e34a92bba97bbe70b00d1c0f241cc687'
    if sha(qualification)!=qualification_sha:raise ValueError('Exact new root LA qualification required')
    new=json.loads(qualification.read_text());old_labels=json.loads((alias/'label_qualification.json').read_text())
    if not new['root_review_complete']or new['reviewer']!='root':raise ValueError('Source review not complete')
    archive=c/'frozen_LC_semantic_head_archives/recovery1_20261107_v1'
    baseline=archive/'runs/prior001_fresh2064/head.pt'
    if sha(baseline)!='bdb2cc0d8138e4c22c61b4d411227b272b95ceac9e8295d474eecc4a84aa98bc':raise ValueError('Existing controlled p01 checkpoint required')
    original_spec=alias/'specs/prior001_fresh2064.json';spec=json.loads(original_spec.read_text())
    capture=json.loads((c/'frozen_LC_semantic_capture_plan_v4/capture/complete.json').read_text())
    import numpy as np
    offset=0;observed=[]
    for row in capture['records']:
        if not row['rays']:continue
        if row['camera']=='LA':
            with np.load(row['archive'])as z:
                for q in new['unknown_queries']:
                    i=q['camera_ray_index']
                    if offset+i!=q['global_query_index']or z['pixel_xy'][i].tolist()!=q['pixel_xy']or z['semantic_target'][i]!=1 or z['label_weight'][i]!=1:
                        raise ValueError('Exact existing LA positive query differs')
                    observed.append(q['global_query_index'])
        offset+=row['rays']
    if sorted(observed)!=[54411,54594]or offset!=69888:raise ValueError('Fixed query identity differs')
    output.mkdir(parents=True)
    labels=dict(old_labels)
    labels.update(kind='DERIVED_FIT_ONLY_FOUR_UNKNOWN_OVERLAY',source_cache_unknown_count=2,fit_overlay_unknown_count=4,
        original_two_IA_label_qualification_sha256=sha(alias/'label_qualification.json'),
        root_additional_LA_qualification=str(sc/qualification.relative_to(c)),root_additional_LA_qualification_sha256=qualification_sha,
        note='Onlyfitconfidence0 for exactIA2+LA2; originalcache/contracts still2IA. Targets1 preserved. No newnegative/free-space/alphaGT label.')
    labels['unknown_queries']=list(old_labels['unknown_queries'])+[dict(q,qualification=str(sc/qualification.relative_to(c)),qualification_sha256=qualification_sha)for q in new['unknown_queries']]
    label_path=output/'label_qualification.json';write(label_path,labels)
    spec.update(phase='direct_LC_p01_four_UNKNOWN_fresh2064',root_release=str(remote/'root_release.json'),
        label_qualification=str(remote/'label_qualification.json'),output=str(remote/'runs/prior001_fresh2064'),
        qualified_UNKNOWN_count=4,source_cache_unknown_count=2,fit_overlay_unknown_count=4,
        baseline_control_checkpoint_sha256=sha(baseline),controlled_change='Only2additionalroot-qualifiedLApositiveconfidences0; noquerydrop/targetchange/newnegative.')
    spec_path=output/'spec.json';write(spec_path,spec)
    base=json.loads((alias/'root_release.json').read_text());bound=dict(base['bound_files'])
    bound[str(sc/qualification.relative_to(c))]=qualification_sha
    # Explicitly include every qualification leaf read by the unchanged worker.
    for q in labels['unknown_queries']:bound[q['qualification']]=q['qualification_sha256']
    bound[str(remote/'label_qualification.json')]=sha(label_path)
    bound[str(remote_alias/'specs/prior001_fresh2064.json')]=sha(original_spec)
    bound[str(remote_alias/'runs/prior001_fresh2064/head.pt')]=sha(baseline)
    equivalence=alias/'canonical_equivalence.json';bound[str(remote_alias/'canonical_equivalence.json')]=sha(equivalence)
    plan=json.loads((alias/'controller_plan.json').read_text());plan['hosts']['silver']['root']=str(remote/'queue')
    # Match the already verified p01 control's physical GPU1/SM120 runtime.
    plan['hosts']['silver']['gpus']=[1]
    job=dict(plan['jobs'][1]);job.update(name='LC_p01_four_UNKNOWN_fresh2064',command=[str(sc/'python'),
        str(sc/'frozen_LC_semantic_head_plan_v1/code/probe_marshawn_LC_frozen_semantic_head.py'),'--spec',str(remote/'spec.json'),'--execute'],
        dependencies=[],progress_files=[str(remote/'runs/prior001_fresh2064/progress.json')],metrics_file=str(remote/'runs/prior001_fresh2064/complete.json'))
    job['env']=dict(job['env'],PYTHONUNBUFFERED='1');plan['jobs']=[job]
    controller_path=output/'controller_plan.json';write(controller_path,plan)
    controller=Path(plan['hosts']['silver']['controller']);bound[str(controller)]=sha(c/controller.relative_to(sc))
    write(output/'root_review_TEMPLATE.json',dict(reviewer='root',LC_technical_head_authorized=False,LC_first2064_head_authorized=False,
        LC_physical_render_authorized=False,LC_uncapped_qualification_authorized=False,production_after_visualgate=False,
        label_qualification_sha256=sha(label_path),bound_files=bound,spec_sha256={str(remote/'spec.json'):sha(spec_path)},
        controller_plan_sha256=sha(controller_path),source_capture_and_code_unchanged=True,
        permitted_updates_if_released=2064,control_rerun_authorized=False))
    difference={k:dict(before=json.loads(original_spec.read_text()).get(k),after=v)for k,v in spec.items()if json.loads(original_spec.read_text()).get(k)!=v}
    numeric_keys=['updates','seed','grid_resolution','learning_rate','initial_probability','torch_version','cuda_version','camera_history_reference','manifest','capture_receipt','capacity_policy']
    if any(spec[k]!=json.loads(original_spec.read_text())[k]for k in numeric_keys):raise ValueError('Scientificnumeric/sourcecontract change beyond labeloverlay')
    write(output/'controlled_factor_CPU_proof.json',dict(CPU_only=True,CUDA_started=False,optimizer_created=False,
        control_checkpoint_sha256=sha(baseline),control_spec_sha256=sha(original_spec),new_spec_sha256=sha(spec_path),
        unchanged_numeric_and_source_keys=numeric_keys,spec_metadata_differences=difference,
        actual_original_cache_unknown_count=2,new_fit_overlay_unknown_count=4,
        old_IA_global_queries=[40470,40622],additional_LA_global_queries=[54411,54594],
        all69888_source_rows_targets_and_arrays_unchanged=True,no_fresh_control_rerun=True,
        source_capture_receipt_sha256='fb8b875d8c7f3676c3dfbd187b4c76c3ee40ec8a3b62a940fdf33bb97d99073d',
        source_pipeline_digest='c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636',
        unknown_overlay_qualification_sha256=qualification_sha,remote_root=str(remote),
        source_code_not_copied_or_modified=True,outputs_must_not_exist_before_release=True,
        controller_exactly_one_head_job=True,physical_GPU_index=1,baseline_p01_GPU_UUID='GPU-26c63492-0230-896f-208e-10e4eaca4c3c',
        numeric_caveat='Unchangedcode/seed/history; CUDAgrid_sample atomics can differaround6e-7 inprobability. No bitwise quality claim.'))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--silver-campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.campaign,args.silver_campaign,args.output)
