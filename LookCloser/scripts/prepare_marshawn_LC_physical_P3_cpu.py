"""Reference-only LC physical recovery, bound to actual archived silver heads."""
import argparse
import difflib
import json
from pathlib import Path
import shutil
from prepare_marshawn_frozen_semantic_cpu import sha,write


def prepare(p2,archive,output):
    if output.exists():raise FileExistsError('Use a new immutable physical namespace')
    old_policy=json.loads((p2/'release_TEMPLATE.json').read_text());bound=dict(old_policy['bound_files'])
    for p,digest in bound.items():
        if sha(p)!=digest:raise ValueError('Original P2 binding changed: '+p)
    equivalence=archive/'canonical_equivalence.json';eq=json.loads(equivalence.read_text())
    if sha(equivalence)!='e0519e6ef5ef8309605003efc0c5a0da109549054e721df374265c41996bea39':raise ValueError('Exact root-reviewed execution alias equivalence required')
    release_path=archive/'root_release.json';remote_release=json.loads(release_path.read_text())
    if sha(release_path)!='ee71eaec653dd39d32d63d45a9f0534bf53f06eca1990136054779772466e676' or remote_release.get('LC_first2064_head_authorized')is not True:
        raise ValueError('Exact successful fresh2064 recovery release required')
    label=archive/'label_qualification.json'
    if sha(label)!=eq['alias_label_qualification_sha256']:raise ValueError('Actual alias label metadata required')
    audit=archive/'CPU_completed_head_audit.json'
    if not audit.exists():raise ValueError('Completed-head CPU audit required')
    output.mkdir(parents=True);(output/'code').mkdir();(output/'specs').mkdir();(output/'heads').mkdir();(output/'controllers').mkdir()
    for p in (p2/'code').glob('*.py'):
        target=output/'code'/p.name;shutil.copyfile(p,target);bound[str(target)]=sha(target)
    helper=output/'code/marshawn_stable_packed_density_reference.py'
    shutil.copyfile(Path(__file__).with_name(helper.name),helper)
    if sha(helper)!='66b38a39964b89b1f7de702df2d3a69288f7d35360cabd2dc83b1d28c6a8313c':raise ValueError('Use actual-witness-qualified stable helper bytes')
    bound[str(helper)]=sha(helper)
    observer=output/'code/marshawn_LC_semantic_physical_path.py';old_observer=observer.read_text()
    new_observer=old_observer.replace('import torch\n','import torch\nfrom marshawn_stable_packed_density_reference import stable_packed_density_weights\n',1)
    old_line='proof_weights = packed_density_weights(starts, ends, sigma, ids.long(), count)'
    if new_observer.count(old_line)!=1:raise ValueError('Expected exact unchanged proof call')
    new_observer=new_observer.replace(old_line,'proof_weights = stable_packed_density_weights(starts, ends, sigma, ids.long(), count)')
    observer.write_text(new_observer);bound[str(observer)]=sha(observer)
    (output/'observer_reference_only.diff').write_text(''.join(difflib.unified_diff(old_observer.splitlines(True),new_observer.splitlines(True),fromfile='sealed_P2',tofile='P3_reference_only')))
    for p in [equivalence,release_path,label,audit]:bound[str(p)]=sha(p)
    import torch
    import numpy as np
    from marshawn_frozen_semantic_head import FrozenActorGrid,tensor_digest
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('CPU-only preparation required')
    specs={};heads={}
    canonical_contract_path=Path(eq['source_local_head'])/'head_input_contract.json'
    canonical_contract=json.loads(canonical_contract_path.read_text())
    ref_path=canonical_contract_path.parent.parent/'frozen_native_semantic_prior001_candidate_v1/camera_history_reference.json'
    reference=json.loads(ref_path.read_text())['camera_history'];bound[str(ref_path)]=sha(ref_path)
    for name,prior in [('prior001',.01),('prior099',.99)]:
        run=name+'_fresh2064';checkpoint=archive/'runs'/run/'head.pt';original_complete=archive/'runs'/run/'complete.json'
        initial_path=archive/'runs'/run/'initial_state.json';training_spec=archive/'specs'/(run+'.json')
        for p in [checkpoint,original_complete,initial_path,training_spec]:bound[str(p)]=sha(p)
        complete=json.loads(original_complete.read_text());initial=json.loads(initial_path.read_text());trained=torch.load(checkpoint,map_location='cpu',weights_only=False)
        alias_spec_digest=next(digest for path,digest in eq['alias_specs_sha256'].items()if Path(path).name==training_spec.name)
        if (sha(training_spec)!=alias_spec_digest or trained['spec_sha256']!=alias_spec_digest or sha(checkpoint)!=complete['checkpoint_sha256'] or
                complete['actual_head_updates']!=2064 or complete['iteration']!=2064 or trained['iteration']!=2064 or
                initial['initial_iteration']!=0 or initial['initial_Adam_count']!=0 or initial['restoration']is not None or
                not complete['native_model_not_loaded_or_optimized'] or not complete['native_checkpoint_and_caches_unchanged'] or
                trained['source_kind']!='DIRECT_FROZEN_LC_NATIVE_VISIBILITY' or trained['model_digest']!=eq['source_pipeline_digest'] or
                trained['LC_capture_receipt_sha256']!=eq['canonical_capture_receipt_sha256'] or
                trained['label_qualification_sha256']!=sha(label) or trained['initial_probability']!=prior or
                not trained['world_to_box_is_identity'] or trained['camera_history']!=reference):
            raise ValueError('Actual archived fresh2064 scientific metadata differs: '+name)
        head=FrozenActorGrid(canonical_contract['bounds'],np.eye(4),128,initial_probability=prior)
        head.load_state_dict(trained['head'],strict=True)
        if tensor_digest(head)!=trained['head_digest'] or not torch.equal(head.bounds,torch.tensor(canonical_contract['bounds'])) or not torch.equal(head.world_to_box,torch.eye(4)):
            raise ValueError('Archived head state/frame differs')
        destination=output/'heads'/(name+'_local_completion_adapter.json')
        adapter=dict(complete,kind='DERIVED_LOCAL_CHECKPOINT_LOCATOR_COMPLETION_ADAPTER',checkpoint=str(checkpoint),
            original_remote_checkpoint=complete['checkpoint'],original_completion_sha256=sha(original_complete),
            canonical_equivalence_sha256=sha(equivalence),source_remote_release_sha256=sha(release_path),scientific_checkpoint_bytes_unchanged=True)
        write(destination,adapter);bound[str(destination)]=sha(destination)
        heads[name]=dict(archive_checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),original_completion=str(original_complete),
            original_completion_sha256=sha(original_complete),local_completion_adapter=str(destination),local_completion_adapter_sha256=sha(destination),
            actual_remote_training_spec_sha256=sha(training_spec),alias_label_qualification_sha256=sha(label),initial_probability=prior)
        for phase in ['render','TRAIN','metrics']:
            source=json.loads((p2/'specs'/(name+'_'+phase+'.json')).read_text())
            source['root_release']=str(output/'root_release.json')
            source['output']=str(output/name/({'render':'renders','TRAIN':'TRAIN_distributions','metrics':'visual_gate'}[phase]))
            if phase in ['render','TRAIN']:
                source.update(head_completion=str(destination),label_qualification=str(label),head_training_spec_sha256=sha(training_spec),expected_initial_probability=prior)
            if phase=='metrics':source['renders']=str(output/name/'renders')
            specs[name+'_'+phase]=source
    qualification=json.loads((p2/'specs/uncapped_qualification.json').read_text());qualification.update(root_release=str(output/'root_release.json'),output=str(output/'uncapped_qualification'))
    specs['uncapped_qualification']=qualification
    paired=json.loads((p2/'specs/paired_CPU_review.json').read_text());paired.update(root_release=str(output/'root_release.json'),
        prior99=str(output/'prior099'),prior01=str(output/'prior001'),prior99_TRAIN=str(output/'prior099/TRAIN_distributions/distributions.json'),
        prior01_TRAIN=str(output/'prior001/TRAIN_distributions/distributions.json'),output=str(output/'paired_review'))
    specs['paired_CPU_review']=paired
    spec_hashes={}
    for name,value in specs.items():
        path=output/'specs'/(name+'.json');write(path,value);spec_hashes[str(path)]=sha(path)
    base=json.loads((p2/'controllers/uncapped_qualification.json').read_text());host=dict(base['hosts']['olive'])
    host['controller']=str(output/'code/marshawn_tuning_controller.py')
    def job(name,worker,dependencies,folder,complete):
        template=dict(base['jobs'][0]);template.update(name=name,command=[host['python'],str(output/'code'/worker),'--spec',str(output/'specs'/(name+'.json')),'--execute'],
            cwd=str(output/'code'),dependencies=dependencies,progress_files=[str(folder/'progress.json')],metrics_file=str(folder/complete));return template
    native=[];previous=[]
    for phase,worker,folder,done in [('render','render_marshawn_LC_semantic_physical.py','renders','complete.json'),
            ('metrics','evaluate_marshawn_LC_semantic_physical.py','visual_gate','metrics.json'),
            ('TRAIN','audit_marshawn_LC_semantic_train_distributions.py','TRAIN_distributions','distributions.json')]:
        for prior in ['prior001','prior099']:
            name=prior+'_'+phase;native.append(job(name,worker,previous,output/prior/folder,done));previous=[name]
    native.append(job('paired_CPU_review','compare_marshawn_LC_semantic_priors_cpu.py',previous,output/'paired_review','comparison.json'))
    plans={}
    for stage,jobs in [('native_first_visual_gate',native),('uncapped_qualification',[job('uncapped_qualification','render_marshawn_LC_semantic_physical.py',[],output/'uncapped_qualification','complete.json')])]:
        path=output/'controllers'/(stage+'.json');settings=dict(host,root=str(output/'queue'/stage))
        write(path,dict(schema_version=1,poll_seconds=15,hosts=dict(olive=settings),jobs=jobs));plans[stage]=sha(path)
    write(output/'release_TEMPLATE.json',dict(reviewer='root',LC_physical_render_authorized=False,LC_physical_metrics_authorized=False,
        LC_train_distribution_audit_authorized=False,LC_paired_prior_comparison_authorized=False,LC_uncapped_qualification_authorized=False,
        optimizer_authorized=False,production_after_visualgate=False,spec_sha256=spec_hashes,controller_plan_sha256=plans,bound_files=bound))
    write(output/'CPU_readiness.json',dict(CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),optimizer_created=False,
        source_pipeline_digest=eq['source_pipeline_digest'],canonical_capture_receipt_sha256=eq['canonical_capture_receipt_sha256'],
        canonical_equivalence_sha256=sha(equivalence),actual_heads=heads,alias_label_sha256=sha(label),
        original_SC_metadata_and_checkpoint_bytes_unmodified=True,local_completion_adapters_explicit=True,
        source_and_field_sampling_RGB_dtype_unchanged=True,unchanged_reference_tolerance=2e-5,
        only_runtime_code_difference='Observer imports stable helper and changes one independent proof call. Native field/weights/compositor/sampler unchanged.',
        native_controller_priority='p.01 all3FHD/five physicalselectors first, thenp.99, thenfixedmetrics/TRAIN/pairedCPU. Each DCview saves immediately beforeHC/KA.',
        uncapped_controller_separate=True,actual_witness_required_and_available=True,source_P2_preserved=True,
        automatic_promotion=False,all_permissions_false=True))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--P2',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.P2,args.archive,args.output)
