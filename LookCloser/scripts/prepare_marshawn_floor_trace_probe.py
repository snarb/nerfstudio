"""Stage an isolated reviewed probe proposal; imports no GPU/model runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main(a):
    root=a.root;root.mkdir(parents=True,exist_ok=False)
    helpers=root/'helpers';helpers.mkdir();queries=root/'queries';queries.mkdir();proof=root/'proof';proof.mkdir()
    successful=a.campaign/'independent_occupancy/semantic_selected_geometry/helpers'
    for name in ['audit_marshawn_semantic_checkpoint.py','audit_marshawn_tuning_checkpoint.py','probe_marshawn_occupancy.py']:
        shutil.copyfile(successful/name,helpers/name)
    original=helpers/'probe_marshawn_occupancy.py';text=original.read_text();prior=sha(original)
    edits=[("rendered = result['rgb'].detach().float().cpu().numpy().reshape(-1).tolist()",
            "rendered = result['rgb'].detach().float().cpu().numpy().reshape(-1).tolist(); accumulated = float(result['accumulation'].detach().float().item())"),
           ('actual_weights = None; saturation = None; rendered = None',
            'actual_weights = None; saturation = None; rendered = None; accumulated = None'),
           ('model_rgb=rendered, saturation=saturation','model_rgb=rendered, model_accumulation=accumulated, saturation=saturation')]
    for before,after in edits:
        assert text.count(before)==1,before
        text=text.replace(before,after)
    original.write_text(text)
    (proof/'trace_alpha_receipt_delta.json').write_text(json.dumps(dict(original=str(successful/original.name),original_sha256=prior,
        modified_sha256=sha(original),edits=edits,only_adds_original_renderer_accumulation_to_receipt=True),indent=2)+'\n')
    base=helpers/'audit_marshawn_tuning_checkpoint.py';prior=sha(base);text=base.read_text()
    before='    import probe_marshawn_occupancy as probe'
    after='''    # Bind the receipt-instrumented helper explicitly; frozen source sys.path has priority above.
    import importlib.util
    probe_path = Path(__file__).resolve().parent / 'probe_marshawn_occupancy.py'
    probe_spec = importlib.util.spec_from_file_location('probe_marshawn_occupancy', probe_path)
    probe = importlib.util.module_from_spec(probe_spec)
    sys.modules[probe_spec.name] = probe
    probe_spec.loader.exec_module(probe)
    assert Path(probe.__file__).resolve() == probe_path'''
    assert text.count(before)==1;text=text.replace(before,after)
    before_mkdir='    args.output.mkdir(parents=True, exist_ok=False)'
    after_mkdir=before_mkdir+'''
    import hashlib
    import_binding = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                      for p in [__file__, probe.__file__, Path(__file__).parent / 'audit_marshawn_semantic_checkpoint.py']}
    (args.output / 'imported_helpers.json').write_text(json.dumps(import_binding, indent=2) + '\\n')
    print('IMPORTED_HELPERS ' + json.dumps(import_binding), flush=True)'''
    assert text.count(before_mkdir)==1;text=text.replace(before_mkdir,after_mkdir);base.write_text(text)
    (proof/'explicit_helper_binding_delta.json').write_text(json.dumps(dict(original=str(successful/base.name),
        original_sha256=prior,modified_sha256=sha(base),edits=[[before,after],[before_mkdir,after_mkdir]],
        model_sampling_or_weight_changes=False),indent=2)+'\n')
    for name in ['marshawn_signed_height_bins.py','partition_marshawn_trace_heights.py','run_marshawn_floor_trace_probe.py',
                 'prepare_marshawn_floor_trace_probe.py','prepare_marshawn_floor_trace_queries.py','test_signed_height_bins.py']:
        shutil.copyfile(a.inputs/name,helpers/name)
    prepared=[]
    for split in ['eval','train']:
        path=queries/(split+'_queries.json');shutil.copyfile(a.inputs/path.name,path)
        request=json.loads(path.read_text());assert request['split']==split
        prepared.append(dict(split=split,path=str(path),sha256=sha(path),rays=len(request['pixels'])))
    for p in a.inputs.glob('*_source_queries.png'):shutil.copyfile(p,queries/p.name)
    for name in ['source_query_contact.jpg','source_query_review.json']:
        shutil.copyfile(a.inputs/name,queries/name)
    total=sum(row['rays']for row in prepared);assert total==33 and total<=64
    result=subprocess.run([a.python,str(helpers/'test_signed_height_bins.py')],capture_output=True,text=True)
    (proof/'CPU_tests.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    shutil.copyfile('/tmp/signed_weight_clothing103_CPU_validation/partition.json',proof/'existing_clothing103_partition.json')
    models=[];trees=[];bindings={}
    for label,relative,code,manifest,key,step,coefficient in [
        ('R0_selected111316','independent_occupancy/semantic_actor_progress/step-000111316','semantic_actor_candidate_v2_20261106','source_manifest.json','files_sha256',111316,.01),
        ('control001_34376','independent_occupancy/semantic_late_pair_v2/control001/step-000034376','semantic_late_distortion_candidate_v2','source_inventory.json','files',34376,.01),
        ('treatment1_34376','independent_occupancy/semantic_late_pair_v2/treatment1/step-000034376','semantic_late_distortion_candidate_v2','source_inventory.json','files',34376,1.)]:
        archive=a.campaign/relative;source=a.campaign/code/'code';inventory=a.campaign/code/manifest
        record=json.loads((archive/'archive_receipt.json').read_text())
        if 'artifacts_sha256'in record:
            for name,digest in record['artifacts_sha256'].items():assert sha(archive/name)==digest,name
            assert record['effective_distortion_coefficient']==coefficient
        else:
            assert sha(archive/'checkpoint.ckpt')==record['checkpoint_sha256']
            assert sha(archive/('step-%09d_grid.npz'%step))==record['grid_sha256']
        if str(inventory)not in [r['manifest']for r in trees]:
            trees.append(dict(root=str(source),manifest=str(inventory),key=key))
        for name in ['checkpoint.ckpt','config.yml','archive_receipt.json','step-%09d_grid.json'%step,'step-%09d_grid.npz'%step]:
            bindings[str(archive/name)]=sha(archive/name)
        bindings[str(inventory)]=sha(inventory)
        pngs={camera:str(archive/'renders'/(camera+'.png'))for camera in ['DC','HC','KA']}
        bindings.update({path:sha(path)for path in pngs.values()})
        models.append(dict(label=label,checkpoint=str(archive/'checkpoint.ckpt'),checkpoint_sha256=sha(archive/'checkpoint.ckpt'),
            config=str(archive/'config.yml'),receipt=str(archive/('step-%09d_grid.json'%step)),code_root=str(source),
            step=step,coefficient=coefficient,feature_reweighting_strength=.3 if step>75940 else 1.,reference_PNGs=pngs))
    for source in trees:
        for name,digest in json.loads(Path(source['manifest']).read_text())[source['key']].items():
            assert sha(Path(source['root'])/name)==digest,name
    for p in root.rglob('*'):
        if p.is_file()and'__pycache__'not in str(p):bindings[str(p)]=sha(p)
    for p in [a.campaign/'data/ready.json',a.campaign/'data/transforms.json']:
        bindings[str(p)]=sha(p)
    plan=dict(kind='READONLY_SMALL_NATIVE_WEIGHT_FOOT_PROBE',python=a.python,gpu=a.gpu,data=str(a.campaign/'data'),
        helpers_source=str(successful),bound_files=bindings,source_trees=trees,models=models,queries=prepared,
        total_unique_rays=total,total_model_ray_pairs=total*len(models),optimizer_updates=0,
        allocator_cap_gib=8,max_process_mib=10000,max_seconds=3600,
        CPU_tests_passed=8,real103_clothing_reference_validation='Numeric implementation test only; not floor evidence',
        no_new_masks=True,no_carving=True,bounded_crops_authorized=False,
        parity=dict(max_float_RGB_error_in_PNG_bytes=1.01,max_weight_alpha_or_RGB_error=2e-6,
                    archived_PNG_has_alpha=False,alpha_reference='Original renderer accumulation'),
        scope='27 fixed foot-ROI EVAL diagnostic rays and 6 measured TRAIN shoe observations; no fitted EVAL geometry.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (root/'release.template.json').write_text(json.dumps(dict(approved_readonly_probe=False,plan_sha256=sha(root/'plan.json')),indent=2)+'\n')
    print(json.dumps(dict(root=str(root),plan_sha256=sha(root/'plan.json'),bindings=len(bindings),models=len(models),rays=total)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','campaign','inputs']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--gpu',type=int,default=2);main(p.parse_args())
