"""Seal a separate exact6522-pixel TRAIN RGB-confidence correction; never launch."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Frozen source anchor changed: '+old[:80])
    return text.replace(old,new)

def main(c,out):
    parent=c/'semantic_actor_candidate_v2_20261106';qualified=c/'native_semantic_grid_20000/bag_qualification'
    proof=qualified/'train_bag_qualification.json'
    if sha(proof)!='5e904fe04a4647a6312496017ae928e24f0077340f93ba016beacebd942d15d7':raise ValueError('Qualified source changed')
    out.mkdir(parents=True,exist_ok=False);source=out/'bag_loss_source';source.mkdir()
    meta=json.loads(proof.read_text());shutil.copy2(proof,source/proof.name);records=[]
    ready=json.loads((c/'data/ready.json').read_text());inputs={r['camera']:r for r in ready['records']}
    for row in meta['records']:
        camera=row['camera'];assert row['split']=='train'and camera not in ['DC','HC','KA']
        original=c/'data/semantics'/f'{camera}.npz'
        if sha(original)!=inputs[camera]['semantic_sha256']:raise ValueError('Main data inventory changed')
        for key,name in [('exclusion',f'{camera}_bag_exclusion_HD.npy'),('unknown',f'{camera}_bag_unknown_HD.npy')]:
            if sha(qualified/name)!=row[key+'_sha256']:raise ValueError('Qualified mask changed')
            shutil.copy2(qualified/name,source/name)
        mask=np.load(source/f'{camera}_bag_exclusion_HD.npy',allow_pickle=False);shell=np.load(source/f'{camera}_bag_unknown_HD.npy',allow_pickle=False)
        with np.load(original,allow_pickle=False)as z:
            count=int((mask&z['grey']).sum())
            assert count==row['existing_actor_pixels_removed']['grey']and not(mask&z['black']).any()and not(mask&shell).any()
        native=Path(row['raw_source'])
        if sha(native)!=row['raw_source_sha256']:raise ValueError('Native source crop changed')
        shutil.copy2(native,source/f'{camera}_native_source.png')
        records.append(dict(camera=camera,exclusion=f'{camera}_bag_exclusion_HD.npy',unknown=f'{camera}_bag_unknown_HD.npy',qualified_polygon_pixels=int(mask.sum()),false_grey_pixels=count,original_semantic_sha256=sha(original)))
    for name in ['MC_TRAIN_temporal.png','NA_TRAIN_temporal.png']:shutil.copy2(qualified/name,source/name)
    assert sum(r['false_grey_pixels']for r in records)==6522
    manifest=dict(kind='QUALIFIED_TRAIN_FALSE_GREY_RGB_WEIGHT_ONLY',expected_false_grey_pixels=6522,qualified_polygon_pixels=6874,uses_EVAL=False,fixed_admission_changed=False,proposal_pools_changed=False,records=records,files_sha256={p.name:sha(p)for p in sorted(source.iterdir())},qualification_sha256=sha(proof),unchanged_data_ready_sha256=sha(c/'data/ready.json'),unchanged_transforms_sha256=sha(c/'data/transforms.json'),policy='Zero only exclusion AND original grey; preserve every other confidence value, unknown shell, masks and proposal pools; recompute exact E_q[w]')
    write(source/'manifest.json',manifest)
    shutil.copytree(parent/'code',out/'code',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(parent/'source',out/'source',copy_function=os.link)
    scripts=out/'code/LookCloser/scripts';worker=scripts/'train_marshawn_semantic_admission.py';text=worker.read_text()
    text=replace_once(text,"'data', 'output', 'followup_variant', 'semantic_admission',","'data', 'output', 'followup_variant', 'semantic_admission', 'bag_loss_correction',")
    text=replace_once(text,'    return spec\n','    from marshawn_bag_loss_correction import validate_manifest\n    correction = spec.get("bag_loss_correction", {})\n    if set(correction) != {"manifest", "manifest_sha256"}: raise ValueError("Exact bag-loss source required")\n    validate_manifest(correction["manifest"], correction["manifest_sha256"])\n    return spec\n')
    text=replace_once(text,"('train_marshawn_semantic_admission.py', 'marshawn_semantic_admission.py',","('train_marshawn_semantic_admission.py', 'marshawn_bag_loss_correction.py', 'marshawn_semantic_admission.py',")
    text=replace_once(text,"    semantic_root = Path(spec['semantic_admission']['manifest']).parent",'''    correction_root = Path(spec['bag_loss_correction']['manifest']).parent
    input_hashes['bag_loss_manifest'] = sha(correction_root / 'manifest.json')
    for name, digest in json.loads((correction_root / 'manifest.json').read_text())['files_sha256'].items():
        if sha(correction_root / name) != digest: raise ValueError('Bag correction source changed: ' + name)
        input_hashes['bag_loss/' + name] = digest
    semantic_root = Path(spec['semantic_admission']['manifest']).parent''')
    text=replace_once(text,"    if 'sampling_state' in state:\n",'''    from marshawn_bag_loss_correction import apply_correction
    correction_receipt = apply_correction(sampler, spec['bag_loss_correction']['manifest'], spec['bag_loss_correction']['manifest_sha256'])
    atomic_json(output / 'bag_loss_application.json', correction_receipt)
    state['bag_loss_application'] = correction_receipt
    if 'sampling_state' in state:
''')
    worker.unlink();worker.write_text(text)
    guard=scripts/'run_marshawn_semantic_release.py';text=guard.read_text()
    text=replace_once(text,"    return release\n",'''    correction=spec.get('bag_loss_correction',{})
    if (release.get('bag_loss_cpu_proof_passed')is not True or release.get('bag_manifest_sha256')!=correction.get('manifest_sha256')):raise ValueError('Bag correction requires exact CPU/source review')
    from marshawn_bag_loss_correction import validate_manifest
    validate_manifest(correction['manifest'],correction['manifest_sha256'])
    return release
''');guard.unlink();guard.write_text(text)
    shutil.copy2(Path(__file__).with_name('marshawn_bag_loss_correction.py'),scripts/'marshawn_bag_loss_correction.py')
    parent_inventory=json.loads((parent/'source_manifest.json').read_text())['files_sha256'];files={str(p.relative_to(out/'code')):sha(p)for p in (out/'code').rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'}
    changed=[n for n in parent_inventory if files[n]!=parent_inventory[n]];added=sorted(set(files)-set(parent_inventory))
    if sorted(changed)!=['LookCloser/scripts/run_marshawn_semantic_release.py','LookCloser/scripts/train_marshawn_semantic_admission.py']or added!=['LookCloser/scripts/marshawn_bag_loss_correction.py']:raise ValueError('Unexpected frozen code changes')
    write(out/'source_manifest.json',dict(parent_source_manifest_sha256=sha(parent/'source_manifest.json'),files_sha256=files,changed_from_R0=changed,added_from_R0=added))
    spec=json.loads((parent/'spec.json').read_text());old=str(Path(spec['output']).parent);new=str(Path(old).with_name(out.name));spec=json.loads(json.dumps(spec).replace(old,new));spec['bag_loss_correction']=dict(manifest=new+'/bag_loss_source/manifest.json',manifest_sha256=sha(source/'manifest.json'));write(out/'spec.json',spec)
    plan=json.loads((parent/'controller_plan.json').read_text());plan=json.loads(json.dumps(plan).replace(old,new).replace('exp_actor_semantic_W64pad1_r0','exp_actor_semantic_W64pad1_bagloss_r0'));write(out/'controller_plan.pending.json',plan)
    write(out/'runtime_policy.json',dict(prepared_only=True,production_started=False,only_effective_RGB_weight_change_pixels=6522,qualified_polygon_total=6874,unchanged_pixels_include_unlabeled_polygon_remainder=352,unknown_shell_unchanged=True,black_hair_feet_unchanged=True,raw_masks_proposal_pools_FAS_unchanged=True,intentionally_stale_proposal_labels=True,exact_normalizer_recomputed=True,model_and_fixed_admission_unchanged=True,required='Root CPU/source/config review and exact separate release',spec_sha256=sha(out/'spec.json'),bag_manifest_sha256=sha(source/'manifest.json')))
    print(json.dumps(dict(root=str(out),source_manifest_sha256=sha(out/'source_manifest.json'),spec_sha256=sha(out/'spec.json'),bag_manifest_sha256=sha(source/'manifest.json'),launched=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
