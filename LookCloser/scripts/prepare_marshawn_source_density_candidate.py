"""Freeze a one-factor source-coordinate density lookup candidate; never launch."""
import argparse,hashlib,json,os,shutil
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n')
def replace_once(t,old,new):
    if t.count(old)!=1: raise ValueError('Frozen anchor changed: '+old[:100])
    return t.replace(old,new)
def edit(p,fn):
    t=fn(p.read_text());p.unlink();p.write_text(t)


def main(c,out):
    parent=c/'semantic_actor_candidate_v2_20261106';out.mkdir(exist_ok=False)
    for name in ['code','source']:shutil.copytree(parent/name,out/name,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    scripts=out/'code/LookCloser/scripts'
    for name in ['marshawn_source_point_admission.py','marshawn_source_density_adapter.py']:shutil.copy2(Path(__file__).with_name(name),scripts/name)
    def model(t):
        t=replace_once(t,'        semantic_identity_control: bool=False','        semantic_identity_control: bool=False\n        semantic_density_lookup: str="source_point_pad1"')
        t=replace_once(t,'            self.semantic_lookup=install_density_admission(self.field,self.semantic_admission,bounds)', '''            if c.semantic_density_lookup != 'source_point_pad1': raise ValueError('Reviewed source-point density lookup required')
            from marshawn_source_density_adapter import install_source_density_admission
            # load_support above validated this exact source and its positive-support receipts.
            with np.load(Path(c.semantic_manifest).parent/'semantic_grid.npz',allow_pickle=False) as src:
                source_mask=src['semantic_gate_pad1'].copy();source_bounds=src['aabb'].copy()
            if c.semantic_identity_control: source_mask[:]=True
            density_mask=torch.from_numpy(source_mask).to(bounds.device)
            self.semantic_lookup=install_source_density_admission(self.field,density_mask,source_bounds,epsilon=2e-7)
            receipt.update(density_lookup='source_point_pad1',density_mask_sha256=hashlib.sha256(source_mask.tobytes()).hexdigest(),density_source_bounds=source_bounds.tolist(),density_epsilon=2e-7,traversal_cover_unchanged=True)''')
        return t
    edit(scripts/'marshawn_semantic_admission.py',model)
    def worker(t):
        t=replace_once(t,"{'manifest', 'manifest_sha256', 'identity_control', 'release_receipt'}","{'manifest', 'manifest_sha256', 'identity_control', 'release_receipt', 'density_lookup'}")
        t=replace_once(t,'    return spec\n',"    if gate['density_lookup'] != 'source_point_pad1': raise ValueError('Exact source-point density mode required')\n    return spec\n")
        t=replace_once(t,"semantic_data=spec['data'], semantic_identity_control=spec['semantic_admission']['identity_control'])","semantic_data=spec['data'], semantic_identity_control=spec['semantic_admission']['identity_control'],\n        semantic_density_lookup=spec['semantic_admission']['density_lookup'])")
        return replace_once(t,"'marshawn_semantic_admission.py', 'marshawn_semantic_cell_admission.py'","'marshawn_semantic_admission.py', 'marshawn_source_point_admission.py', 'marshawn_source_density_adapter.py', 'marshawn_semantic_cell_admission.py'")
    edit(scripts/'train_marshawn_semantic_admission.py',worker)
    def guard(t):
        return replace_once(t,'    return release\n',"    if gate.get('density_lookup') != 'source_point_pad1' or release.get('source_point_integrated_native_proof_passed') is not True: raise ValueError('Source-point integration requires specific native review')\n    return release\n")
    edit(scripts/'run_marshawn_semantic_release.py',guard)
    files={str(p.relative_to(out/'code')):sha(p)for p in (out/'code').rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'};old=json.loads((parent/'source_manifest.json').read_text())['files_sha256']
    changed=sorted(k for k in old if old[k]!=files[k]);added=sorted(set(files)-set(old))
    assert changed==['LookCloser/scripts/marshawn_semantic_admission.py','LookCloser/scripts/run_marshawn_semantic_release.py','LookCloser/scripts/train_marshawn_semantic_admission.py']
    assert added==['LookCloser/scripts/marshawn_source_density_adapter.py','LookCloser/scripts/marshawn_source_point_admission.py']
    write(out/'source_manifest.json',dict(parent_source_manifest_sha256=sha(parent/'source_manifest.json'),files_sha256=files,changed_from_R0=changed,added_from_R0=added))
    spec=json.loads((parent/'spec.json').read_text());oldroot=str(Path(spec['output']).parent);newroot=str(Path(oldroot).with_name(out.name));spec=json.loads(json.dumps(spec).replace(oldroot,newroot));spec['semantic_admission']['density_lookup']='source_point_pad1';write(out/'spec.json',spec)
    plan=json.loads((parent/'controller_plan.json').read_text());plan=json.loads(json.dumps(plan).replace(oldroot,newroot).replace('exp_actor_semantic_W64pad1_r0','exp_actor_semantic_sourcepoint_W64pad1_r0'));write(out/'controller_plan.pending.json',plan)
    proofs=out/'inherited_source_proofs';proofs.mkdir()
    for name in ['source_point_lookup_cpu_proof.json','source_point_lookup_GPU_proof.json']:shutil.copy2(c/'independent_occupancy'/name,proofs/name)
    write(out/'candidate_policy.json',dict(prepared_only=True,production_launched=False,only_density_lookup_changed=True,traversal_admission_cells=131911,source_density_cells=105049,native_radius=0,actor_confidence_width=64,bag_correction=False,all_original_RGB_confidence_and_sampling_unchanged=True,outside_source_is_UNKNOWN=True,closed_contact_epsilon=2e-7,source_bounds_float64=True,additional_world_to_box_transform=False,positive_interval_receipt_inherited=True,integrated_native_identity_and_Adam_resume_pending=True))
    print(json.dumps(dict(root=str(out),source_manifest_sha256=sha(out/'source_manifest.json'),source_files=len(files),spec_sha256=sha(out/'spec.json'),launched=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
