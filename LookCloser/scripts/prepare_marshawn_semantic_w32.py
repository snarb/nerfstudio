"""Prepare, but never launch, the exact W32 loss-margin semantic control."""
import argparse,hashlib,json,os,shutil
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(parent,out):
    out.mkdir(parents=True,exist_ok=False)
    # Copy immutable files with independent inodes for the sole edited worker.
    shutil.copytree(parent/'code',out/'code',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(parent/'source',out/'source',copy_function=os.link)
    worker=out/'code/LookCloser/scripts/train_marshawn_semantic_admission.py';text=worker.read_text()
    original="reference = dict(objective='full', sampling='body80', width=64, radius=0,"
    replacement="reference = dict(objective='full', sampling='body80', width=32, radius=0,"
    if text.count(original)!=1:raise ValueError('R0 exact validation anchor changed')
    worker.unlink();worker.write_text(text.replace(original,replacement))
    meta=json.loads((parent/'source_manifest.json').read_text());files={str(f.relative_to(out/'code')):sha(f)for f in (out/'code').rglob('*')if f.is_file()and '__pycache__'not in f.parts and f.suffix!='.pyc'}
    changed=[n for n,h in meta['files_sha256'].items()if files[n]!=h]
    if changed!=['LookCloser/scripts/train_marshawn_semantic_admission.py']:raise ValueError('Unexpected source diff')
    (out/'source_manifest.json').write_text(json.dumps(dict(parent_source_manifest_sha256=sha(parent/'source_manifest.json'),files_sha256=files,changed_from_R0=changed,change='W32 spec validation only; model and proposal sampling unchanged'),indent=2)+'\n')
    spec=json.loads((parent/'spec.json').read_text());before=dict(spec);spec['width']=32
    old_remote=str(Path(spec['output']).parent);new_remote=str(Path(old_remote).with_name(out.name))
    spec=json.loads(json.dumps(spec).replace(old_remote,new_remote));(out/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    plan=json.loads((parent/'controller_plan.json').read_text());plan=json.loads(json.dumps(plan).replace(old_remote,new_remote).replace('exp_actor_semantic_W64pad1_r0','exp_actor_semantic_W32loss_W64pad1_r0'))
    (out/'controller_plan.pending.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'preparation.json').write_text(json.dumps(dict(prepared=True,production_started=False,production_release=False,requires='Separate exact root review; prepared only, no queue or training',parent_spec_sha256=sha(parent/'spec.json'),spec_sha256=sha(out/'spec.json'),unchanged_fixed_support_manifest_sha256=spec['semantic_admission']['manifest_sha256'],only_recipe_parameter_change={'actor_loss_width':[64,32]},fixed_W64_semantic_pad1_unchanged=True,body80_proposal_pools_and_FAS_unchanged=True,loss_normalizer_recomputed_for_W32=True),indent=2)+'\n')
    print(json.dumps(dict(root=str(out),source_manifest_sha256=sha(out/'source_manifest.json'),spec_sha256=sha(out/'spec.json'),launched=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.parent,a.output)
