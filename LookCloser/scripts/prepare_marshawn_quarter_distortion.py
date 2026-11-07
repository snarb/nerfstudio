"""Freeze quarter-step × distortion.1, without adding the frequency focus hook."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def prepare(parent,out):
    out.mkdir(exist_ok=False)
    for name in ['code','source','inherited_proofs']:
        shutil.copytree(parent/name,out/name,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    worker=out/'code/LookCloser/scripts/train_marshawn_semantic_admission.py';text=worker.read_text();anchor='coarse_step=.0015625, distortion_final=.01, seed=42,'
    if text.count(anchor)!=1:raise ValueError('Exact quarter validator anchor changed')
    worker.unlink();worker.write_text(text.replace(anchor,anchor.replace('distortion_final=.01','distortion_final=.1')))
    prior=json.loads((parent/'source_manifest.json').read_text())['files_sha256']
    files={str(p.relative_to(out/'code')):sha(p)for p in(out/'code').rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'}
    changed=[p for p in prior if prior[p]!=files[p]]
    if set(files)!=set(prior)or len(files)!=1495 or changed!=['LookCloser/scripts/train_marshawn_semantic_admission.py']:raise ValueError('Unexpected source delta')
    write(out/'source_manifest.json',dict(files_sha256=files,parent_source_manifest_sha256=sha(parent/'source_manifest.json'),changed_from_quarter=changed,change='Only exact distortion_final validator .01→.1; original uniformfrequency update/model/sampler unchanged'))
    spec=json.loads((parent/'spec.json').read_text());oldroot=str(Path(spec['output']).parent);newroot=str(Path(oldroot).with_name(out.name))
    if 'frequency_focus'in spec:raise ValueError('Focused proposal must not be bundled')
    spec=json.loads(json.dumps(spec).replace(oldroot,newroot));spec['distortion_final']=.1;write(out/'spec.json',spec)
    plan=json.loads(json.dumps(json.loads((parent/'controller_plan.json').read_text())).replace(oldroot,newroot).replace('exp_actor_semantic_W64pad1_quarterstep_r0','exp_actor_semantic_quarterstep_distortion01_r0'));write(out/'controller_plan.json',plan)
    for label,start in [('warmup',0),('ARM',4096)]:write(out/f'technical_{label}.json',dict(spec,output=newroot+'/technical_'+label,technical_start_step=start,technical_pause_after=start+15,technical_memory_cap_gib=10.))
    write(out/'candidate_policy.json',dict(kind='predeclared_2x2_interaction',factors=['coarse_step','distortion_final'],reference_quarter_spec_sha256=sha(parent/'spec.json'),only_change_from_quarter='distortion_final .01→.1; original4096..12288ramp',unchanged=['quarter.0015625','uniformfrequency2048','originalmaps','W64/R0/body80','legacyprojection','legacyallocator1024cap','LR'],focused_hook=False,minimum_trajectory_step=106316,stopping='unchanged validation plateau',geometry_quality_approved=False,authorization='Root delegated scopedtechnical/resume then coldidleexclusive launch'))
    print(json.dumps(dict(root=str(out),source_sha256=sha(out/'source_manifest.json'),spec_sha256=sha(out/'spec.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.parent,a.output)
