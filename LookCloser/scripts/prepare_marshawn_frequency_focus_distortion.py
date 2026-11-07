"""Freeze the declared focus × distortion interaction, without launching jobs."""
import argparse, hashlib, json, os, shutil
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value): p.write_text(json.dumps(value, indent=2)+'\n')

def prepare(parent, output):
    output.mkdir(exist_ok=False)
    for name in ['code', 'source', 'inherited_proofs']:
        shutil.copytree(parent/name, output/name, copy_function=os.link,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    worker=output/'code/LookCloser/scripts/train_marshawn_semantic_admission.py'
    text=worker.read_text(); old='coarse_step=.00625, distortion_final=.01, seed=42,'
    if text.count(old)!=1: raise ValueError('Exact validator anchor changed')
    worker.unlink(); worker.write_text(text.replace(old, old.replace('distortion_final=.01','distortion_final=.1')))
    previous=json.loads((parent/'source_manifest.json').read_text())['files_sha256']
    files={str(p.relative_to(output/'code')):sha(p) for p in (output/'code').rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    changed=[p for p in previous if previous[p]!=files[p]]
    if set(files)!=set(previous) or len(files)!=1497 or changed!=['LookCloser/scripts/train_marshawn_semantic_admission.py']:
        raise ValueError('Unexpected source delta')
    write(output/'source_manifest.json', dict(files_sha256=files, parent_source_manifest_sha256=sha(parent/'source_manifest.json'),
          changed_from_focus=changed, change='Only exact distortion_final validator .01→.1; all focus runtime and model bytes unchanged'))
    spec=json.loads((parent/'spec.json').read_text()); oldroot=str(Path(spec['output']).parent)
    newroot=str(Path(oldroot).with_name(output.name))
    spec=json.loads(json.dumps(spec).replace(oldroot,newroot)); spec['distortion_final']=.1
    write(output/'spec.json',spec)
    plan=json.loads(json.dumps(json.loads((parent/'controller_plan.json').read_text())).replace(oldroot,newroot).replace('exp_actor_semantic_frequencyfocus_r0','exp_actor_semantic_frequencyfocus_distortion01_r0'))
    write(output/'controller_plan.json',plan)
    for label,start in [('warmup',1023),('ARM',4096)]:
        write(output/f'technical_{label}.json',dict(spec,output=newroot+'/technical_'+label,technical_start_step=start,technical_pause_after=start+31,technical_memory_cap_gib=10.))
    write(output/'candidate_policy.json',dict(kind='predeclared_2x2_interaction',factors=['frequency_focus','distortion_final'],
          reference_focus_spec_sha256=sha(parent/'spec.json'),only_change_from_focus='distortion_final .01→.1; original ramp4096..12288',
          unchanged=['coarse .00625','native R0','W64 confidence/admission','body80','maps','legacy projection','legacy allocator/cap1024','LR','private frequency RNG/state'],
          minimum_trajectory_step=106316,stopping='unchanged validation plateau',geometry_quality_approved=False,
          authorization='Root delegated technical/resume+frequency-boundary proof, then cold production on idle locked GPU'))
    print(json.dumps(dict(root=str(output),source_sha256=sha(output/'source_manifest.json'),spec_sha256=sha(output/'spec.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.parent,a.output)
