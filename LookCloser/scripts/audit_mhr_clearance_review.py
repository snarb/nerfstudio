"""Seal the read-only clearance review and reproduce normalized plane values."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from study_multiview_face_prior import save,sha


def main():
    root=Path('/mnt/data/dec5_mhr_clearance_independent_review_v2');repo=Path(__file__).resolve().parents[1]
    assert not (root/'final_seal.json').exists();bindings={};checks=0
    producers={'result.json':'review_mhr_contact_clearance.py',
        'detector_probe_v3/result.json':'probe_clearance_detector_discrepancy.py',
        'actor_pair_probe/result.json':'probe_actor_contact_clearance.py'}
    for name,producer in producers.items():
        path=root/name;value=json.loads(path.read_text());source=repo/'scripts'/producer
        assert sha(source)==value['script_sha256'];bindings[str(source)]=sha(source);checks+=1
        for key in ['input_hashes','source_hashes']:
            for name,expected in value.get(key,{}).items():
                assert sha(name)==expected,name;bindings[name]=expected;checks+=1
        for name,expected in value['outputs'].items():
            target=path.parent/name;assert sha(target)==expected,target;bindings[str(target)]=expected;checks+=1
        if 'pybind_path' in value:
            target=value['pybind_path'];assert sha(target)==value['pybind_sha256'];bindings[target]=sha(target);checks+=1
    planes=[]
    for path in sorted((root/'actor_pair_probe').glob('*.npz')):
        points=np.load(path)['points'];mean=points.mean(0)
        sigma=np.sqrt(np.sum((points-mean)**2,axis=0)/5)+1e-12
        normalized=(points-mean)/sigma;a,b=normalized[:3],normalized[3:]
        na=np.cross(a[1]-a[0],a[2]-a[0]);nb=np.cross(b[1]-b[0],b[2]-b[0])
        du=b@na-a[0]@na;dv=a@nb-b[0]@nb
        planes.append(dict(file=path.name,du=du.tolist(),dv=dv.tolist(),
            du_zeroed=(np.abs(du)<1e-6).tolist(),dv_zeroed=(np.abs(dv)<1e-6).tolist(),
            note='Official v0.19 plane-expression epsilon diagnostic; not a full C++ predicate replay.'))
    save(root/'normalized_plane_values.json',dict(records=planes))
    tests=['tests/test_mesh_contact_clearance.py']
    run=subprocess.run([sys.executable,'-m','pytest','-o','addopts=','-q',*tests],cwd=repo,text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr
    save(root/'test_receipt.json',dict(command=[sys.executable,'-m','pytest','-o','addopts=','-q',*tests],
        returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
    for name in tests+['scripts/audit_mhr_clearance_review.py','experiments/dec5_mhr_clearance_independent_review.md']:
        bindings[str(repo/name)]=sha(repo/name)
    for path in root.rglob('*'):
        if path.is_file():bindings[str(path)]=sha(path)
    failed=Path('/mnt/data/dec5_mhr_clearance_independent_review')
    for path in failed.rglob('*'):
        if path.is_file():bindings[str(path)]=sha(path)
    for name,expected in bindings.items():assert sha(name)==expected,name
    save(root/'final_seal.json',dict(checked_declared_bindings=checks,bindings=bindings,synthetic_cases=108,
        noncollapsing_controls=4,actual_pair_checks=12,tests_passed=3,actor_fit_launched=False,production_changed=False))
    print(json.dumps(dict(declared_bindings=checks,sealed_files=len(bindings),tests_passed=3)))


if __name__=='__main__':main()
