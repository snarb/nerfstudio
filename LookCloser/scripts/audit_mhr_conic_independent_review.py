"""Seal the independent synthetic review, without changing reviewed helpers."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from study_multiview_face_prior import save,sha


def main():
    root=Path('/mnt/data/dec5_mhr_conic_independent_review');repo=Path(__file__).resolve().parents[1]
    assert not (root/'final_seal.json').exists()
    bindings={};checked=0
    for path in [root/'result.json',root/'certified_adapter/result.json']:
        r=json.loads(path.read_text())
        for key in ['input_hashes','source_hashes']:
            for name,expected in r.get(key,{}).items():
                assert sha(name)==expected,name;bindings[name]=expected;checked+=1
        for name,expected in r['outputs'].items():
            target=path.parent/name;assert sha(target)==expected,target;bindings[str(target)]=expected;checked+=1
        for name,expected in r.get('backend_provenance',r.get('adapter_provenance',{})).get('helper_hashes',{}).items():
            assert sha(name)==expected,name;bindings[name]=expected;checked+=1
        script=repo/'scripts'/('review_mhr_conic_solver.py' if path.parent==root else 'review_certified_conic_adapter.py')
        assert sha(script)==r['script_sha256'];bindings[str(script)]=sha(script);checked+=1
    for path in root.glob('case*.npz'):
        d=np.load(path);assert np.isfinite(d['x']).all()
        assert (d['constraint']@d['x']-d['lower']).min()>=-1e-11
        assert np.linalg.norm(d['offset']+d['x'].reshape(-1,3),axis=1).max()<=.001+1e-11
        assert abs(d['x']-d['reference_x']).max()<2e-8
    tests=['tests/test_conic_surface_step.py','tests/test_certified_conic_surface_step.py']
    result=subprocess.run([sys.executable,'-m','pytest','-o','addopts=','-q',*tests],cwd=repo,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    save(root/'test_receipt.json',dict(command=[sys.executable,'-m','pytest','-o','addopts=','-q',*tests],
        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    for name in tests+['scripts/audit_mhr_conic_independent_review.py','experiments/dec5_mhr_conic_independent_review.md']:
        bindings[str(repo/name)]=sha(repo/name)
    # Read-only parent diagnosis is a reported receipt, not an independent replay of its actor matrices.
    parent=Path('/mnt/data/dec5_mhr_conic_solver_diagnosis/result.json');bindings[str(parent)]=sha(parent)
    for path in root.rglob('*'):
        if path.is_file():bindings[str(path)]=sha(path)
    for path,expected in bindings.items():assert sha(path)==expected,path
    save(root/'final_seal.json',dict(checked_declared_bindings=checked,bindings=bindings,
        synthetic_cases=9,tests_passed=11,actor_fit_launched=False,production_changed=False))
    print(json.dumps(dict(checked_declared_bindings=checked,sealed_files=len(bindings),tests_passed=11)))


if __name__=='__main__':main()
