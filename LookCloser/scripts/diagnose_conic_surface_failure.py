"""Replay a retained failed cone subproblem; never returns geometry to fitting."""
from pathlib import Path
import inspect
import numpy as np
from scipy import sparse
import conic_surface_step as backend
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_conic_solver_diagnosis')
SOURCE=Path('/mnt/data/dec5_mhr_conic_correction/solver_failure')


def main():
    assert not ROOT.exists();ROOT.mkdir()
    source=inspect.getsource(backend.solve)
    old="if str(result.status)!='Solved':raise ValueError(f'Clarabel did not solve: {result.status}')"
    assert source.count(old)==1
    replacement="capture(result)"
    def capture(result):
        save(ROOT/'backend.json',dict(status=str(result.status),iterations=result.iterations,
            r_prim=result.r_prim,r_dual=result.r_dual,obj_val=result.obj_val,obj_val_dual=result.obj_val_dual))
    generated=source.replace(old,replacement)
    ns=dict(backend.__dict__,capture=capture)
    exec(compile(generated,'<cone_diagnostic_status_bypass_not_fit>','exec'),ns)
    inputs={str(p):sha(p) for p in SOURCE.iterdir() if p.is_file()}
    save(ROOT/'request.json',dict(input_hashes=inputs,backend=backend.provenance(),
        generated_source=generated,script_sha256=sha(__file__),diagnostic_not_fit=True))
    arrays=np.load(SOURCE/'arrays.npz')
    try:
        x,record=ns['solve'](sparse.load_npz(SOURCE/'system.npz'),arrays['rhs'],
            sparse.load_npz(SOURCE/'constraint.npz'),arrays['lower'],arrays['offset'],.001)
        np.save(ROOT/'solution.npy',x)
        result=dict(independent_certificate_passed=True,record=record)
    except Exception as error:
        result=dict(independent_certificate_passed=False,error=repr(error))
    save(ROOT/'result.json',dict(result,production_accepted=False,request_sha256=sha(ROOT/'request.json')))
    print(result,flush=True)


if __name__=='__main__':main()
