"""Replay the exact failed dense8 cone problem; never relax acceptance gates."""
from pathlib import Path
from types import FunctionType
import time
import numpy as np
from scipy import sparse
import certified_conic_surface_step as backend
from study_multiview_face_prior import read, save, sha

SOURCE = Path('/mnt/data/dec5_mhr_sampling_dense8/solver_failure')
OUT = Path('/mnt/data/dec5_mhr_dense8_solver_precision')
CONTROLS = dict(
    baseline={},
    tighter_tolerances=dict(tol_feas=1e-12,tol_gap_abs=1e-12,tol_gap_rel=1e-12,max_iter=300),
    smaller_termination_step=dict(min_terminate_step_length=1e-8),
    weaker_kkt_regularization=dict(static_regularization_constant=1e-10),
    no_equilibration=dict(equilibrate_enable=False),
    tighter_refinement=dict(iterative_refinement_reltol=1e-15,iterative_refinement_abstol=1e-14,
                            iterative_refinement_max_iter=30))


def main():
    assert not OUT.exists()
    m=sparse.load_npz(SOURCE/'system.npz'); a=sparse.load_npz(SOURCE/'constraint.npz')
    data=np.load(SOURCE/'arrays.npz'); OUT.mkdir()
    inputs=[SOURCE/p for p in ['system.npz','constraint.npz','arrays.npz','result.json']]
    save(OUT/'request.json',dict(inputs={str(p):sha(p) for p in inputs}, controls=CONTROLS,
        backend_provenance=backend.provenance(), script_sha256=sha(__file__),
        fixed_problem_and_radius=True, certificate_tolerances_unchanged=True, actor_fit_not_rerun=True))
    records=[]
    for name,overrides in CONTROLS.items():
        settings=dict(backend.solve.__globals__['SETTINGS'],**overrides)
        solve=FunctionType(backend.solve.__code__,dict(backend.solve.__globals__,SETTINGS=settings))
        start=time.monotonic()
        try:
            x,r=solve(m,data['rhs'],a,data['lower'],data['offset'],.001)
            np.savez_compressed(OUT/(name+'.npz'),x=x)
            record=dict(name=name,status='accepted_by_unchanged_certificate',solver=r,
                solution_sha256=sha(OUT/(name+'.npz')))
        except ValueError as error:
            record=dict(name=name,status='rejected',error=str(error))
        record.update(seconds=time.monotonic()-start,settings=settings); records.append(record)
        save(OUT/'progress.json',dict(records=records))
        print(name,record['status'],record.get('error',record.get('solver',{}).get('certificate')),flush=True)
    save(OUT/'result.json',dict(request_sha256=sha(OUT/'request.json'),records=records,
        no_acceptance_relaxation=True,no_actor_or_production_mesh_modified=True))


if __name__=='__main__': main()
