"""Read-only synthetic adapter checks; never starts an actor fit."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import numpy as np
from scipy import sparse
import certified_conic_surface_step as adapter
from study_multiview_face_prior import save, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();args.output.mkdir();records=[]
    for path in sorted(args.input.glob('case*.npz')):
        data=np.load(path);x,receipt=adapter.solve(data['system'],data['rhs'],data['constraint'],
            data['lower'],data['offset'],.001)
        np.testing.assert_array_equal(x,data['x'])
        records.append(dict(path=str(path),sha256=sha(path),exact_original_solution=True,certificate=receipt['certificate']))
    try:
        adapter.certificate(sparse.eye(3),np.array([np.nan,0,0]),sparse.csc_matrix((4,3)),
            np.array([1.,0,0,0]),np.zeros(3),np.zeros(4),0)
    except ValueError as error:nan_check=str(error)
    else:raise AssertionError('Adapter accepted NaN stationarity input')
    original_solver=adapter.original.clarabel.DefaultSolver;control={};status_checks=[]
    class StatusProbe:
        def __init__(self,*params):self.solver=original_solver(*params)
        def solve(self):
            result=self.solver.solve();x=np.asarray(result.x).copy()
            if control.get('bad'):x[:]=0  # Feasible, but not stationary for the test objective.
            return SimpleNamespace(x=x,z=result.z,status=control['status'],iterations=result.iterations)
    adapter.original.clarabel.DefaultSolver=StatusProbe
    try:
        for status,bad,expected in [('Solved',False,True),('AlmostSolved',False,True),
                                   ('AlmostSolved',True,False),('MaxIterations',False,False)]:
            control.update(status=status,bad=bad)
            try:
                _,receipt=adapter.solve(np.eye(3),[.002,.003,.001],sparse.csc_matrix((0,3)),[],np.zeros((1,3)),.001)
                accepted=True;detail=receipt['certificate']
            except ValueError as error:accepted=False;detail=str(error)
            assert accepted==expected,(status,bad,detail)
            status_checks.append(dict(injected_status=status,corrupted_primal=bad,accepted=accepted,detail=detail))
    finally:adapter.original.clarabel.DefaultSolver=original_solver
    files=[Path(adapter.__file__),Path(adapter.__file__).with_name('run_mhr_certified_conic_correction.py')]
    for path in files:(args.output/path.name).write_bytes(path.read_bytes())
    save(args.output/'result.json',dict(replays=records,nan_rejection=nan_check,status_checks=status_checks,
        adapter_provenance=adapter.provenance(),source_hashes={str(p):sha(p) for p in files},
        script_sha256=sha(__file__),actor_fit_launched=False,
        outputs={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps(dict(exact_replays=len(records),nan_rejection=nan_check,status_checks=status_checks)),flush=True)


if __name__=='__main__':main()
