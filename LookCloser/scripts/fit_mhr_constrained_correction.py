"""Warm-start correction with signed-area inequalities inside the LS solve.

Only the proposal solver changes relative to guarded correction. The exact
discrete checks and displacement/area bounds remain in force.
"""
from pathlib import Path
import numpy as np
import fit_mhr_guarded_correction as parent
from constrained_surface_step import lower_bound_lsq,oriented_area_constraints
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_constrained_correction')
BaseGuard=parent.StepGuard
original_optimizer=parent.optimizer


class ConstraintGuard(BaseGuard):
    def __init__(self,*args):
        super().__init__(*args)
        n=self.original_cross/np.maximum(np.linalg.norm(self.original_cross,axis=1,keepdims=True),1e-30)
        self.reference_unit=n
        self.floor=np.sum(self.cross*n,axis=1)*.01
        self.solves=[]

    def check(self,trial):
        signed=np.sum(parent.crosses(trial,self.t)*self.reference_unit,axis=1)
        if (signed[self.original_good]<self.floor[self.original_good]-1e-16).any():
            return False,dict(reason='signed_area_floor')
        return super().check(trial)


def optimizer(guard):
    worker,stats=original_optimizer(guard)
    source=read(ROOT/'optimizer_source.json')['generated_source']
    old="solution = lsmr(system, rhs, atol=1e-8, btol=1e-8, maxiter=RECIPE['maximum_lsmr_iterations'])"
    assert source.count(old)==1
    source=source.replace(old,'solution = constrained_solve(system, rhs, current, ids)')
    def solve(system,rhs,current,ids):
        a,b,facets=oriented_area_constraints(current,guard.t,guard.original_cross,
            guard.original_good,ids,guard.floor)
        x,record=lower_bound_lsq(system,rhs,a,b)
        record.update(constraint_count=len(b),constrained_faces=facets[record['active']].tolist())
        guard.solves.append(record);save(ROOT/'qp_progress.json',dict(solves=guard.solves))
        return x,1,record['iterations']
    namespace=dict(worker.__globals__,constrained_solve=solve)
    exec(compile(source,'<signed_area_constrained_correction>','exec'),namespace)
    save(ROOT/'optimizer_qp_source.json',dict(generated_source=source,signed_area_floor_fraction=.01,
        qp_helper_sha256=sha(Path(__file__).with_name('constrained_surface_step.py'))))
    return namespace['optimize'],stats


if __name__=='__main__':
    original_save=parent.save
    def annotated(path,value):
        if Path(path).name=='protocol.json':
            value=dict(value,constrained_correction=dict(wrapper_sha256=sha(__file__),
                solver_sha256=sha(Path(__file__).with_name('constrained_surface_step.py')),
                signed_area_floor_fraction=.01,trust_and_discrete_guards_unchanged=True))
        if Path(path).name=='result.json':
            value=dict(value,qp_progress_sha256=sha(ROOT/'qp_progress.json'),
                qp_source_sha256=sha(ROOT/'optimizer_qp_source.json'))
        original_save(path,value)
    parent.ROOT=ROOT;parent.StepGuard=ConstraintGuard;parent.optimizer=optimizer;parent.save=annotated
    parent.main()
