"""Train-only correction with exact per-vertex ball cones, not tangent planes."""
from pathlib import Path
import numpy as np
from scipy import sparse
import fit_mhr_guarded_correction as parent
import fit_mhr_constrained_correction as qp
from constrained_surface_step import oriented_area_constraints
from mesh_contact_constraints import contact_constraints
import conic_surface_step as backend
from study_multiview_face_prior import save,sha

ROOT=Path('/mnt/data/dec5_mhr_conic_correction')


def optimizer(guard):
    worker,stats=qp.optimizer(guard);contacts=set()
    def solve(system,rhs,current,ids):
        area,lower,facets=oriented_area_constraints(current,guard.t,guard.original_cross,
            guard.original_good,ids,guard.floor)
        inner=[]
        for attempt in range(9):
            contact,b,records=contact_constraints(current,guard.t,contacts,ids)
            a=sparse.vstack((area,contact),format='csr');bounds=np.r_[lower,b]
            try:
                x,record=backend.solve(system,rhs,a,bounds,current[ids]-guard.base[ids],
                    parent.SETTINGS['maximum_displacement'])
            except Exception as error:
                failure=ROOT/'solver_failure';failure.mkdir(exist_ok=False)
                sparse.save_npz(failure/'system.npz',system);sparse.save_npz(failure/'constraint.npz',a)
                np.savez_compressed(failure/'arrays.npz',rhs=rhs,lower=bounds,
                    offset=current[ids]-guard.base[ids],current=current,ids=ids)
                save(failure/'result.json',dict(error=repr(error),completed_solves=len(guard.solves),
                    inner_attempt=attempt,production_accepted=False))
                raise
            step=x.reshape(-1,3);maximum=np.linalg.norm(step,axis=1).max()
            factor=min(1.,parent.SETTINGS['maximum_step']/max(maximum,1e-20))
            trial=current.copy();trial[ids]+=step*factor
            new=parent.strict_pairs(trial,guard.t)-guard.allowed_pairs
            inner.append(dict(attempt=attempt,qp=record,contact_rows=records,
                remaining_new_pairs=[list(map(int,p)) for p in sorted(new)]))
            save(ROOT/'contact_progress.json',dict(completed_solves=guard.solves,current=inner))
            if not new or new.issubset(contacts):break
            contacts.update(new)
        record=dict(record,inner=inner,contact_pairs=[list(map(int,p)) for p in sorted(contacts)],
            area_constraints=len(lower),constrained_area_faces=facets[[i for i in record['active'] if i<len(facets)]].tolist())
        guard.solves.append(record);save(ROOT/'qp_progress.json',dict(solves=guard.solves))
        return x,1,record['iterations']
    worker.__globals__['constrained_solve']=solve
    return worker,stats


if __name__=='__main__':
    original_save=parent.save
    def annotated(path,value):
        if Path(path).name=='protocol.json':
            helpers=[Path(__file__),Path(qp.__file__),Path(__file__).with_name('mesh_contact_constraints.py'),
                Path(__file__).with_name('constrained_surface_step.py')]
            value=dict(value,contact_correction=dict(helper_hashes={str(p):sha(p) for p in helpers},
                qp_backend=backend.provenance(),maximum_constraint_generation_rounds=9,
                signed_area_floor_fraction=.01,maximum_displacement_unchanged=.001,
                exact_ball_cones_replace_outer_approximation=True,discrete_guards_unchanged=True))
        if Path(path).name=='result.json':
            value=dict(value,solver_outputs={str(p):sha(p) for p in [ROOT/'qp_progress.json',
                ROOT/'optimizer_qp_source.json',ROOT/'contact_progress.json']})
        original_save(path,value)
    qp.ROOT=ROOT;parent.ROOT=ROOT;parent.StepGuard=qp.ConstraintGuard
    parent.optimizer=optimizer;parent.save=annotated
    parent.main()
