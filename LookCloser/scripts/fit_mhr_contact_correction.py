"""Constrained warm-start correction with dynamically detected contacts."""
from pathlib import Path
import numpy as np
from scipy import sparse
import fit_mhr_guarded_correction as parent
import fit_mhr_constrained_correction as qp
from constrained_surface_step import lower_bound_lsq,oriented_area_constraints
from mesh_contact_constraints import contact_constraints
from study_multiview_face_prior import save,sha

ROOT=Path('/mnt/data/dec5_mhr_contact_correction_v2')


def optimizer(guard):
    worker,stats=qp.optimizer(guard);contacts=set()
    def solve(system,rhs,current,ids):
        area,lower,facets=oriented_area_constraints(current,guard.t,guard.original_cross,
            guard.original_good,ids,guard.floor)
        inner=[]
        for attempt in range(9):
            contact,b,records=contact_constraints(current,guard.t,contacts,ids)
            a=sparse.vstack((area,contact),format='csr');bounds=np.r_[lower,b]
            x,record=lower_bound_lsq(system,rhs,a,bounds)
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
                maximum_constraint_generation_rounds=9,signed_area_floor_fraction=.01,
                current_contact_planes_not_target_constraints=True,discrete_guards_unchanged=True))
        if Path(path).name=='result.json':
            value=dict(value,qp_progress_sha256=sha(ROOT/'qp_progress.json'),
                qp_source_sha256=sha(ROOT/'optimizer_qp_source.json'),
                contact_progress_sha256=sha(ROOT/'contact_progress.json'))
        original_save(path,value)
    qp.ROOT=ROOT;parent.ROOT=ROOT;parent.StepGuard=qp.ConstraintGuard
    parent.optimizer=optimizer;parent.save=annotated
    parent.main()
