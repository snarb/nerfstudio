"""Solve the unchanged per-vertex displacement bound with contact constraints.

Supporting planes are generated only from proposed geometry, never target
pixels. Exact nonlinear guards remain the final acceptance condition.
"""
from pathlib import Path
import inspect
import numpy as np
from scipy import sparse
import fit_mhr_guarded_correction as parent
import fit_mhr_constrained_correction as qp
import fit_mhr_contact_correction as contact
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_bounded_contact_correction')


def ball_planes(current,base,ids,planes,limit):
    rows=[];cols=[];values=[];lower=[]
    for ri,(index,normal) in enumerate(planes):
        rows.extend([ri]*3);cols.extend((3*index+np.arange(3)).tolist());values.extend((-normal).tolist())
        lower.append(float(normal@(current[ids[index]]-base[ids[index]])-limit))
    a=sparse.coo_matrix((values,(rows,cols)),shape=(len(lower),len(ids)*3)).tocsr()
    return a,np.asarray(lower)


def optimizer(guard):
    source=inspect.getsource(contact.optimizer)
    replacements={
        'contacts=set()':'contacts=set();balls=[]',
        "a=sparse.vstack((area,contact),format='csr');bounds=np.r_[lower,b]":
            "ball,bl=ball_planes(current,guard.base,ids,balls,parent.SETTINGS['maximum_displacement'])\n"
            "            a=sparse.vstack((area,contact,ball),format='csr');bounds=np.r_[lower,b,bl]",
        'new=parent.strict_pairs(trial,guard.t)-guard.allowed_pairs':
            "new=parent.strict_pairs(trial,guard.t)-guard.allowed_pairs\n"
            "            delta=trial[ids]-guard.base[ids];length=np.linalg.norm(delta,axis=1)\n"
            "            violating=np.flatnonzero(length>parent.SETTINGS['maximum_displacement']+1e-12)\n"
            "            for index in violating:balls.append((int(index),delta[index]/length[index]))",
        'if not new or new.issubset(contacts):break':
            'if (not new or new.issubset(contacts)) and not len(violating):break',
        'remaining_new_pairs=[list(map(int,p)) for p in sorted(new)]':
            'remaining_new_pairs=[list(map(int,p)) for p in sorted(new)],ball_planes=len(balls),remaining_ball_violations=len(violating)',
    }
    generated=source
    for old,new in replacements.items():
        assert generated.count(old)==1,old
        generated=generated.replace(old,new)
    ns=dict(contact.__dict__,ROOT=ROOT,ball_planes=ball_planes)
    exec(compile(generated,'<bounded_contact_solver>','exec'),ns)
    save(ROOT/'bounded_optimizer_source.json',dict(original_source=source,generated_source=generated,
        replacements=replacements,ball_limit_unchanged=parent.SETTINGS['maximum_displacement']))
    return ns['optimizer'](guard)


if __name__=='__main__':
    original_save=parent.save
    def annotated(path,value):
        if Path(path).name=='protocol.json':
            helpers=[Path(__file__),Path(qp.__file__),Path(contact.__file__),
                Path(__file__).with_name('mesh_contact_constraints.py'),Path(__file__).with_name('constrained_surface_step.py')]
            value=dict(value,bounded_contact_correction=dict(helper_hashes={str(p):sha(p) for p in helpers},
                maximum_constraint_generation_rounds=9,signed_area_floor_fraction=.01,
                maximum_displacement_unchanged=.001,discrete_guards_unchanged=True))
        if Path(path).name=='result.json':
            value=dict(value,solver_outputs={str(p):sha(p) for p in [ROOT/'qp_progress.json',
                ROOT/'optimizer_qp_source.json',ROOT/'contact_progress.json',ROOT/'bounded_optimizer_source.json']} )
        original_save(path,value)
    qp.ROOT=ROOT;parent.ROOT=ROOT;parent.StepGuard=qp.ConstraintGuard
    parent.optimizer=optimizer;parent.save=annotated
    parent.main()
