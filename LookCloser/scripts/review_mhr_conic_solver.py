"""Independent coupled-vertex checks of the optional exact-ball solver.

Synthetic optimization only. No actor data or production inputs are read.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
import conic_surface_step as backend
from study_multiview_face_prior import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output
    assert not root.exists();root.mkdir();rng=np.random.default_rng(416193);records=[]
    captured={};original_solver=backend.clarabel.DefaultSolver
    class Spy:
        def __init__(self,*args):
            captured['arguments']=args;self.solver=original_solver(*args)
        def solve(self):
            result=self.solver.solve();captured['result']=result;return result
    backend.clarabel.DefaultSolver=Spy
    try:
        for case,count in enumerate([2,3,3]):
            n=count*3;m=np.r_[rng.normal(size=(n+5,n)),np.eye(n)*.3]
            wanted=rng.normal(size=(count,3));wanted*=.002/np.linalg.norm(wanted,axis=1)[:,None]
            rhs=m@wanted.ravel();offset=rng.normal(size=(count,3));offset*=.0004/np.linalg.norm(offset,axis=1)[:,None]
            linear=rng.normal(size=(count+2,n));feasible=np.zeros(n)
            if case==2:feasible[0]=.00015;feasible[3]=-.00015
            lower=linear@feasible-rng.uniform(.00003,.00015,len(linear))
            if case==2:linear[0]=0;linear[0,0]=1;linear[0,3]=-1;lower[0]=.0002
            unit=.001;radius=.001;h=m.T@m;g=m.T@rhs;scale=h.diagonal().max()
            def objective(y):return .5*np.sum((m@y-rhs/unit)**2)/scale
            def gradient(y):return m.T@(m@y-rhs/unit)/scale
            def ball(y):return np.ones(count)-np.sum((offset/unit+y.reshape(count,3))**2,axis=1)
            def ball_jac(y):
                result=np.zeros((count,n));value=-2*(offset/unit+y.reshape(count,3))
                for i in range(count):result[i,3*i:3*i+3]=value[i]
                return result
            reference=minimize(objective,feasible/unit,jac=gradient,method='SLSQP',
                constraints=[dict(type='ineq',fun=lambda y:linear@y-lower/unit,jac=lambda y:linear),
                             dict(type='ineq',fun=ball,jac=ball_jac)],options=dict(ftol=1e-13,maxiter=2000))
            assert reference.success,reference.message
            for factor in [.01,1.,100.]:
                mm=m*factor;rr=rhs*factor;x,receipt=backend.solve(mm,rr,linear,lower,offset,radius)
                hh=mm.T@mm;gg=mm.T@rr;ss=hh.diagonal().max();args=captured['arguments'];raw=captured['result']
                np.testing.assert_allclose(args[0].toarray(),np.triu(hh/ss),atol=1e-15)
                np.testing.assert_allclose(args[1],-gg/(ss*unit),atol=1e-15)
                aa=args[2].toarray();bb=args[3];k=len(lower)
                np.testing.assert_array_equal(aa[:k],-linear);np.testing.assert_array_equal(bb[:k],-lower/unit)
                expected=np.zeros((4*count,n))
                for i in range(count):expected[4*i+1:4*i+4,3*i:3*i+3]=-np.eye(3)
                np.testing.assert_array_equal(aa[k:],expected)
                np.testing.assert_array_equal(bb[k:].reshape(count,4),np.c_[np.ones(count),offset/unit])
                y=np.asarray(raw.x);z=np.asarray(raw.z);slack=bb-aa@y
                cone_z=z[k:].reshape(count,4);cone_s=slack[k:].reshape(count,4)
                lam=z[:k]*ss*unit;cone_gradient=-cone_z[:,1:].ravel()*ss*unit
                station=hh@x-gg-linear.T@lam+cone_gradient
                record=dict(case=case,vertices=count,system_scale=factor,slsqp_success=bool(reference.success),
                    maximum_x_difference=float(abs(x-unit*reference.x).max()),
                    scaled_objective_difference=float(abs(objective(x/unit)-reference.fun)),
                    original_linear_minimum_slack=float((linear@x-lower).min()),
                    maximum_displacement=float(np.linalg.norm(offset+x.reshape(count,3),axis=1).max()),
                    original_stationarity_divided_by_scale_unit=float(abs(station).max()/(ss*unit)),
                    cone_complementarity_max=float(abs(np.sum(cone_s*cone_z,axis=1)).max()),certificate=receipt['certificate'])
                assert record['maximum_x_difference']<2e-8 and record['scaled_objective_difference']<1e-8
                assert record['original_stationarity_divided_by_scale_unit']<=backend.TOLERANCES['stationarity']
                records.append(record)
                np.savez_compressed(root/f'case{case}_scale{factor:g}.npz',system=mm,rhs=rr,constraint=linear,lower=lower,
                    offset=offset,x=x,reference_x=unit*reference.x,dual=z,cone_slack=slack)
    finally:backend.clarabel.DefaultSolver=original_solver
    probes={}
    cases={'noncomplementary_soc':(np.zeros(3),np.array([1.,0,0,0])),
           'nonfinite_stationarity_input':(np.array([np.nan,0,0]),np.zeros(4))}
    for name,(q,z) in cases.items():
        try:
            value=backend.certificate(sparse.eye(3),q,sparse.csc_matrix((4,3)),np.array([1.,0,0,0]),np.zeros(3),z,0)
            probes[name]=dict(accepted=True,stationarity_finite=bool(np.isfinite(value['stationarity'])))
        except ValueError as error:probes[name]=dict(accepted=False,error=str(error))
    source=Path(backend.__file__);wrapper=source.with_name('fit_mhr_conic_correction.py')
    (root/'conic_surface_step_reviewed.py').write_bytes(source.read_bytes())
    (root/'fit_mhr_conic_correction_reviewed.py').write_bytes(wrapper.read_bytes())
    save(root/'result.json',dict(cases=records,certificate_probes=probes,
        backend_provenance=backend.provenance(),input_hashes={str(source):sha(source),str(wrapper):sha(wrapper)},
        script_sha256=sha(__file__),new_dec5_fit=False,source_or_input_mutation=False,
        outputs={f.name:sha(f) for f in sorted(root.iterdir()) if f.is_file()}))
    print('coupled cases',len(records),'max x difference',max(r['maximum_x_difference'] for r in records),
          'max objective difference',max(r['scaled_objective_difference'] for r in records),'certificate probes',probes,flush=True)


if __name__=='__main__':main()
