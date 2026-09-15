"""Independent synthetic positive-clearance review, without actor input."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from mesh_contact_clearance import contact_constraints,CLEARANCE
from guard_mhr_anatomical_correction import all_pairs
from study_multiview_face_prior import save,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    root=args.output;assert not root.exists();root.mkdir();rng=np.random.default_rng(8841193);records=[]
    for mode in ['coplanar','parallel','skew']:
        for trial in range(12):
            a=np.array([[0.,0,0],[1,0,0],[0,1,0]])*.0002
            b=a+([0,0,.0002] if mode=='parallel' else [.0004,0,0])
            if mode=='skew':b=(b-b.mean(0))@Rotation.from_rotvec([.6,0,0]).as_matrix().T+b.mean(0)
            rotation=Rotation.random(random_state=rng).as_matrix();translation=rng.normal(size=3)*.2
            v=np.r_[a,b]@rotation.T+translation;t=np.array([[0,1,2],[3,4,5]])
            if trial%2:t=t[:,::-1]
            for ids in [np.array([3,4,5]),np.array([1,3,4,5]),np.arange(6)]:
                matrix,lower,rows=contact_constraints(v,t,{(0,1)},ids);axis=np.array(rows[0]['axis'])
                np.testing.assert_allclose(np.linalg.norm(axis),1,atol=1e-14)
                np.testing.assert_allclose(np.sqrt(matrix.multiply(matrix).sum(1)).A.ravel(),1,atol=1e-14)
                desired=np.zeros((6,3));desired[3:]=a.mean(0)-b.mean(0);desired=desired@rotation.T
                wanted=desired[ids].ravel()/.001
                solution=minimize(lambda y:.5*np.sum((y-wanted)**2),np.zeros(len(wanted)),jac=lambda y:y-wanted,
                    constraints=[dict(type='ineq',fun=lambda y:matrix@y-lower/.001,jac=lambda y:matrix.toarray())],
                    method='SLSQP',options=dict(ftol=1e-13,maxiter=1000))
                assert solution.success,solution.message
                x=solution.x*.001;out=v.copy();out[ids]+=x.reshape(-1,3)
                gap=float((out[t[1]]@axis).min()-(out[t[0]]@axis).max())
                assert gap>=CLEARANCE-1e-12,(mode,trial,gap)
                intersections=all_pairs(out,t)
                if intersections:
                    np.savez_compressed(root/f'detector_{mode}_{trial}_{len(ids)}.npz',vertices=out,triangles=t,axis=axis,
                        current=v,step=x,ids=ids,rows=matrix.toarray(),lower=lower)
                # Independently reconstruct every vertex-pair inequality in original units.
                pairgap=(out[t[1]][None,:,:]-out[t[0]][:,None,:])@axis
                assert pairgap.min()>=CLEARANCE-1e-12
                records.append(dict(mode=mode,rotation=trial,active_ids=ids.tolist(),rows=len(lower),
                    minimum_projected_gap=gap,clearance=CLEARANCE,all_intersection_pairs=len(intersections)))
    base=np.array([[0.,0,0],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]])*.0002
    t=np.array([[0,1,2],[3,4,5]])
    m,b,r=contact_constraints(base,t,{(0,1)},[]);assert m.shape==(0,0) and len(b)==0
    fixed_checks={'separated_all_fixed_accepts':True}
    for name,delta in [('touching',-.0002),('overlapping',-.0003)]:
        value=base.copy();value[3:,0]+=delta
        try:contact_constraints(value,t,{(0,1)},[])
        except ValueError as error:fixed_checks[name]=str(error)
        else:raise AssertionError(name)
    # Shared mesh vertices are outside the actual Open3D pair proposal domain.
    shared=np.array([[0.,0,0],[-.0002,0,0],[-.0002,.0002,0],[.0002,0,0],[.0002,.0002,0]])
    st=np.array([[0,1,2],[0,3,4]])
    matrix,lower,rows=contact_constraints(shared,st,{(0,1)},np.arange(5))
    zero_feasible=bool((matrix@np.zeros(15)-lower).min()>=0)
    shared_probe=dict(zero_step_accepted=zero_feasible,actual_gap=0.,required_clearance=CLEARANCE,
        open3d_proposes_pair=bool(all_pairs(shared,st)),missing_zero_coefficient_row=True)
    sources=[Path(__file__).with_name(name) for name in ['mesh_contact_clearance.py','run_mhr_clearance_correction.py',
        'run_mhr_anatomical_correction.py','guard_mhr_anatomical_correction.py','fit_mhr_conic_correction.py']]
    for source in sources:(root/source.name).write_bytes(source.read_bytes())
    save(root/'result.json',dict(cases=records,fixed_checks=fixed_checks,shared_vertex_probe=shared_probe,
        script_sha256=sha(__file__),input_hashes={str(p):sha(p) for p in sources},actor_fit_launched=False,
        outputs={p.name:sha(p) for p in root.iterdir() if p.is_file()}))
    print(json.dumps(dict(cases=len(records),minimum_gap=min(r['minimum_projected_gap'] for r in records),
        fixed_checks=fixed_checks,shared_vertex_probe=shared_probe)),flush=True)


if __name__=='__main__':main()
