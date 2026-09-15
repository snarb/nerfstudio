"""Independent bounded solver review: synthetic algebra and actual pair inventory.

Does not run optimization on DEC5, change helpers, or edit fitted artifacts.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from constrained_surface_step import lower_bound_lsq,oriented_area_constraints
from mesh_contact_constraints import contact_constraints
from fit_mhr_guarded_correction import StepGuard,strict_pairs
from fit_mhr_bounded_contact_correction import ball_planes
from check_mhr_conformance_crossings import transverse_crossings
from study_multiview_face_prior import read,save,sha


def main():
    import open3d as o3d
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output
    assert not root.exists();root.mkdir();rng=np.random.default_rng(830193);worst=0.;kkt=0.;failures=[]
    for i in range(100):
        m=rng.normal(size=(8,4));m=np.r_[m,np.eye(4)*.2];rhs=rng.normal(size=len(m));c=rng.normal(size=(9,4))
        feasible=rng.normal(size=4) if i%2 else np.zeros(4);b=c@feasible-rng.uniform(.01,1,len(c))
        try:x,r=lower_bound_lsq(m,rhs,c,b)
        except Exception as error:failures.append(dict(case=i,error=str(error)));continue
        opt=minimize(lambda z:.5*np.sum((m@z-rhs)**2),feasible,jac=lambda z:m.T@(m@z-rhs),
            constraints={'type':'ineq','fun':lambda z:c@z-b,'jac':lambda z:c},method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
        worst=max(worst,abs(.5*np.sum((m@x-rhs)**2)-opt.fun));grad=m.T@(m@x-rhs)
        if r['active']:grad-=c[r['active']].T@np.array(r['multipliers'])
        kkt=max(kkt,np.linalg.norm(grad,np.inf));assert np.min(c@x-b)>=-1e-9
    errors=[]
    for i in range(100):
        v=rng.normal(size=(5,3));t=np.array([[0,1,2],[2,3,4]]);cross=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])
        ids=np.array([4,1,2]);c,b,sel=oriented_area_constraints(v,t,cross,np.ones(2,bool),ids,np.full(2,.01))
        direction=rng.normal(size=(len(ids),3));eps=1e-6;n=cross/np.linalg.norm(cross,axis=1)[:,None]
        def val(z):return np.sum(np.cross(z[t[:,1]]-z[t[:,0]],z[t[:,2]]-z[t[:,0]])*n,axis=1)
        vp=v.copy();vm=v.copy();vp[ids]+=eps*direction;vm[ids]-=eps*direction;grads=[]
        for j in range(len(ids)*3):
            up=v.copy();down=v.copy();up[ids[j//3],j%3]+=eps;down[ids[j//3],j%3]-=eps
            grads.append((val(up)-val(down))/(2*eps))
        grad=np.array(grads).T;errors.append(np.max(abs(c@direction.ravel()-(val(vp)-val(vm))/(2*eps)/np.linalg.norm(grad,axis=1))))
    v=np.array([[0,0,0],[.0002,0,0],[0,.0002,0],[.0004,0,0],[.0006,0,0],[.0004,.0002,0]],float)
    t=np.array([[0,1,2],[3,4,5]]);c,b,records=contact_constraints(v,t,[(0,1)],np.arange(6));step=np.zeros_like(v);step[3:,0]=-.00035;trial=v+step
    guard=StepGuard(v,t,v);mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(trial),o3d.utility.Vector3iVector(t))
    ok,detail=guard.check(trial)
    synthetic=dict(qp_cases=100,qp_failures=failures,maximum_objective_difference=float(worst),maximum_kkt_stationarity=float(kkt),
        signed_area_cases=100,maximum_signed_area_gradient_error=float(max(errors)),
        coplanar=dict(contact_rows=records,minimum_slack=float(np.min(c@step.ravel()-b)),
            open3d_intersections=np.asarray(mesh.get_self_intersecting_triangles()).tolist(),strict_pairs=[list(p) for p in strict_pairs(trial,t)],
            step_guard_accepted=ok,step_guard=detail))
    base=np.zeros((1,3));current=np.array([[.0008,0,0]])
    c,b=ball_planes(current,base,np.array([0]),[(0,np.array([1.,0,0]))],.001);step=np.array([.0002,.0008,0])
    synthetic['ball_supporting_plane']=dict(slack=(c@step-b).tolist(),endpoint_norm=float(np.linalg.norm(current[0]+step)),limit=.001,
        interpretation='Finite supporting halfspaces are outer approximation; exact ball guard remains necessary')
    np.savez_compressed(root/'coplanar_synthetic.npz',original=v,triangles=t,trial=trial)
    warm_path=Path('/mnt/data/dec5_mhr_silhouette_convergence/fit.npz');warm=np.load(warm_path);tri=warm['triangles'];neutral=warm['neutral']
    paths=dict(warm=warm_path,contact_final=Path('/mnt/data/dec5_mhr_contact_correction_v2/fit.npz'),
        bounded_last_009=Path('/mnt/data/dec5_mhr_bounded_contact_correction/iterates/009.npz'))
    hashes={str(p):sha(p) for p in paths.values()};all_pairs={};actual=[]
    for name,path in paths.items():
        fit=np.load(path);vertices=fit['vertices']
        if 'triangles' in fit:np.testing.assert_array_equal(fit['triangles'],tri)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(tri))
        pairs=np.asarray(mesh.get_self_intersecting_triangles(),int).reshape(-1,2);pairs=np.sort(pairs,axis=1)
        passed=transverse_crossings(vertices[tri[pairs[:,0]]],vertices[tri[pairs[:,1]]]);all_pairs[name]=set(map(tuple,pairs))
        new=np.array(sorted(all_pairs[name]-all_pairs['warm']),int).reshape(-1,2)
        strictnew=transverse_crossings(vertices[tri[new[:,0]]],vertices[tri[new[:,1]]]) if len(new) else np.zeros(0,bool)
        detail=[]
        for pair,strict in zip(new,strictnew):
            aa,bb=vertices[tri[pair]];na=np.cross(aa[1]-aa[0],aa[2]-aa[0]);nb=np.cross(bb[1]-bb[0],bb[2]-bb[0]);na/=np.linalg.norm(na);nb/=np.linalg.norm(nb)
            plane=max(np.max(abs((bb-aa[0])@na)),np.max(abs((aa-bb[0])@nb)));cos=float(abs(na@nb))
            detail.append(dict(pair=pair.tolist(),transverse=bool(strict),absolute_normal_cosine=cos,maximum_mutual_plane_distance=float(plane),
                coplanar_within1e10=bool(plane<=1e-10 and 1-cos<=1e-6),shared_vertices=len(set(tri[pair[0]])&set(tri[pair[1]])),
                neutral_y_range=[float(neutral[tri[pair],1].min()),float(neutral[tri[pair],1].max())]))
        np.savez_compressed(root/(name+'_pairs.npz'),all_pairs=pairs,transverse=passed,new_pairs=new,new_transverse=strictnew)
        record=dict(name=name,all_open3d_pairs=len(pairs),transverse_pairs=int(passed.sum()),new_all_pairs=len(new),
            new_transverse=int(strictnew.sum()),new_nontransverse=int((~strictnew).sum()),new_pair_details=detail)
        actual.append(record);print(record,flush=True)
    for path,h in hashes.items():assert sha(path)==h
    names=['constrained_surface_step.py','mesh_contact_constraints.py','fit_mhr_guarded_correction.py','fit_mhr_constrained_correction.py',
        'fit_mhr_contact_correction.py','fit_mhr_bounded_contact_correction.py','fit_mhr_silhouette_conformance.py','check_mhr_conformance_crossings.py']
    save(root/'result.json',dict(synthetic=synthetic,actual=actual,input_hashes=hashes,script_sha256=sha(__file__),
        helper_hashes={n:sha(Path(__file__).with_name(n)) for n in names},new_fit=False,inputs_or_helpers_modified=False,
        outputs={f.name:sha(f) for f in sorted(root.iterdir()) if f.is_file()}))
    print('independent review terminal',synthetic,flush=True)


if __name__=='__main__':main()
