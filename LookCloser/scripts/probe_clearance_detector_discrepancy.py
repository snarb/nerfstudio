"""Localize synthetic detector disagreements; no actor geometry is loaded."""
from pathlib import Path
from decimal import Decimal,localcontext
import argparse
import sys
import urllib.request
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from mesh_contact_clearance import contact_constraints,CLEARANCE
from guard_mhr_anatomical_correction import all_pairs
from study_multiview_face_prior import save,sha


def exact_projections(v,axis):
    with localcontext() as context:
        context.prec=80
        return [sum((Decimal(float(x))*Decimal(float(y)) for x,y in zip(row,axis)),Decimal(0)) for row in v]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.output;assert not root.exists();root.mkdir();records=[]
    assert o3d.__version__=='0.19.0'
    for path in sorted(args.input.glob('detector*.npz')):
        d=np.load(path);v=d['vertices'];t=d['triangles'];axis=d['axis'];q=exact_projections(v,axis)
        gap=min(q[i] for i in t[1])-max(q[i] for i in t[0]);assert gap>0
        transformed=[]
        for centered in [False,True]:
            for scale in [1,10,1000]:
                vv=(v-v.mean(0) if centered else v)*scale;qq=exact_projections(vv,axis)
                gg=min(qq[i] for i in t[1])-max(qq[i] for i in t[0]);assert gg>0
                transformed.append(dict(centered=centered,scale=scale,positive_gap_decimal=str(gg),
                    open3d_pair=bool(all_pairs(vv,t))))
        current=d['current'];matrix,lower,rows=contact_constraints(current,t,{(0,1)},[3,4,5])
        repeat=np.tile(np.eye(3),(3,1));reduced=matrix@repeat
        wanted=(current[:3].mean(0)-current[3:].mean(0))/.001
        solution=minimize(lambda x:.5*np.sum((x-wanted)**2),np.zeros(3),jac=lambda x:x-wanted,
            constraints=[dict(type='ineq',fun=lambda x:reduced@x-lower/.001,jac=lambda x:reduced)],
            method='SLSQP',options=dict(ftol=1e-13,maxiter=1000));assert solution.success
        rigid=current.copy();rigid[3:]+=.001*solution.x
        unit=np.array(rows[0]['axis']);rigid_gap=float((rigid[t[1]]@unit).min()-(rigid[t[0]]@unit).max())
        assert rigid_gap>=CLEARANCE-1e-12 and not all_pairs(rigid,t)
        area=np.linalg.norm(np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]]),axis=1)
        rigid_area=np.linalg.norm(np.cross(rigid[t[:,1]]-rigid[t[:,0]],rigid[t[:,2]]-rigid[t[:,0]]),axis=1)
        np.savez_compressed(root/path.name,vertices=v,triangles=t,axis=axis,projections=np.array(q,dtype=str),
            rigid_vertices=rigid,rigid_axis=unit)
        records.append(dict(source=str(path),source_sha256=sha(path),exact_gap=str(gap),
            cross_norms=area.tolist(),transform_checks=transformed,rigid_gap=rigid_gap,
            rigid_cross_norms=rigid_area.tolist(),rigid_open3d_pair=False))
    urls={
        'IntersectionTest.cpp':'cpp/open3d/geometry/IntersectionTest.cpp',
        'TriangleMesh.cpp':'cpp/open3d/geometry/TriangleMesh.cpp',
        'opttritri.h':'3rdparty/tomasakeninemoeller/include/tomasakeninemoeller/opttritri.h'}
    sources=[]
    for name,suffix in urls.items():
        url='https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/'+suffix
        payload=urllib.request.urlopen(url,timeout=30).read();(root/name).write_bytes(payload)
        sources.append(dict(url=url,file=name,sha256=sha(root/name)))
    loaded=[module.__file__ for name,module in sys.modules.items()
            if name.startswith('open3d.') and name.endswith('.pybind') and getattr(module,'__file__',None)]
    assert len(loaded)==1,loaded
    binary=loaded[0]
    save(root/'result.json',dict(records=records,official_sources=sources,installed_version=o3d.__version__,
        pybind_path=binary,pybind_sha256=sha(binary),script_sha256=sha(__file__),
        source_review='Pairwise per-axis sigma+1e-12 normalization before plane-expression epsilon1e-6; excludes shared vertices.',
        actor_guard_changed=False,actor_fit_launched=False,
        outputs={p.name:sha(p) for p in root.iterdir() if p.is_file()}))
    print('Disjoint but degenerate detector positives:',len(records),'Rigid nondegenerate controls pass:',len(records),flush=True)


if __name__=='__main__':main()
