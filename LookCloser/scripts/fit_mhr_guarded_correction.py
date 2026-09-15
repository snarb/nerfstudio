"""Train-only warm-start silhouette correction with discrete geometry guards.

This deliberately changes the fit, not admission. The selected margin-two
prior is the new local displacement reference. No target residual or target ray
is read. Guards certify sampled iterates, not continuous collision freedom.
"""
from pathlib import Path
import inspect
import time
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
import fit_mhr_silhouette_conformance as fit
from check_mhr_conformance_crossings import transverse_crossings
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_guarded_correction')
WARM=Path('/mnt/data/dec5_mhr_silhouette_convergence')
SETTINGS=dict(outer_iterations=40,maximum_step=.00025,maximum_displacement=.001,
    magnitude_sigma=.001,minimum_relative_area=.25,minimum_normal_cosine=.5,
    maximum_halvings=12,silhouette_offset_pixels=0.,
    maximum_new_transverse_pairs=0,maximum_new_normal_reversals=0,
    discrete_iterate_guard_not_continuous_collision_detection=True)


def crosses(v,t):
    return np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])


def strict_pairs(v,t):
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    pairs=np.asarray(mesh.get_self_intersecting_triangles(),int).reshape(-1,2)
    if not len(pairs):return set()
    good=transverse_crossings(v[t[pairs[:,0]]],v[t[pairs[:,1]]])
    return set(map(tuple,np.sort(pairs[good],axis=1)))


class StepGuard:
    def __init__(self,base,triangles,original):
        self.base=base.copy();self.t=triangles.copy();self.cross=crosses(base,triangles)
        self.area=np.linalg.norm(self.cross,axis=1);assert (self.area>0).all()
        self.original_cross=crosses(original,triangles)
        self.original_good=np.sum(self.original_cross*self.cross,axis=1)>0
        self.allowed_pairs=strict_pairs(base,triangles);self.history=[];self.stalled=False

    def check(self,trial):
        if not np.isfinite(trial).all():return False,dict(reason='nonfinite')
        displacement=float(np.linalg.norm(trial-self.base,axis=1).max())
        c=crosses(trial,self.t);area=np.linalg.norm(c,axis=1)
        ratio=area/self.area;cos=np.sum(c*self.cross,axis=1)/np.maximum(area*self.area,1e-30)
        reversals=self.original_good & (np.sum(self.original_cross*c,axis=1)<=0)
        detail=dict(maximum_displacement=displacement,minimum_area_ratio=float(ratio.min()),
                    minimum_normal_cosine=float(cos.min()),new_original_normal_reversals=int(reversals.sum()))
        if displacement>SETTINGS['maximum_displacement']+1e-12:return False,dict(detail,reason='displacement_bound')
        if ratio.min()<SETTINGS['minimum_relative_area']:return False,dict(detail,reason='area')
        if cos.min()<SETTINGS['minimum_normal_cosine']:return False,dict(detail,reason='normal_rotation')
        if reversals.any():return False,dict(detail,reason='new_original_normal_reversal')
        pairs=strict_pairs(trial,self.t);new=pairs-self.allowed_pairs
        detail.update(strict_pairs=len(pairs),new_strict_pairs=len(new))
        return not new,dict(detail,reason='new_intersection' if new else 'accepted')

    def update(self,current,ids,step):
        rejected=[]
        for halving in range(SETTINGS['maximum_halvings']+1):
            factor=2.**-halving;trial=current.copy();trial[ids]+=factor*step
            ok,detail=self.check(trial)
            if ok:
                self.history.append(dict(factor=factor,halvings=halving,rejected=rejected,**detail))
                return trial,step*factor
            rejected.append(detail)
        self.stalled=True
        self.history.append(dict(factor=0.,reason='guard_stalled_not_converged',rejected=rejected))
        return current.copy(),np.zeros_like(step)


def optimizer(guard):
    source=inspect.getsource(fit.optimize)
    replacements={
        'excess = np.maximum(values-2, 0)':'excess = np.maximum(values, 0)',
        'current[ids] += step':'current, step = guard.update(current, ids, step)',
        "save(ROOT/'progress.json', dict(history=history))":
            "save(ROOT/'progress.json', dict(history=history, guards=guard.history))\n"
            "        np.savez_compressed(ROOT/'iterates'/f'{outer+1:03d}.npz',vertices=current)\n"
            "        if guard.stalled: break",
    }
    generated=source
    for old,new in replacements.items():
        assert generated.count(old)==1,old
        generated=generated.replace(old,new)
    recipe=dict(fit.RECIPE,outer_iterations=SETTINGS['outer_iterations'],
        maximum_step=SETTINGS['maximum_step'],magnitude_sigma=SETTINGS['magnitude_sigma'],
        boundary_tolerance_pixels=0.)
    namespace=dict(fit.__dict__,ROOT=ROOT,RECIPE=recipe,guard=guard)
    # Keep diagnostic statistics aligned with this zero-offset experiment.
    def stats(v,rows,sdfs):
        values=np.concatenate([r[1] for r in fit.silhouette_samples(v,rows,sdfs)])
        return dict(available_samples=len(values),outside_samples=int((values>0).sum()),
            mean_positive_sdf=float(np.maximum(values,0).mean()),max_positive_sdf=float(np.maximum(values,0).max()))
    namespace['silhouette_stats']=stats
    exec(compile(generated,'<guarded_warm_start_correction>','exec'),namespace)
    save(ROOT/'optimizer_source.json',dict(original_source=source,generated_source=generated,
        replacements=replacements,recipe=recipe))
    return namespace['optimize'],stats


def main():
    start=time.monotonic();assert not ROOT.exists()
    parent,rows,masks,names,evidence,validation=fit.prepare()
    warm_result=read(WARM/'result.json')
    assert warm_result['hashes']['fit.npz']==sha(WARM/'fit.npz')
    assert warm_result['protocol_sha256']==sha(WARM/'protocol.json')
    warm=np.load(WARM/'fit.npz');initial=np.load(fit.SOURCE/'initial.npz')
    base,t,neutral=warm['vertices'],warm['triangles'],warm['neutral']
    np.testing.assert_array_equal(t,initial['triangles']);np.testing.assert_array_equal(neutral,initial['neutral'])
    original=np.load(fit.SOURCE/'smooth100/fit.npz')['vertices']
    active=(neutral[:,1]>135)&(neutral[:,1]<153)
    ROOT.mkdir();(ROOT/'iterates').mkdir()
    files=[WARM/'fit.npz',WARM/'result.json',WARM/'protocol.json',fit.SOURCE/'smooth100/fit.npz',
        fit.SOURCE/'initial.npz',fit.SOURCE/'anchors.npz',fit.SOURCE/'anchors.json',fit.SOURCE/'protocol.json']
    helpers=[Path(fit.__file__),Path(__file__).with_name('check_mhr_conformance_crossings.py'),
        Path(__file__).with_name('conform_mhr_measured_surface.py'),Path(__file__).with_name('admit_mhr_local_patch_depth.py')]
    save(ROOT/'protocol.json',dict(frame='001193',settings=SETTINGS,
        input_hashes={str(p):sha(p) for p in files+helpers},script_sha256=sha(__file__),
        fit_cameras=[r['physical_camera'] for r,v in zip(rows,validation) if not v],
        validation_cameras=[r['physical_camera'] for r,v in zip(rows,validation) if v],
        evidence=evidence,original_mesh=parent['original_mesh'],original_mesh_sha256=parent['original_mesh_sha256'],
        reference='margin-two fitted prior; not original smooth100 base',target_used=False,
        admission_changed=False,production_accepted=False))
    guard=StepGuard(base,t,original);worker,stats=optimizer(guard)
    sdfs=[]
    for row in rows:
        mask=masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~mask)-distance_transform_edt(mask)).astype(np.float32))
    tr=[r for r,v in zip(rows,validation) if not v];ts=[s for s,v in zip(sdfs,validation) if not v]
    vr=[r for r,v in zip(rows,validation) if v];vs=[s for s,v in zip(sdfs,validation) if v]
    obs=np.load(fit.SOURCE/'anchors.npz');train=~validation[obs['camera']]
    current,history=worker(base,t,neutral,obs['points'][train],obs['normals'][train],obs['neck'][train],tr,ts)
    np.testing.assert_array_equal(current[~active],base[~active]);ok,safety=guard.check(current);assert ok
    np.savez_compressed(ROOT/'fit.npz',vertices=current,baseline=base,original_reference=original,
        triangles=t,neutral=neutral,active=active,displacement=current-base)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(current),o3d.utility.Vector3iVector(t));mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(ROOT/'prior_only.ply'),mesh)
    save(ROOT/'result.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),
        before=dict(train=stats(base[active],tr,ts),validation=stats(base[active],vr,vs)),
        after=dict(train=stats(current[active],tr,ts),validation=stats(current[active],vr,vs)),
        safety=safety,history=history,guards=guard.history,seconds=time.monotonic()-start,
        stop_reason='guard_stalled_not_converged' if guard.stalled else 'iteration_cap_not_converged',
        original_geometry_unchanged=True,prior_only=True,production_accepted=False,visual_status='pending',
        hashes={n:sha(ROOT/n) for n in ['fit.npz','prior_only.ply','optimizer_source.json']}))
    print('terminal',read(ROOT/'result.json')['stop_reason'],'seconds',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
