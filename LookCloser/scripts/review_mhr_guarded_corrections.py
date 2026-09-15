"""Fixed-cohort fit review and posthoc residual checks; never modifies fitting."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import distance_transform_edt
import fit_mhr_silhouette_conformance as fit
from fit_mhr_guarded_correction import StepGuard
from bake_joint_temporal_mesh import camera_depth
from study_jaw_repair_transfer import mask_votes
from review_jaw_repair_transfer import panel
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_guarded_correction_review')
ARMS={'margin2':Path('/mnt/data/dec5_mhr_silhouette_convergence'),
      'guarded':Path('/mnt/data/dec5_mhr_guarded_correction'),
      'area_qp':Path('/mnt/data/dec5_mhr_constrained_correction'),
      'contacts':Path('/mnt/data/dec5_mhr_contact_correction_v2')}


def main():
    assert not ROOT.exists();ROOT.mkdir()
    _,rows,masks,names,evidence,validation=fit.prepare()
    sdfs=[]
    for row in rows:
        mask=masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~mask)-distance_transform_edt(mask)).astype(np.float32))
    source=Path('/mnt/data/dec5_mhr_production_patch_001193')
    residual=source/'residual_hole/evidence.npz';xy=np.load(residual)['portrait_xy']
    camera_path=source/'admission/rgb/F004_E/baseline/frames/001193/result.json'
    camera=read(camera_path)['camera'];fits={};grids={};stats={};clays=[];bindings={}
    arrays={'portrait_xy':xy,'validation':validation}
    for name,root in ARMS.items():
        result=read(root/'result.json');assert result['protocol_sha256']==sha(root/'protocol.json')
        for filename,h in result['hashes'].items():assert sha(root/filename)==h
        data=np.load(root/'fit.npz');fits[name]=data
        for p in [root/'fit.npz',root/'protocol.json',root/'result.json']:bindings[str(p)]=sha(p)
        v,t,active=data['vertices'],data['triangles'],data['active']
        grid=np.full((62,int(active.sum())),np.nan)
        for ci,(ids,values,_) in enumerate(fit.silhouette_samples(v[active],rows,sdfs)):grid[ci,ids]=values
        grids[name]=grid;arrays[name+'_sdf']=grid
        scene=fit.Scene2(v,t);depth,ids,bary=camera_depth(scene,camera)
        d=np.rot90(depth)[xy[:,1],xy[:,0]];faces=np.rot90(ids)[xy[:,1],xy[:,0]]
        uv=np.rot90(bary)[xy[:,1],xy[:,0]];hit=np.isfinite(d)
        weights=np.c_[1-uv[hit].sum(1),uv[hit]]
        points=(v[t[faces[hit]]]*weights[:,:,None]).sum(1)
        repeated=np.repeat(np.arange(len(points))[:,None],3,axis=1)
        support,outside=mask_votes(points,repeated,rows,masks,names)
        parent_support,parent_outside=mask_votes(v,t[faces[hit]],rows,masks,names)
        if name=='margin2':guard=StepGuard(v,t,np.load(fit.SOURCE/'smooth100/fit.npz')['vertices'])
        ok,safety=guard.check(v);assert ok
        stats[name]=dict(prior_ray_hits=int(hit.sum()),first_hit_binary_mask_pass=int(((support>=2)&(outside==0)).sum()),
            parent_triangle_binary_mask_pass=int(((parent_support>=2)&(parent_outside==0)).sum()),safety=safety,
            ray_point_checks_not_admitted_patch=True)
        arrays.update({name+'_depth':d,name+'_faces':faces,name+'_points':points,
                       name+'_point_mask_outside':outside,name+'_parent_mask_outside':parent_outside})
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_triangle_normals()
        valid=np.isfinite(depth);rgb=np.full((*depth.shape,3),20,np.uint8)
        rgb[valid]=(60+170*abs(np.asarray(mesh.triangle_normals)[ids[valid]]@np.array([.3,.4,.866])))[:,None]
        clays.append(np.rot90(rgb))
    common=np.logical_and.reduce([np.isfinite(g) for g in grids.values()]);arrays['common']=common
    cohorts={}
    for split,selected in [('train',~validation),('reserved',validation)]:
        take=common&selected[:,None]
        cohorts[split]=dict(samples=int(take.sum()),arms={name:dict(outside=int((g[take]>0).sum()),
            mean_positive_sdf=float(np.maximum(g[take],0).mean()),maximum_positive_sdf=float(np.maximum(g[take],0).max()))
            for name,g in grids.items()})
    panel(ROOT/'residual_clay.png',clays,list(ARMS),(630,1080,775,1205))
    np.savez_compressed(ROOT/'evidence.npz',**arrays)
    for name,root in ARMS.items():
        if name=='margin2':continue
        import review_mhr_silhouette_conformance as native
        native.ROOT=root
        if not (root/'review_v2').exists():native.main()
    for cam in ['C004_E005_1210X7','G004_B005_1210FG','M004_B005_12109O']:
        images=[]
        for root in ARMS.values():
            im=Image.open(root/'review_v2'/(cam+'_clay.png')).convert('RGB')
            images.append(np.array(im.crop((im.width*3//4,24,im.width,im.height))))
        h,w=images[0].shape[:2];panel(ROOT/(cam+'.png'),images,list(ARMS),(0,0,w,h))
    for p in [residual,camera_path,Path(__file__).resolve()]:bindings[str(p)]=sha(p)
    save(ROOT/'result.json',dict(fixed_cohorts=cohorts,residual_posthoc=stats,input_hashes=bindings,
        evidence=evidence,hashes={p.name:sha(p) for p in ROOT.iterdir() if p.is_file()},
        target_used_in_fit=False,visual_status='pending',production_accepted=False))
    print(cohorts,flush=True);print(stats,flush=True)


if __name__=='__main__':main()
