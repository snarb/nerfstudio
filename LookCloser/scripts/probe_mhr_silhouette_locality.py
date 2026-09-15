"""Frozen requested-ray/rim posthoc checks; not fit input or patch construction."""
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from fit_mhr_silhouette_conformance import ROOT, prepare, silhouette_samples
from admit_mhr_local_patch_depth import Scene2
from study_multiview_face_prior import read, save, sha
from study_confidence_depth_prior import unproject
from triangulate_face_prior import quantiles


def main():
    import open3d as o3d
    from bake_joint_temporal_mesh import camera_depth
    dest=ROOT/'locality';dest.mkdir(exist_ok=False)
    protocol=read(ROOT/'protocol.json');a=np.load(ROOT/'fit.npz')
    original=o3d.io.read_triangle_mesh(protocol['original_mesh']);original.compute_triangle_normals()
    oldscene=Scene2(np.asarray(original.vertices),np.asarray(original.triangles))
    parent=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    spots=Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')
    rimpath=Path('/mnt/data/dec5_mhr_local_head_prior/probe_head20_neck6/head20_neck6.npz')
    rim=np.load(rimpath)['rim_points']
    camera=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']=='001193')
    x0,y0,x1,y1=next(r['bbox_inclusive'] for r in read(spots)['selected_components'] if r['frame_id']=='001193')
    depth,_,_=camera_depth(oldscene,camera);missing=~np.isfinite(np.rot90(depth)[y0:y1+1,x0:x1+1]);y,x=np.nonzero(missing)
    native=np.c_[1919-(y+y0),x+x0]
    center=np.asarray(camera['transform_matrix'])[:3,3]
    direction=unproject(camera,native[:,0],native[:,1],np.ones(len(native)),offset=.5)-center
    rays=o3d.core.Tensor(np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32))
    _,rows,masks,names,_,_=prepare();sdfs=[]
    for row in rows:
        mask=masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~mask)-distance_transform_edt(mask)).astype(np.float32))
    rc=oldscene.compute_closest_points(o3d.core.Tensor(rim.astype(np.float32)))
    rimnormals=np.asarray(original.triangle_normals)[rc['primitive_ids'].numpy()]
    records=[]
    for label,key in [('baseline','baseline'),('silhouette','vertices')]:
        v=a[key];scene=Scene2(v,a['triangles']);hit=scene.cast_rays(rays);d=hit['t_hit'].numpy();valid=np.isfinite(d)
        points=center+direction[valid]*d[valid,None]
        closest=oldscene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))['points'].numpy()
        distance=np.linalg.norm(closest-points,axis=1)
        values=np.full((62,len(points)),np.nan)
        for ci,(ids,s,_) in enumerate(silhouette_samples(points,rows,sdfs)):values[ci,ids]=s
        outside=(values>2).sum(0)
        rp=scene.compute_closest_points(o3d.core.Tensor(rim.astype(np.float32)))['points'].numpy()
        rd=rp-rim;signed=np.sum(rd*rimnormals,axis=1)
        np.savez_compressed(dest/(label+'.npz'),native_xy=native,hit_depth=d,hit_valid=valid,points=points,
            surface_distance=distance,sdf=values,outside_cameras=outside,rim_points=rim,rim_closest=rp,rim_signed_offset=signed)
        records.append(dict(arm=label,original_missing=len(native),prior_hits=int(valid.sum()),
            hit_surface_distance=quantiles(distance),hits_within_locality=int((distance<=.002).sum()),
            mask_disagreement_cameras=quantiles(outside),all62_mask_pass=int((outside==0).sum()),
            rim_distance=quantiles(np.linalg.norm(rd,axis=1)),rim_signed_normal_offset=quantiles(signed)))
    bindings={str(p):sha(p) for p in [parent,spots,rimpath,ROOT/'fit.npz',ROOT/'protocol.json']}
    save(dest/'result.json',dict(records=records,input_hashes=bindings,script_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/'protocol.json'),target_used_posthoc_only=True,mask_soft_tolerance=2.,
        production_accepted=False,patch_generated=False))


if __name__=='__main__':main()
