"""Per-time native prior review and original missing-ray/rim diagnostics only."""
import argparse
import importlib
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation,distance_transform_edt
from transfer_mhr_001195 import ROOT,HEAD,FINAL,FRAME,configure,verify
from study_multiview_face_prior import read,save,sha
from triangulate_face_prior import quantiles


def locality():
    import open3d as o3d
    from admit_mhr_local_patch_depth import Scene2
    from bake_joint_temporal_mesh import camera_depth
    from study_confidence_depth_prior import unproject
    import fit_mhr_silhouette_conformance as fit
    folder=FINAL/'locality';folder.mkdir(exist_ok=False)
    q=read(FINAL/'protocol.json');a=np.load(FINAL/'fit.npz');old=o3d.io.read_triangle_mesh(q['original_mesh']);old.compute_triangle_normals()
    scene=Scene2(np.asarray(old.vertices),np.asarray(old.triangles))
    parent=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json');spots=Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')
    camera=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']==FRAME)
    x0,y0,x1,y1=next(r['bbox_inclusive'] for r in read(spots)['selected_components'] if r['frame_id']==FRAME)
    depth,_,_=camera_depth(scene,camera);portrait=np.rot90(depth);missing=np.zeros(portrait.shape,bool)
    missing[y0:y1+1,x0:x1+1]=~np.isfinite(portrait[y0:y1+1,x0:x1+1])
    assert missing.sum()==73,'Frozen original requested-hole count changed'
    ring=binary_dilation(missing)&~missing&np.isfinite(portrait)
    yy,xx=np.nonzero(ring);rim_native=np.c_[1919-yy,xx]
    rim=unproject(camera,rim_native[:,0],rim_native[:,1],portrait[yy,xx],offset=.5)
    yy,xx=np.nonzero(missing);native=np.c_[1919-yy,xx]
    center=np.asarray(camera['transform_matrix'])[:3,3];direction=unproject(camera,native[:,0],native[:,1],np.ones(len(native)),offset=.5)-center
    rays=o3d.core.Tensor(np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32))
    _,rows,masks,names,_,_=fit.prepare();sdfs=[]
    for row in rows:
        m=masks[names.index(row['physical_camera'])].astype(bool);sdfs.append((distance_transform_edt(~m)-distance_transform_edt(m)).astype(np.float32))
    rc=scene.compute_closest_points(o3d.core.Tensor(rim.astype(np.float32)));normals=np.asarray(old.triangle_normals)[rc['primitive_ids'].numpy()];records=[]
    for label,key in [('baseline','baseline'),('silhouette','vertices')]:
        prior=Scene2(a[key],a['triangles']);hit=prior.cast_rays(rays);d=hit['t_hit'].numpy();valid=np.isfinite(d);points=center+direction[valid]*d[valid,None]
        cp=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))['points'].numpy();distance=np.linalg.norm(cp-points,axis=1)
        sdf=np.full((62,len(points)),np.nan)
        for ci,(ids,s,_) in enumerate(fit.silhouette_samples(points,rows,sdfs)):sdf[ci,ids]=s
        rp=prior.compute_closest_points(o3d.core.Tensor(rim.astype(np.float32)))['points'].numpy();delta=rp-rim
        np.savez_compressed(folder/(label+'.npz'),native_xy=native,hit_depth=d,hit_valid=valid,points=points,surface_distance=distance,sdf=sdf,
            outside_cameras=(sdf>2).sum(0),rim_points=rim,rim_closest=rp,rim_signed_offset=np.sum(delta*normals,axis=1))
        records.append(dict(arm=label,original_missing=len(native),prior_hits=int(valid.sum()),hit_surface_distance=quantiles(distance),
            soft_all62_pass=int(((sdf<=2).sum(0)==62).sum()),strict_all62_pass=int(((sdf<=0).sum(0)==62).sum()),
            available_cameras=quantiles(np.isfinite(sdf).sum(0)),rim_points=len(rim),rim_distance=quantiles(np.linalg.norm(delta,axis=1)),
            rim_signed_offset=quantiles(np.sum(delta*normals,axis=1))))
    save(folder/'result.json',dict(records=records,script_sha256=sha(__file__),input_hashes={str(p):sha(p) for p in [parent,spots,FINAL/'fit.npz',FINAL/'protocol.json']},
        rim_definition='one native pixel dilation of fixed original missing pixels intersect original hits',target_used_posthoc_only=True))
    print(records,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['head','final']);a=p.parse_args();verify();_,_,_,fit,_=configure()
    if a.stage=='head':importlib.import_module('review_mhr_local_head_prior').main(['similarity','head20','head20_neck6'])
    else:
        fit.ROOT=FINAL;importlib.import_module('review_mhr_silhouette_conformance').main();locality()
    save(ROOT/('review_'+a.stage+'_complete.json'),dict(config_sha256=sha(ROOT/'config.json'),script_sha256=sha(__file__)))


if __name__=='__main__':main()
