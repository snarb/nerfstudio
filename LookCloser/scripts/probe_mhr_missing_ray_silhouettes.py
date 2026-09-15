"""Bounded post-hoc ray/silhouette feasibility, not a reconstruction update."""
from pathlib import Path
import numpy as np
import cv2
from build_train_hair_semantics import read, sha, write
from joint_temporal_texture import project
from admit_mhr_local_patch_depth import inputs, PRIOR, OUT


def main():
    _,rows,_,masks,names,binding=inputs()
    dest=OUT/'ray_silhouette_probe'; dest.mkdir(exist_ok=False)
    parent=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    camera=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']=='001193')
    path=PRIOR/'probe_smooth025_smooth100_smooth400/smooth025.npz'
    q=np.load(path)['target_prior_points']; center=np.asarray(camera['transform_matrix'])[:3,3]
    unit=q-center; unit/=np.linalg.norm(unit,axis=1,keepdims=True)
    offsets=np.linspace(-.01,.01,401); points=(q[:,None,:]+unit[:,None,:]*offsets[None,:,None]).reshape(-1,3)
    records=[]; witnesses={'C004_E005_1210X7','D004_E005_1210GX','E004_D005_1210L4','E004_E005_1210WX'}
    for radius in [0,2]:
        outside=np.zeros((len(rows),len(points)),bool); available=np.zeros_like(outside)
        for ci,row in enumerate(rows):
            uv,z=project(points,[row]); uv=uv[0]; z=z[0]; xy=np.rint(uv).astype(int)
            ok=(z>0)&(uv[:,0]>2)&(uv[:,0]<1917)&(uv[:,1]>2)&(uv[:,1]<1077); available[ci]=ok
            mask=masks[names.index(row['physical_camera'])].astype(np.uint8)
            if radius: mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
            ids=np.flatnonzero(ok); outside[ci,ids]=~mask[xy[ids,1],xy[ids,0]].astype(bool)
        all_count=outside.sum(0).reshape(len(q),-1)
        subset=np.array([r['physical_camera'] in witnesses for r in rows]); witness_count=outside[subset].sum(0).reshape(len(q),-1)
        feasible=(all_count==0)&(available.sum(0).reshape(len(q),-1)>=2)
        credible=witness_count==0; best=np.argmin(all_count,axis=1)
        np.savez_compressed(dest/('radius'+str(radius)+'.npz'),offsets=offsets,query_points=q,unit=unit,
            outside_counts=all_count,witness_outside_counts=witness_count,feasible=feasible,
            outside_by_camera=outside.reshape(len(rows),len(q),-1),available_by_camera=available.reshape(len(rows),len(q),-1))
        distances=np.where(feasible,abs(offsets)[None],np.inf).min(1)
        result=dict(mask_dilation_pixels=radius,rays=len(q),samples_per_ray=len(offsets),
            rays_with_any_all_camera_feasible_point=int(feasible.any(1).sum()),
            rays_with_any_four_witness_feasible_point=int(credible.any(1).sum()),
            minimum_outside_cameras_per_ray=all_count.min(1).tolist(),best_offsets=offsets[best].tolist(),
            nearest_feasible_offset=[float(d) if np.isfinite(d) else None for d in distances],
            zero_offset_outside_counts=all_count[:,200].tolist())
        records.append(result); print({k:v for k,v in result.items() if not isinstance(v,list)},flush=True)
    write(dest/'result.json',dict(arms=records,inputs=binding,query_sha256=sha(path),target_parent_sha256=sha(parent),
        script_sha256=sha(__file__),probe_interval_scene_units=[-.01,.01],step=.00005,
        witnesses=sorted(witnesses),mask_dilation_diagnostic_only=True,target_used_posthoc_only=True,
        geometry_or_masks_changed=False,depth_support_not_tested=True,production_accepted=False))


if __name__=='__main__': main()
