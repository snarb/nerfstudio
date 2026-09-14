"""Train-only conservative guards for frozen camera-independent notch proposals."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from study_jaw_boundary_notches import PARENT,PHASE,FRAMES,SETTINGS

SOURCE=Path('/mnt/data/dec5_jaw_3d_boundary_notches')


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    parent=read(PARENT/'request.json');phase=read(PHASE/'request.json')
    request=dict(source_request_sha256=sha(SOURCE/'request.json'),script_sha256=sha(__file__),
        min_distinct_mask_support=2,mask_samples='three vertices and centroid',
        observed_surface='parent mesh raycasts, NOT independent measured depths',
        occlusion_tolerance=SETTINGS['max_occlusion_depth'],train_camera_count=62,
        ray_lattices=['half_pixel','integer'],heldout_used=False,production_changed=False)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Request mismatch')
    atomic_json(output/'request.json',request)
    for frame in FRAMES:
        root=output/frame;root.mkdir(exist_ok=True)
        row=next(r for r in parent['inventory'] if r['frame_id']==frame);source=read(SOURCE/frame/'result.json')
        if sha(row['mesh'])!=source['source_mesh_sha256'] or sha(SOURCE/frame/'candidate.ply')!=source['candidate_sha256']:
            raise ValueError('Changed mesh')
        mesh=o3d.io.read_triangle_mesh(row['mesh']);candidate=o3d.io.read_triangle_mesh(str(SOURCE/frame/'candidate.ply'))
        v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles);tt=np.asarray(candidate.triangles)
        if not np.array_equal(v,np.asarray(candidate.vertices)) or not np.array_equal(tt[:len(t)],t):raise ValueError('Prefix changed')
        proposals=tt[len(t):];points=np.concatenate([v[proposals],v[proposals].mean(1)[:,None]],axis=1)
        train,_,_=cameras(frame);maskroot=Path(row['source_masks']['root'])
        if sha(maskroot/'masks.npz')!=row['source_masks']['masks_sha256']:raise ValueError('Changed masks')
        masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
        support=np.zeros(len(proposals),int);outside=np.zeros(len(proposals),int);mask_veto=[]
        for cam in train:
            uv,z=project(points.reshape(-1,3),[cam]);uv=uv[0];z=z[0]
            available=(z>0)&(uv[:,0]>2)&(uv[:,0]<1917)&(uv[:,1]>2)&(uv[:,1]<1077)
            xy=np.rint(uv).astype(int);inside=np.zeros(len(uv),bool);ids=np.flatnonzero(available)
            inside[ids]=masks[names.index(cam['physical_camera']),xy[ids,1],xy[ids,0]]
            support+=inside.reshape(-1,4).all(1)
            veto=(available&~inside).reshape(-1,4).any(1)
            outside+=veto;mask_veto.append(veto)
        semantic_rejected=(support<2)|(outside>0)
        rejected=semantic_rejected.copy();records=[];occlusion_veto=[]
        old=scene_for(v,t);new=scene_for(v,tt)
        for i,cam in enumerate(train):
            per={};view_veto=np.zeros(len(proposals),bool)
            for lattice in ['half_pixel','integer']:
                # create_rays_pinhole samples pixel centers. Subtracting .5
                # from cx/cy would shift the wrong way; +.5 gives integer rays.
                actual=dict(cam)
                if lattice=='integer':actual['cx']+=.5;actual['cy']+=.5
                d0,_,_=camera_depth(old,actual);d1,ids,_=camera_depth(new,actual)
                bad=np.isfinite(d0)&np.isfinite(d1)&(ids>=len(t))&(ids<len(tt))&(d1<d0-.001)
                faces=np.unique(ids[bad]).astype(int)-len(t);rejected[faces]=True;view_veto[faces]=True
                per[lattice]=int(bad.sum())
            records.append(dict(camera=cam['physical_camera'],raw_occlusion_pixels=per))
            occlusion_veto.append(view_veto)
            if (i+1)%10==0:print(frame,'train_guard',i+1,'/62',flush=True)
        np.savez_compressed(root/'admission.npz',support=support,outside=outside,mask_veto=np.array(mask_veto),
            occlusion_veto=np.array(occlusion_veto),rejected=rejected)
        spots=read('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')['selected_components']
        spot=next(s for s in spots if s['frame_id']==frame);x0,y0,x1,y1=spot['bbox_inclusive']
        _,rawids,_=camera_depth(new,row['camera'])
        localids=np.unique(np.rot90(rawids)[y0:y1+1,x0:x1+1]).astype(np.int64)-len(t)
        localids=localids[(localids>=0)&(localids<len(proposals))]
        attribution=[]
        for k in localids:
            attribution.append(dict(proposal=int(k),support=int(support[k]),mask_veto_cameras=[c['physical_camera'] for c,a in zip(train,mask_veto) if a[k]],
                occlusion_veto_cameras=[c['physical_camera'] for c,a in zip(train,occlusion_veto) if a[k]],kept=bool(not rejected[k])))
        atomic_json(root/'spot_admission.json',dict(triangles=attribution,admission_sha256=sha(root/'admission.npz')))
        kept=proposals[~rejected];final=np.concatenate([t,kept]);new=scene_for(v,final)
        result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(final))
        result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(root/'candidate.ply'),result)
        checks=[]
        # Recast final geometry; deleting front proposals can reveal deeper ones.
        for cam in train:
            for lattice in ['half_pixel','integer']:
                actual=dict(cam)
                if lattice=='integer':actual['cx']+=.5;actual['cy']+=.5
                d0,_,_=camera_depth(old,actual);d1,ids,_=camera_depth(new,actual)
                bad=np.isfinite(d0)&np.isfinite(d1)&(ids>=len(t))&(ids<len(final))&(d1<d0-.001)
                checks.append(dict(camera=cam['physical_camera'],lattice=lattice,remaining_old_occlusions=int(bad.sum())))
        spots=read('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')['selected_components']
        spot=next(s for s in spots if s['frame_id']==frame);x0,y0,x1,y1=spot['bbox_inclusive']
        moving=[]
        for label,cam in [('old_moving',row['camera']),('phase_moving',next(r for r in phase['inventory'] if r['frame_id']==frame)['camera'])]:
            d0,_,_=camera_depth(old,cam);d1,ids,_=camera_depth(new,cam)
            valid=np.isfinite(d1);normals=np.asarray(result.triangle_normals)
            lighting=np.abs(normals@np.array([.3,.4,.866]));rgb=np.zeros((*ids.shape,3),np.uint8)
            rgb[valid]=(60+170*lighting[ids[valid],None]).astype(np.uint8);rgb[valid&(ids>=len(t))]=[240,60,50]
            portrait=np.rot90(rgb);Image.fromarray(portrait).save(root/f'{label}_clay_added.png')
            rec=dict(camera=label,newly_visible=int((valid&~np.isfinite(d0)).sum()),
                old_occlusions=int((valid&np.isfinite(d0)&(d1<d0-.001)).sum()))
            if label=='old_moving':
                rec['spot_remaining_misses']=int((~np.rot90(valid)[y0:y1+1,x0:x1+1]).sum())
                Image.fromarray(portrait).crop((x0-65,y0-65,x1+66,y1+66)).save(root/'spot_added_native.png')
            moving.append(rec)
        atomic_json(root/'result.json',dict(candidate_sha256=sha(root/'candidate.ply'),source_result_sha256=sha(SOURCE/frame/'result.json'),
            original_triangle_positions_unchanged=True,proposed=len(proposals),kept=len(kept),
            train_guard_records=records,final_checks=checks,moving=moving,
            visual_status='pending',production_accepted=False))
        print(frame,'kept',len(kept),'remaining_train_occlusions',sum(x['remaining_old_occlusions'] for x in checks),moving,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_jaw_3d_boundary_guarded'))
    run(p.parse_args().output)
