"""Confidence-qualified ray fallback for untexturable, unsupported inward hits.

Combine two verified renders of the identical mesh/camera. Only an originally
black backface with zero measured near views can reveal the next front surface,
which needs >=3 measured near views and an admitted train RGB source. This is
ray visibility, not mesh editing, RGB inpainting, averaging or screen-space blur.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from study_multiview_face_prior import read,save,sha
from study_target_backface_culling import ROOT as CULLED,PARENT,FRAME
from review_measured_free_surface import VIEWS
from study_confidence_depth_prior import load_real,project_integer
from review_full_block_transfer import ROOT as DEPTH_ROOT
from prune_measured_free_surface import near_tap_evidence
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for

ROOT=Path('/mnt/data/dec5_supported_front_surface_fallback')


def eligible(original_black,original_backface,original_source_missing,original_near,
             candidate_colored,candidate_source_valid,candidate_near,behind):
    return (original_black & original_backface & original_source_missing & (original_near==0)
            & candidate_colored & candidate_source_valid & (candidate_near>=3) & behind)


def main():
    assert not ROOT.exists();ROOT.mkdir()
    rows,depths,receipt=load_real(DEPTH_ROOT,FRAME)
    for view in VIEWS:
        roots=[PARENT/'carved'/FRAME/'rgb'/view,CULLED/FRAME/view]
        req=[];res=[];images=[];source=[];checks={};raycasts=[];vertices=None;triangles=None
        for root in roots:
            folder=root/'frames'/FRAME;complete=read(folder/'complete.json')
            assert complete['request_sha256']==sha(root/'request.json')
            for name,h in complete['hashes'].items():assert sha(folder/name)==h;checks[str(folder/name)]=h
            checks[str(root/'request.json')]=sha(root/'request.json');req.append(read(root/'request.json'));res.append(read(folder/'result.json'))
            images.append(np.array(Image.open(folder/'prediction_native.png')));source.append(np.array(Image.open(folder/'source_ids.png')))
        for k in ['mesh_sha256','camera','source_cameras','fixed_exposure']:assert res[0][k]==res[1][k]
        for k in ['recipe','profiles_sha256','exposure_sha256','calibration_sha256']:assert req[0][k]==req[1][k]
        assert not any(r['target_rgb_read'] or r['rgb_averaging'] for r in res)
        assert res[0]['source_cameras']==[r['physical_camera'] for r in rows]
        mesh=o3d.io.read_triangle_mesh(res[0]['mesh_path']);assert sha(res[0]['mesh_path'])==res[0]['mesh_sha256']
        v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
        kept=np.load(roots[1]/'target_retained_faces.npy')
        full=camera_depth(scene_for(v,t),res[0]['camera']);front=camera_depth(scene_for(v,t[kept]),res[0]['camera'])
        for casts,root in [(full,roots[0]),(front,roots[1])]:
            np.testing.assert_array_equal(np.where(np.isfinite(casts[0]),casts[0],0),np.load(root/'frames'/FRAME/'target_depth.npz')['depth'])
        full_ids=full[1];front_ids=np.full(front[1].shape,np.iinfo(np.uint32).max,np.uint32)
        finite=np.isfinite(front[0]);front_ids[finite]=kept[front[1][finite]]
        candidates=(images[0].max(2)==0)&(source[0]==255)&(images[1].max(2)>0)&(source[1]<62)&np.isfinite(full[0])&finite
        yy,xx=np.nonzero(candidates);oldfaces=full_ids[candidates];newfaces=front_ids[candidates]
        points=[]
        for cast,faces in [(full,oldfaces),(front,newfaces)]:
            b=cast[2][candidates];weights=np.c_[1-b.sum(1),b]
            points.append((v[t[faces]]*weights[:,:,None]).sum(1))
        allpoints=np.concatenate(points);near=np.zeros(len(allpoints),np.uint8)
        for row,d in zip(rows,depths):
            uv,z=project_integer(row,allpoints);near+=near_tap_evidence(d,uv,z,radius=0)
        oldnear,newnear=np.split(near,2)
        tv=v[t[oldfaces]];normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);center=np.array(res[0]['camera']['transform_matrix'])[:3,3]
        back=np.sum(normal*(center-tv[:,0]),axis=1)<=0
        behind=front[0][candidates]>full[0][candidates]+1e-6
        accept=eligible(np.ones(len(xx),bool),back,np.ones(len(xx),bool),oldnear,
            np.ones(len(xx),bool),np.ones(len(xx),bool),newnear,behind)
        mask=np.zeros(candidates.shape,bool);mask[yy[accept],xx[accept]]=True
        rgb=images[0].copy();rgb[mask]=images[1][mask];ids=source[0].copy();ids[mask]=source[1][mask]
        depth=np.where(np.isfinite(full[0]),full[0],0);depth[mask]=front[0][mask]
        np.testing.assert_array_equal(rgb[~mask],images[0][~mask]);assert not ((images[0].max(2)>0)&(rgb.max(2)==0)).any()
        out=ROOT/FRAME/view;out.mkdir(parents=True)
        Image.fromarray(rgb).save(out/'prediction_native.png');Image.fromarray(np.rot90(rgb)).save(out/'frame.png')
        Image.fromarray(ids).save(out/'source_ids.png');np.savez_compressed(out/'target_depth.npz',depth=depth)
        np.savez_compressed(out/'evidence.npz',mask=mask,candidate_xy=np.c_[xx,yy],accepted=accept,
            original_points=points[0],candidate_points=points[1],original_near=oldnear,candidate_near=newnear,
            original_backface=back,behind=behind,old_face_ids=oldfaces,new_face_ids=newfaces)
        save(out/'result.json',dict(view=view,input_hashes=checks,mesh_sha256=res[0]['mesh_sha256'],
            candidates=len(xx),accepted=int(accept.sum()),original_near=oldnear.tolist(),candidate_near=newnear.tolist(),
            selected_source_ids=ids[mask].tolist(),minimum_next_surface_views=3,
            near_tolerance=.0015,depth_receipt=receipt,original_colored_pixels_unchanged=True,
            all_other_rgb_depth_and_sources_unchanged=True,target_ray_visibility_changed=True,mesh_changed=False,
            target_rgb_used=False,rgb_averaging=False,production_promoted=False,visual_status='pending',
            script_sha256=sha(__file__),hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print(view,'candidates',len(xx),'accepted',int(accept.sum()),'near old/new',oldnear.tolist(),newnear.tolist(),flush=True)


if __name__=='__main__':main()
