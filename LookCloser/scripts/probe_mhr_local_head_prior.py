"""Observed-anchor anatomical support and requested-hole boundary diagnostics."""
import argparse
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from study_mhr_local_head_prior import OUT,FRAME
from study_multiview_face_prior import read,save,sha
from triangulate_face_prior import quantiles
from probe_canonical_face_locality import boundary_edges

def main(arms):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    from study_confidence_depth_prior import unproject
    root=OUT/('probe_'+'_'.join(arms));root.mkdir(exist_ok=False);proto=read(OUT/'protocol.json');initial=np.load(OUT/'initial.npz');obs=np.load(OUT/'anchors.npz');val=obs['validation'][obs['camera']]
    old=o3d.io.read_triangle_mesh(proto['original_mesh']);old.compute_triangle_normals();ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);scene=scene_for(ov,ot);tree=cKDTree(ov[np.unique(boundary_edges(ot))])
    parent='/mnt/data/dec5_elevated_camera_dynamic_150/request.json';spots='/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json';cam=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']==FRAME);spot=next(r for r in read(spots)['selected_components'] if r['frame_id']==FRAME)
    d0,ids0,_=camera_depth(scene,cam);valid=np.isfinite(d0);pv=np.rot90(valid);x0,y0,x1,y1=spot['bbox_inclusive'];pm=np.zeros_like(pv);pm[y0:y1+1,x0:x1+1]=~pv[y0:y1+1,x0:x1+1];missing=np.rot90(pm,-1);ring=binary_dilation(missing)&valid;yy,xx=np.nonzero(ring);rp=unproject(cam,xx,yy,d0[yy,xx],offset=.5);norm=np.asarray(old.triangle_normals)[ids0[yy,xx]]
    images=[];records=[];crop=(x0-65,y0-65,x1+66,y1+66)
    def clay(label,depth,ids,mesh):
        mesh.compute_triangle_normals();nn=np.asarray(mesh.triangle_normals);rgb=np.zeros((*depth.shape,3),np.uint8);ok=np.isfinite(depth);rgb[ok]=(60+170*abs(nn[ids[ok]]@np.array([.3,.4,.866])))[:,None];images.append((label,Image.fromarray(np.rot90(rgb)).crop(crop)))
    clay('original',d0,ids0,old)
    for arm in arms:
        f=np.load(OUT/arm/'fit.npz');v=f['vertices'];t=initial['triangles'];subtri=t[(initial['neutral'][t,1]>140).all(1)];sub=scene_for(v,subtri);cp=sub.compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)));uv=cp['primitive_uvs'].numpy();bary=np.column_stack((1-uv.sum(1),uv));neutral=(initial['neutral'][subtri[cp['primitive_ids'].numpy()]]*bary[:,:,None]).sum(1)
        dist=np.linalg.norm(cp['points'].numpy()-obs['points'],axis=1);regions={'lower_face_145_153':(neutral[:,1]>=145)&(neutral[:,1]<153)&(neutral[:,2]>=0),'neck_135_145':(neutral[:,1]>=135)&(neutral[:,1]<145),'upper_face_ge153':neutral[:,1]>=153};groups={}
        for label,m in regions.items():
            groups[label]=dict(train_associated=int((~val&m&(dist<=.006)).sum()),validation=int((val&m).sum()),validation_distance=quantiles(dist[val&m]),validation_point_plane=quantiles(abs(f['plane'][val&m])),other_votes=quantiles(obs['other_votes'][m]))
        prior=scene_for(v,t);cp=prior.compute_closest_points(o3d.core.Tensor(rp.astype(np.float32)))['points'].numpy();delta=cp-rp;rd=np.linalg.norm(delta,axis=1);signed=np.sum(delta*norm,axis=1);d1,ids1,_=camera_depth(prior,cam);hit=missing&np.isfinite(d1);hy,hx=np.nonzero(hit);hp=unproject(cam,hx,hy,d1[hy,hx],offset=.5);close=scene.compute_closest_points(o3d.core.Tensor(hp.astype(np.float32)))['points'].numpy();sd=np.linalg.norm(hp-close,axis=1);bd=tree.query(hp)[0];passed=(sd<=.002)&(bd<=.003)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));clay(arm,d1,ids1,mesh)
        np.savez_compressed(root/(arm+'.npz'),anchor_associated_neutral=neutral,anchor_distance=dist,rim_points=rp,rim_prior_points=cp,rim_distance=rd,rim_signed_offset=signed,target_prior_points=hp,target_surface_distance=sd,target_boundary_distance=bd,target_locality_pass=passed)
        records.append(dict(arm=arm,anatomical_associations=groups,raw_candidate_groups_are_not_anatomical_ground_truth=True,
            target_original_misses=int(missing.sum()),target_prior_hits=int(hit.sum()),target_hits_passing_locality=int(passed.sum()),target_surface_distance=quantiles(sd),target_boundary_distance=quantiles(bd),
            rim_distance=quantiles(rd),rim_signed_original_normal_offset=quantiles(signed),rim_within_surface_limit=int((rd<=.002).sum()),rim_count=len(rd),fit_sha256=sha(OUT/arm/'fit.npz')))
    w,h=images[0][1].size;panel=Image.new('RGB',(w*len(images),h+24));draw=ImageDraw.Draw(panel)
    for i,(label,im) in enumerate(images):panel.paste(im,(i*w,24));draw.text((i*w+2,4),label,fill='white');draw.rectangle((i*w+65,89,i*w+65+x1-x0,89+y1-y0),outline='red',width=1)
    panel.save(root/'requested_hole_clay_native.png');save(root/'result.json',dict(arms=records,files=[dict(path=str(p),sha256=sha(p)) for p in root.iterdir() if p.is_file()],script_sha256=sha(__file__),parent_sha256=sha(parent),spot_sha256=sha(spots),target_used_for_fit=False,original_geometry_changed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['similarity','head20']);a=p.parse_args();main(a.arms)
