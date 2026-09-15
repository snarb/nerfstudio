"""Post-fit locality and known jaw-hole reach; no fitted parameters or geometry edits."""
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from study_canonical_face_prior import OUT,SOURCE,FRAME,JAW,CHEEK
from study_multiview_face_prior import read,save,sha
from triangulate_face_prior import quantiles

def boundary_edges(t):
    e,c=np.unique(np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    return e[c==1]

def edge_distance(points,vertices,edges):
    a=vertices[edges[:,0]];d=vertices[edges[:,1]]-a
    q=points[:,None,:]-a;u=np.clip(np.sum(q*d,axis=2)/np.sum(d*d,axis=1),0,1)
    return np.linalg.norm(q-u[:,:,None]*d,axis=2).min(1)

def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    from study_confidence_depth_prior import unproject
    root=OUT/'locality';root.mkdir(exist_ok=False);protocol=read(OUT/'protocol.json')
    old=o3d.io.read_triangle_mesh(protocol['original_mesh']);old.compute_triangle_normals();ov=np.asarray(old.vertices);ot=np.asarray(old.triangles)
    scene=scene_for(ov,ot);btree=cKDTree(ov[np.unique(boundary_edges(ot))]);canonical=np.load(OUT/'canonical.npz');t=canonical['triangles'];edges=boundary_edges(t)
    v=np.load(OUT/'regularized8'/'fit.npz')['vertices'];prior=scene_for(v,t)
    # Uniform topology-only samples, not selected from the requested camera/ROI.
    lower=np.isin(t,JAW+CHEEK).any(1);pieces=v[t[lower]]
    for _ in range(3):
        a,b,c=pieces[:,0],pieces[:,1],pieces[:,2];ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
        pieces=np.concatenate([np.stack(z,axis=1) for z in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
    points=pieces.mean(1);hit=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)));closest=hit['points'].numpy();surface=np.linalg.norm(points-closest,axis=1);bd=btree.query(points)[0]
    normals=np.asarray(old.triangle_normals)[hit['primitive_ids'].numpy()];signed=np.sum((points-closest)*normals,axis=1);local=(surface<=.002)&(bd<=.003)
    np.savez_compressed(root/'dense_samples.npz',points=points,surface_distance=surface,boundary_distance=bd,signed_normal_offset=signed,locality_pass=local)
    # Frozen parent diagnostic camera and component are evaluation-only, after fit.
    parent='/mnt/data/dec5_elevated_camera_dynamic_150/request.json';spotfile='/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json'
    cam=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']==FRAME);spot=next(r for r in read(spotfile)['selected_components'] if r['frame_id']==FRAME)
    d0,ids0,_=camera_depth(scene,cam);d1,ids1,_=camera_depth(prior,cam);valid=np.isfinite(d0);priorvalid=np.isfinite(d1)
    x0,y0,x1,y1=spot['bbox_inclusive'];pvalid=np.rot90(valid);pmask=np.zeros_like(pvalid);pmask[y0:y1+1,x0:x1+1]=~pvalid[y0:y1+1,x0:x1+1]
    missing=np.rot90(pmask,k=-1);ring=binary_dilation(missing)&valid;yy,xx=np.nonzero(ring);rp=unproject(cam,xx,yy,d0[yy,xx],offset=.5)
    cp=prior.compute_closest_points(o3d.core.Tensor(rp.astype(np.float32)));cq=cp['points'].numpy();dist=np.linalg.norm(cq-rp,axis=1);norm=np.asarray(old.triangle_normals)[ids0[yy,xx]];sn=np.sum((cq-rp)*norm,axis=1);ed=edge_distance(cq,v,edges)
    np.savez_compressed(root/'requested_hole_boundary.npz',original_points=rp,prior_closest_points=cq,distance=dist,signed_original_normal_offset=sn,prior_open_edge_distance=ed,original_pixel_xy=np.column_stack((xx,yy)))
    crop=(x0-65,y0-65,x1+66,y1+66);images=[]
    for label,dd,ids,mesh in [('original',d0,ids0,old),('canonical prior only',d1,ids1,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t)))]:
        mesh.compute_triangle_normals();nn=np.asarray(mesh.triangle_normals);rgb=np.zeros((*dd.shape,3),np.uint8);ok=np.isfinite(dd);rgb[ok]=(60+170*np.abs(nn[ids[ok]]@np.array([.3,.4,.866])))[:,None]
        im=Image.fromarray(np.rot90(rgb)).crop(crop);images.append((label,im))
    panel=Image.new('RGB',(images[0][1].width*2,images[0][1].height+24));draw=ImageDraw.Draw(panel)
    for i,(label,im) in enumerate(images):panel.paste(im,(i*im.width,24));draw.text((i*im.width+2,4),label,fill='white')
    panel.save(root/'requested_hole_clay_native.png')
    obs=np.load(OUT/'observations.npz');fit=np.load(OUT/'regularized8'/'fit.npz');val=obs['validation_cameras'];lm=val[obs['landmark_camera']]&np.isin(obs['landmark_indices'],CHEEK);am=val[obs['anchor_camera']]&np.isin(obs['anchor_indices'],CHEEK)
    save(root/'result.json',dict(dense_topology_lower_face_samples=len(points),samples_passing_frozen_locality=int(local.sum()),local_sample_signed_normal_offset=quantiles(signed[local]),
        requested_hole_original_miss_pixels=int(missing.sum()),requested_hole_prior_hits=int((missing&priorvalid).sum()),
        requested_hole_boundary=dict(count=len(rp),prior_distance=quantiles(dist),signed_original_normal_offset=quantiles(sn),
            closest_prior_points_on_open_template_edge=int((ed<1e-6).sum()),within_frozen_surface_limit=int((dist<=.002).sum())),
        validation_cheek_pixels=quantiles(fit['landmark_errors'][lm]),validation_cheek_point_plane=quantiles(abs(fit['point_plane_errors'][am])),validation_cheek_other_votes=quantiles(obs['anchor_other_votes'][am]),
        parent_request_sha256=sha(parent),spot_audit_sha256=sha(spotfile),fit_summary_sha256=sha(OUT/'fit_summary.json'),
        files=[dict(path=str(p),sha256=sha(p)) for p in root.iterdir() if p.is_file()],script_sha256=sha(__file__),
        target_used_for_fit=False,original_geometry_changed=False,geometry_added=False,
        caution='Dense locality alone is not semantic or multiview missing-anatomy support; no candidate accepted. Signed offsets use original triangle winding normals.'))

if __name__=='__main__':main()
