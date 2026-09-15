"""Replay fit residuals, verify bindings, and diagnose target-ray false coverage."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from study_canonical_face_prior import OUT,SOURCE,FRAME,deform
from study_multiview_face_prior import read,save,sha
from triangulate_face_prior import projection_matrices,quantiles
from probe_canonical_face_locality import boundary_edges

def main():
    import open3d as o3d
    from joint_temporal_texture import HELD_CAMERAS
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    from study_confidence_depth_prior import unproject
    checked=[]
    def check(path,expected):
        assert sha(path)==expected,str(path);checked.append(str(path))
    protocol=read(OUT/'protocol.json');obsmeta=read(OUT/'observations.json');review=read(OUT/'review/manifest.json');loc=read(OUT/'locality/result.json')
    for name,record in protocol['assets'].items():check(OUT/name,record['sha256'])
    for name in ['canonical','observations']:check(OUT/(name+'.npz'),protocol['canonical_npz_sha256'] if name=='canonical' else obsmeta['observations_sha256'])
    check(OUT/'protocol.json',obsmeta['protocol_sha256']);check(protocol['original_mesh'],protocol['original_mesh_sha256'])
    check(SOURCE/FRAME/'input.json',protocol['source_input_sha256']);check(SOURCE/'inference.json',protocol['source_inference_sha256']);check(SOURCE/'request.json',protocol['source_request_sha256'])
    check(Path(__file__).with_name('study_canonical_face_prior.py'),protocol['script_sha256']);check(Path(__file__).with_name('review_canonical_face_prior.py'),review['script_sha256']);check(Path(__file__).with_name('probe_canonical_face_locality.py'),loc['script_sha256'])
    for item in read(SOURCE/FRAME/'input.json')['inputs']:check(item['path'],item['sha256'])
    receipt=obsmeta['depth_receipt'];raw=read(receipt['transforms']);mapping={r['physical_camera']:r for r in raw['frames']}
    for name,digest in receipt['depth_sha256'].items():check(Path(receipt['dense'])/'stereo/depth_maps'/(mapping[name]['file_path']+'.geometric.bin'),digest)
    for item in review['files']+loc['files']:check(item['path'],item['sha256'])
    rows=obsmeta['cameras'];assert not ({r['physical_camera'] for r in rows}&HELD_CAMERAS)
    obs=np.load(OUT/'observations.npz');canon=np.load(OUT/'canonical.npz');validation=obs['validation_cameras'];assert validation.sum()==8
    expected=np.array([any(r['physical_camera'].startswith(p) for p in protocol['validation_prefixes']) for r in rows]);np.testing.assert_array_equal(validation,expected)
    matrices=projection_matrices(rows);replayed=0
    for arm in protocol['arms']:
        result=read(OUT/arm/'result.json');check(OUT/arm/'fit.npz',result['fit_sha256']);check(OUT/arm/'prior_only.ply',result['mesh_sha256'])
        f=np.load(OUT/arm/'fit.npz');v=deform(f['parameters'],canon['vertices'],canon['basis']);np.testing.assert_allclose(v,f['vertices'],atol=1e-14)
        uv=obs['landmark_uv'];li=obs['landmark_indices'];lc=obs['landmark_camera'];q=np.einsum('nij,nj->ni',matrices[lc],np.column_stack((v[li],np.ones(len(li)))))
        err=np.linalg.norm(q[:,:2]/q[:,2,None]-uv,axis=1);pe=np.sum((v[obs['anchor_indices']]-obs['anchor_points'])*obs['anchor_normals'],axis=1)
        np.testing.assert_allclose(err,f['landmark_errors'],atol=1e-10);np.testing.assert_allclose(pe,f['point_plane_errors'],atol=1e-12);replayed+=len(err)+len(pe)
        assert not (set(result['fit_camera_names'])&{r['physical_camera'] for r,k in zip(rows,validation) if k})
    old=o3d.io.read_triangle_mesh(protocol['original_mesh']);ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);oldscene=scene_for(ov,ot)
    v=np.load(OUT/'regularized8/fit.npz')['vertices'];scene=scene_for(v,canon['triangles']);btree=cKDTree(ov[np.unique(boundary_edges(ot))])
    parent='/mnt/data/dec5_elevated_camera_dynamic_150/request.json';spotfile='/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json';check(parent,loc['parent_request_sha256']);check(spotfile,loc['spot_audit_sha256'])
    cam=next(r['camera'] for r in read(parent)['inventory'] if r['frame_id']==FRAME);spot=next(r for r in read(spotfile)['selected_components'] if r['frame_id']==FRAME)
    d0,_,_=camera_depth(oldscene,cam);d1,_,_=camera_depth(scene,cam);x0,y0,x1,y1=spot['bbox_inclusive'];pvalid=np.rot90(np.isfinite(d0));mask=np.zeros_like(pvalid);mask[y0:y1+1,x0:x1+1]=~pvalid[y0:y1+1,x0:x1+1];mask=np.rot90(mask,-1)&np.isfinite(d1)
    yy,xx=np.nonzero(mask);points=unproject(cam,xx,yy,d1[yy,xx],offset=.5);closest=oldscene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))['points'].numpy();dist=np.linalg.norm(points-closest,axis=1);bd=btree.query(points)[0];passed=(dist<=.002)&(bd<=.003)
    np.savez_compressed(OUT/'target_ray_locality.npz',prior_points=points,old_surface_distance=dist,old_boundary_distance=bd,locality_pass=passed)
    im=Image.open(OUT/'locality/requested_hole_clay_native.png').copy();draw=ImageDraw.Draw(im);width=im.width//2
    for dx in [0,width]:draw.rectangle((dx+65,24+65,dx+65+x1-x0,24+65+y1-y0),outline='red',width=1)
    im.save(OUT/'requested_hole_marked_native.png')
    actual=sorted({Path(r['path']).name.rsplit('_',1)[0] for r in review['files']});assert len(actual)==5
    save(OUT/'audit.json',dict(status='passed',checked_bindings=len(checked),checked_paths=checked,replayed_residuals=replayed,
        actual_native_review_cameras=actual,frozen_review_camera_field_empty=True,
        note='Supplement corrects empty review_cameras bookkeeping without changing frozen producer or images.',
        target_prior_hits=len(points),target_prior_hits_passing_locality=int(passed.sum()),target_prior_hit_surface_distance=quantiles(dist),target_prior_hit_boundary_distance=quantiles(bd),
        independently_parent_reviewed=['G004_B005_1210FG_clay.png','M004_B005_12109O_clay.png'],
        visually_reviewed=[r['path'] for r in review['files']]+[str(OUT/'locality/requested_hole_clay_native.png')],
        no_new_mesh_added=True,heldout_used=False,original_sha256=sha(protocol['original_mesh']),script_sha256=sha(__file__),
        target_ray_npz_sha256=sha(OUT/'target_ray_locality.npz'),marked_crop_sha256=sha(OUT/'requested_hole_marked_native.png')))
    print('Verified',len(checked),'bindings;',replayed,'replayed residuals; target hits',len(points),'locality pass',int(passed.sum()))

if __name__=='__main__':main()
