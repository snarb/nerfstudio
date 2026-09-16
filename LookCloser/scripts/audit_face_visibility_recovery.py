"""Independently reconstruct every changed intersection and train-RGB sample."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from study_multiview_face_prior import read,save,sha
from joint_temporal_texture import cameras,project,ROOT as COLOR,exr,display
from study_temporal_source_retention import BASE
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from diagnose_gap_texture_admission import gates
from native_texture_footprint import sample_native,snap_centers
from review_face_interior_visibility import support_at
from calibrated_depth_witness import response_gains
from temporal_texture_view_prior import angle_weights
from view_consistent_source_quality import quality


def main(root):
    result=read(root/'result.json'); request=read(root/'request.json'); frame=result['frame']
    assert result['request_sha256']==sha(root/'request.json')
    for name,h in result['hashes'].items(): assert sha(root/name)==h
    for p,h in request['input_hashes'].items(): assert sha(p)==h
    q=read(BASE/'request.json');record=next(x for x in q['inventory'] if x['frame_id']==frame)
    assert sha(record['mesh'])==record['mesh_sha256']
    folder=Path(result['baseline']);r=read(folder/'result.json'); rows,_,_=cameras(frame)
    assert [x['physical_camera'] for x in rows]==r['source_cameras']
    assert not set(r['source_cameras'])&{'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    m=o3d.io.read_triangle_mesh(record['mesh']); v=np.asarray(m.vertices,np.float32);t=np.asarray(m.triangles,np.uint32)
    scene=scene_for(v,t); d,ids,bary=camera_depth(scene,r['camera'])
    np.testing.assert_array_equal(np.where(np.isfinite(d),d,0),np.load(folder/'target_depth.npz')['depth'])
    e=np.load(root/'evidence.npz');px=e['pixels']; old=e['old_sources'];new=e['new_sources']; j=np.arange(len(px))
    faces=ids.ravel()[px];b=bary.reshape(-1,2)[px];w=np.c_[1-b.sum(1),b];points=(v[t[faces]]*w[:,:,None]).sum(1)
    np.testing.assert_array_equal(points,e['points']);uv,z=project(points,rows)
    np.testing.assert_array_equal(uv[new,j],e['uv'])
    with ThreadPoolExecutor(max_workers=4) as pool: depth=np.stack(list(pool.map(lambda row:camera_depth(scene,row)[0],rows)))
    spec=record['source_masks'];maskroot=Path(spec['root']);names=read(maskroot/'cameras.json');masks=np.load(maskroot/'masks.npz')['masks']
    for i,row in enumerate(rows):depth[i][masks[names.index(row['physical_camera'])]==0]=np.inf
    valid=gates(np.where(np.isfinite(depth),depth,0),uv,z)['final']
    skin=support_at(np.load(root/'face_masks.npz')['masks'],uv)
    votes=(valid&skin).sum(0)
    assert valid[old,j].all() and (~valid[new,j]).all() and skin[new,j].all() and (votes>=3).all()
    if request.get('old_selected_source_requires_face_semantics',True):assert skin[old,j].all()
    np.testing.assert_array_equal(votes,e['old_valid_face_votes'])
    centers=np.array([x['transform_matrix'] for x in rows],np.float32)[:,:3,3]
    normal=np.cross(v[t[faces]][:,1]-v[t[faces]][:,0],v[t[faces]][:,2]-v[t[faces]][:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    direction=centers[:,None]-points;length=np.linalg.norm(direction,axis=2);direction/=length[...,None]
    angles,_=angle_weights(rows,r['camera'],q['recipe']['target_angle_sigma_degrees'])
    weights=quality((direction*normal).sum(-1),length,'incidence2')*angles[:,None]
    np.testing.assert_array_equal(weights[old,j],e['old_quality']);np.testing.assert_array_equal(weights[new,j],e['new_quality'])
    assert (weights[new,j]>weights[old,j]).all()
    rays=np.c_[centers[new],points-centers[new]].astype(np.float32)
    direct=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    assert np.isfinite(direct).all() and (abs(direct-1)<=1e-5).all()
    # Bind color state to the independent pre-experiment diagnostic receipt.
    diag=read('/mnt/data/dec5_nose_source_visibility_001123/result.json')
    for p in [COLOR/'parameters.npz',COLOR/'exposure.json']:assert sha(p)==diag['input_hashes'][str(p)]
    gain=response_gains(np.load(COLOR/'parameters.npz')['log_gain']);exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    out=np.array(Image.open(root/'prediction_native.png')).reshape(-1,3)[px];replayed=np.zeros_like(out)
    for ci in np.unique(new):
        use=np.flatnonzero(new==ci);path=rows[ci]['file_path']; assert sha(path)==result['source_rgb_hashes'][path]
        rgb=exr(path);qt=snap_centers(torch.tensor(uv[ci:ci+1,None,use],device='cuda'))
        values=sample_native(torch.tensor(rgb.transpose(2,0,1)[None],device='cuda'),qt)[0,:,0].T.cpu().numpy()*gain[ci]
        replayed[use]=np.rint(display(values.clip(0),exposure)*255).clip(0,255).astype(np.uint8)
    np.testing.assert_array_equal(out,replayed)
    save(root/'independent_audit.json',dict(frame=frame,changed_points_replayed=len(px),
        exact_intersections=True,exact_source_uv=True,original_visibility_replayed=True,
        skin_consensus_replayed=True,quality_replayed=True,exact_source_rgb=True,
        ray_t_min=float(direct.min()),ray_t_max=float(direct.max()),geometry_unchanged=True,
        all_target_depths_exact=True,heldout_used=False,production_promoted=False,
        script_sha256=sha(__file__),request_sha256=sha(root/'request.json'),result_sha256=sha(root/'result.json'),
        mesh_sha256=sha(record['mesh']),color_hashes={str(p):sha(p) for p in [COLOR/'parameters.npz',COLOR/'exposure.json']}))
    print('independent audit passed',root,len(px),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+');a=p.parse_args()
    torch.set_num_threads(2)
    with torch.inference_mode():
        for root in a.roots:main(root)
