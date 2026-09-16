"""Trace frozen cinematic nose-ridge samples into train RGB and exact mesh rays.

Diagnostic only: no retouching, held-out data or prediction changes. Continuous
ray visibility is evidence about this mesh, not ground-truth anatomy.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image,ImageDraw
from joint_temporal_texture import cameras,project,ROOT as COLOR,exr,display,apply_response
from study_multiview_face_prior import read,save,sha
from study_temporal_source_retention import BASE,ROOT as PREVIOUS
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from diagnose_gap_texture_admission import gates
from native_texture_footprint import sample_native,snap_centers
from calibrated_depth_witness import response_gains
from temporal_texture_view_prior import angle_weights
from view_consistent_source_quality import quality
from study_lipstick_instance_masks import portrait_xy

ROOT=Path('/mnt/data/dec5_nose_source_visibility_001123')
FRAME='001123'


def direct_visible(t, tolerance=1e-5):
    t=np.asarray(t)
    return np.isfinite(t)&(abs(t-1)<=tolerance)


def main():
    assert not ROOT.exists();ROOT.mkdir()
    diag=PREVIOUS/'nose_attribution';proof=read(diag/'result.json')
    for n,h in proof['outputs'].items():assert sha(diag/n)==h,n
    e=np.load(diag/'baseline.npz');xy=e['native_xy'];sources=e['source_ids'];N=len(xy)
    folder=BASE/'frames'/FRAME;r=read(folder/'result.json');q=read(BASE/'request.json')
    receipt=read(folder/'complete.json');assert receipt['request_sha256']==sha(BASE/'request.json')
    bindings={}
    for n,h in receipt['hashes'].items():assert sha(folder/n)==h,n;bindings[str(folder/n)]=h
    row=next(r for r in q['inventory'] if r['frame_id']==FRAME)
    assert sha(row['mesh'])==row['mesh_sha256'];bindings[row['mesh']]=row['mesh_sha256']
    rows,_,_=cameras(FRAME);assert [x['physical_camera'] for x in rows]==r['source_cameras']
    mesh=o3d.io.read_triangle_mesh(row['mesh']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
    scene=scene_for(v,t);d,ids,b=camera_depth(scene,r['camera'])
    np.testing.assert_array_equal(np.where(np.isfinite(d),d,0),np.load(folder/'target_depth.npz')['depth'])
    x,y=xy.T;face=ids[y,x];b=b[y,x];weights=np.c_[1-b.sum(1),b];points=(v[t[face]]*weights[:,:,None]).sum(1)
    uv,z=project(points,rows);centers=np.array([r['transform_matrix'] for r in rows],np.float32)[:,:3,3]
    direct=[]
    for center in centers:
        rays=np.c_[np.broadcast_to(center,points.shape),points-center].astype(np.float32)
        direct.append(scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy())
    direct=np.stack(direct);visible=direct_visible(direct)
    spec=row['source_masks'];maskroot=Path(spec['root'])
    for n,k in [('complete.json','complete_sha256'),('masks.npz','masks_sha256'),('cameras.json','cameras_sha256')]:
        assert sha(maskroot/n)==spec[k];bindings[str(maskroot/n)]=spec[k]
    names=read(maskroot/'cameras.json');masks=np.load(maskroot/'masks.npz')['masks']
    masks=np.stack([masks[names.index(r['physical_camera'])] for r in rows])
    with ThreadPoolExecutor(max_workers=4) as pool:depth=np.stack(list(pool.map(lambda cam:camera_depth(scene,cam)[0],rows)))
    depth=np.where(np.isfinite(depth)&(masks>0),depth,0)
    g=gates(depth,uv,z);assert g['final'][sources,np.arange(N)].all(),'Selected-source gates did not replay'
    normal=np.cross(v[t[face]][:,1]-v[t[face]][:,0],v[t[face]][:,2]-v[t[face]][:,0])
    normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    direction=centers[:,None]-points;distance=np.linalg.norm(direction,axis=2);direction/=distance[...,None]
    angle,_=angle_weights(rows,r['camera'],q['recipe']['target_angle_sigma_degrees'])
    weights=quality((direction*normal).sum(-1),distance,'incidence2')*angle[:,None]
    eligible=g['final']&visible;alternative=np.argmax(weights*eligible,axis=0)
    has=eligible.any(0);selected_t=direct[sources,np.arange(N)]
    chosen=sorted(set(sources.tolist()+alternative[has].tolist()))
    gains=response_gains(np.load(COLOR/'parameters.npz')['log_gain']);exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    manifest=next(r for r in q['source_rows'] if Path(r['source_dataset']).name==FRAME)
    expected={r['physical_camera']:r['sha256'] for r in manifest['source_images']}
    samples={};witnesses=[];(ROOT/'witnesses').mkdir()
    for ci in chosen:
        cam=rows[ci];name=cam['physical_camera'];p=Path(cam['file_path']);assert sha(p)==expected[name];bindings[str(p)]=sha(p)
        linear=exr(p);qt=snap_centers(torch.tensor(uv[ci:ci+1,None],device='cpu'))
        sampled=sample_native(torch.tensor(linear.transpose(2,0,1)[None]),qt)[0,:,0].T.numpy()*gains[ci]
        samples[ci]=np.rint(display(sampled.clip(0),exposure)*255).clip(0,255).astype(np.uint8)
        rgb=np.rint(display(linear*gains[ci],exposure)*255).clip(0,255).astype(np.uint8)
        portrait=np.rot90(rgb);coords=portrait_xy(uv[ci],1920)
        lo=np.maximum(np.floor(coords.min(0)).astype(int)-25,[0,0]);hi=np.minimum(np.ceil(coords.max(0)).astype(int)+26,[1080,1920])
        box=[*lo.tolist(),*hi.tolist()];plain=Image.fromarray(portrait).crop(box);marked=plain.copy();draw=ImageDraw.Draw(marked)
        for j,pixel in enumerate(coords-lo):
            color='magenta' if sources[j]==ci else 'cyan'
            px,py=map(float,pixel);draw.ellipse((px-1,py-1,px+1,py+1),outline=color)
        out=Image.new('RGB',(2*plain.width,plain.height+40));out.paste(plain,(0,40));out.paste(marked,(plain.width,40));draw=ImageDraw.Draw(out)
        draw.text((2,3),name,fill='white');draw.text((2,20),'magenta: selected source / cyan: other ridge points',fill='white')
        out.save(ROOT/'witnesses'/f'{name}.png');witnesses.append(dict(camera=name,source_index=ci,crop=box))
    reconstructed=np.stack([samples[int(ci)][j] for j,ci in enumerate(sources)])
    err=abs(reconstructed.astype(int)-e['rgb'].astype(int));assert err.max()<=1,err.max()
    for p in [BASE/'request.json',diag/'result.json',diag/'baseline.npz',COLOR/'parameters.npz',COLOR/'exposure.json',Path(__file__)]:bindings[str(p)]=sha(p)
    np.savez_compressed(ROOT/'evidence.npz',points=points,face_ids=face,native_xy=xy,portrait_xy=e['portrait_xy'],
        source_ids=sources,uv=uv,z=z,direct_t=direct,direct_visible=visible,
        raster_valid=g['final'],quality=weights,alternative=alternative,has_alternative=has,
        reconstructed_rgb=reconstructed,**{f'rgb_{i}':rgb for i,rgb in samples.items()},
        source_depth_interpolated=g['interpolated'],source_depth_taps=g['taps'])
    save(ROOT/'result.json',dict(frame=FRAME,sample_count=N,selected_continuously_visible=int(direct_visible(selected_t).sum()),
        selected_nearer_first_hit=int((selected_t<1-1e-5).sum()),selected_farther_first_hit=int((selected_t>1+1e-5).sum()),
        selected_t_quantiles=np.quantile(selected_t,[0,.5,1]).tolist(),alternative_count=int(has.sum()),
        selected_occluded_with_alternative=int((~direct_visible(selected_t)&has).sum()),rgb_replay_max_error=int(err.max()),
        witnesses=witnesses,source_cameras=[r['physical_camera'] for r in rows],input_hashes=bindings,
        evidence_sha256=sha(ROOT/'evidence.npz'),images={str(p):sha(p) for p in (ROOT/'witnesses').glob('*.png')},
        production_modified=False,prediction_modified=False,heldout_used=False,visual_status='pending',
        direct_ray_visibility_is_same_mesh_not_independent_ground_truth=True))
    print(read(ROOT/'result.json')|{'input_hashes':'retained','images':'retained'},flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with torch.inference_mode():main()
