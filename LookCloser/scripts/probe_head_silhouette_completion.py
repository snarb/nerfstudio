"""Train-only local head silhouette-envelope completion, not observed geometry.

All 62 masks constrain a new spatial surface rather than trim a Poisson shell.
Unknown FOV regions cannot create surface caps. Existing geometry is unchanged;
new local faces remain UNCHECKED until the measured-free-space guard and RGB gate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras
from probe_inset_head_completion import ROOT as INSET,FRAMES,MOVIE
from refine_measured_head_masks import ROOT as MASKS
from local_silhouette_volume import signed_pixels,remove_box_caps
from silhouette_domain_surface import stable_domain_faces
from study_jaw_repair_transfer import mask_votes
from study_confidence_depth_prior import REGIONS,region_masks
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

ROOT=Path('/mnt/data/dec5_head_silhouette_completion')
SETTINGS=dict(spacing=.00025,padding=.003,minimum_views=3,inward_margin_pixels=1.,
    maximum_surface_distance=.006,maximum_boundary_distance=.006,maximum_edge=.0015,
    min_head_x=-.03,minimum_silhouette_support=2,maximum_silhouette_outside=0,
    field_chunk_points=500000)


@torch.no_grad()
def sample_field(points,rows,fields,margin=1.,device='cuda'):
    """62-view signed-distance minimum with explicit 64-bit availability.

Positive margin erodes (unlike the old CPU helper's dilation convention).
Projection uses the same float32 camera poses as the reference helper, with
float64 arithmetic, followed by float32 bilinear texture sampling.
"""
    if not 1<=len(rows)<=62 or len(fields)!=len(rows):raise ValueError('Invalid view inventory')
    points=torch.as_tensor(np.asarray(points),dtype=torch.float64,device=device)
    minimum=torch.full((len(points),),float('inf'),device=device)
    count=torch.zeros(len(points),dtype=torch.int16,device=device)
    bits=torch.zeros(len(points),dtype=torch.int64,device=device)
    for ci,(row,field) in enumerate(zip(rows,fields)):
        pose=torch.as_tensor(np.asarray(row['transform_matrix'],np.float32),dtype=torch.float64,device=device)
        q=(points-pose[:3,3])@pose[:3,:3];z=-q[:,2]
        uv=torch.stack([row['fl_x']*q[:,0]/z+row['cx']-.5,-row['fl_y']*q[:,1]/z+row['cy']-.5],1)
        h,w=field.shape
        known=torch.isfinite(uv).all(1)&torch.isfinite(z)&(z>0)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
        xy=uv.to(torch.float32);grid=((xy+.5)*xy.new_tensor([2/w,2/h])-1).reshape(1,1,-1,2)
        tensor=field if torch.is_tensor(field) else torch.as_tensor(field,device=device)
        value=F.grid_sample(tensor.reshape(1,1,h,w),grid,align_corners=False,padding_mode='zeros').reshape(-1)-margin
        minimum=torch.minimum(minimum,torch.where(known,value,float('inf')))
        count+=known.to(torch.int16);bits|=known.to(torch.int64)<<ci
    # This sentinel never becomes a kept surface: stable-domain filtering below
    # rejects cells with insufficient or changing camera availability.
    minimum=torch.where(count>=3,minimum,-1.)
    return minimum.cpu().numpy(),count.cpu().numpy(),bits.cpu().numpy().astype(np.uint64)


def run(frame):
    out=ROOT/frame;out.mkdir(parents=True,exist_ok=False)
    config=read(INSET/frame/'request.json');assert sha(config['source_mesh'])==config['source_mesh_sha256']
    mr=read(MASKS/frame/'result.json')
    for p,h in mr['hashes'].items():assert sha(MASKS/frame/p)==h
    rows,_,metadata=cameras(frame);names=read(MASKS/frame/'cameras.json')
    assert len(rows)==len(set(names))==62 and set(names)=={r['physical_camera'] for r in rows}
    masks=np.load(MASKS/frame/'masks.npz')['masks'];masks=masks[[names.index(r['physical_camera']) for r in rows]]
    mesh=o3d.io.read_triangle_mesh(config['source_mesh']);ov,ot=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    head=ov[ov[:,0]>SETTINGS['min_head_x']];lower=head.min(0)-SETTINGS['padding'];upper=head.max(0)+SETTINGS['padding']
    spacing=SETTINGS['spacing'];shape=np.ceil((upper-lower)/spacing).astype(int)+1;upper=lower+(shape-1)*spacing
    count=int(np.prod(shape));assert count<30000000
    request=dict(frame=frame,settings=SETTINGS,rows=rows,lower=lower.tolist(),upper=upper.tolist(),shape=shape.tolist(),
        source_mesh=config['source_mesh'],source_mesh_sha256=config['source_mesh_sha256'],
        dependencies={str(p):sha(p) for p in [INSET/frame/'request.json',MASKS/frame/'request.json',MASKS/frame/'result.json',
            MASKS/frame/'masks.npz',MASKS/frame/'cameras.json',metadata,MOVIE/'request.json']},
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in
            [Path(__file__).name,'local_silhouette_volume.py','silhouette_domain_surface.py','joint_temporal_texture.py',
             'study_jaw_repair_transfer.py','bake_joint_temporal_mesh.py']},
        production_updated=False,heldout_used=False,target_used_for_geometry=False,surface_is_inferred_envelope=True,
        geometry_uses_rgb=False,source_texture_masks_changed=False)
    atomic_json(out/'request.json',request)
    with ThreadPoolExecutor(max_workers=4) as pool:fields=list(pool.map(signed_pixels,masks))
    np.savez_compressed(out/'silhouette_fields.npz',fields=np.stack(fields))
    gpu_fields=[torch.as_tensor(f,device='cuda') for f in fields]
    field=np.empty(count,np.float32);bits=np.empty(count,np.uint64);available=np.empty(count,np.uint8)
    started=time.monotonic()
    for start in range(0,count,SETTINGS['field_chunk_points']):
        end=min(start+SETTINGS['field_chunk_points'],count)
        ijk=np.stack(np.unravel_index(np.arange(start,end),shape),1)
        field[start:end],available[start:end],bits[start:end]=sample_field(lower+ijk*spacing,rows,gpu_fields,SETTINGS['inward_margin_pixels'])
        atomic_json(out/'progress.json',dict(stage='silhouette_field',done=end,total=count,seconds=time.monotonic()-started))
        if (start//SETTINGS['field_chunk_points'])%8==0:print(frame,'field',end,count,flush=True)
    field=field.reshape(shape);bits=bits.reshape(shape);available=available.reshape(shape)
    np.savez_compressed(out/'field.npz',field=field,bits=bits,available=available,lower=lower,spacing=spacing)
    assert (field>0).any() and (field<0).any()
    rv,rt,_,_=marching_cubes(field,0,spacing=(spacing,)*3);rv=rv.astype(np.float64)+lower
    box=remove_box_caps(rv,rt,lower,upper,spacing)
    domain=stable_domain_faces(rv,rt,lower,spacing,bits,SETTINGS['minimum_views'])
    original=scene_for(ov,ot);closest=original.compute_closest_points(o3d.core.Tensor(rv.astype(np.float32)))
    distance=np.linalg.norm(rv-closest['points'].numpy(),axis=1)
    edges,n=np.unique(np.sort(ot[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    boundary_distance=cKDTree(ov[np.unique(edges[n==1])]).query(rv)[0]
    local=(distance<=SETTINGS['maximum_surface_distance'])&(boundary_distance<=SETTINGS['maximum_boundary_distance'])&(rv[:,0]>SETTINGS['min_head_x'])
    edge=np.linalg.norm(rv[rt]-rv[rt[:,[1,2,0]]],axis=2).max(1)
    proposals=np.flatnonzero(box&domain&local[rt].all(1)&(edge<=SETTINGS['maximum_edge']))
    support,outside=mask_votes(rv,rt[proposals],rows,masks,[r['physical_camera'] for r in rows])
    keep=proposals[(support>=2)&(outside==0)]
    v=np.concatenate([ov,rv]);t=np.concatenate([ot,rt[keep]+len(ov)])
    new=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));new.compute_triangle_normals()
    assert o3d.io.write_triangle_mesh(str(out/'unchecked_mesh.ply'),new)
    np.savez_compressed(out/'evidence.npz',raw_vertices=rv,raw_triangles=rt,box_pass=box,domain_pass=domain,
        distance=distance,boundary_distance=boundary_distance,local_pass=local,edge=edge,proposals=proposals,
        mask_support=support,mask_outside=outside,retained_raw_triangle_ids=keep)
    native=next(r for r in rows if r['physical_camera']==REGIONS[frame]['camera'])
    moving=next(r['camera'] for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']==frame)
    newscene=scene_for(v,t);records=[]
    for name,cam in [('native_train',native),('moving',moving)]:
        old,_,_=camera_depth(original,cam);d,hit,_=camera_depth(newscene,cam)
        valid=np.isfinite(d);oldvalid=np.isfinite(old);gain=valid&~oldvalid
        rgb=np.zeros((*d.shape,3),np.uint8);shade=np.abs(np.asarray(new.triangle_normals)@np.array([.3,.4,.866]))
        rgb[valid]=(60+170*shade[hit[valid],None]).astype(np.uint8);rgb[gain]=[255,70,70]
        Image.fromarray(np.rot90(rgb).copy()).save(out/(name+'.png'))
        np.savez_compressed(out/(name+'_depth.npz'),baseline=old,candidate=d,triangle_ids=hit)
        record=dict(view=name,gained_depth=int(gain.sum()),lost_depth=int((oldvalid&~valid).sum()),
            visible_added=int((valid&(hit>=len(ot))).sum()),changed_existing_depth=int((oldvalid&valid&(abs(old-d)>1e-6)).sum()))
        if name=='native_train':record['coarse_hair_gained_depth']=int((gain&region_masks(frame)['hair']).sum())
        records.append(record)
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),raw_triangles=len(rt),
        box_rejected=int((~box).sum()),domain_rejected=int((~domain).sum()),local_proposals=len(proposals),admitted=len(keep),
        views=records,hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['request.json','result.json','progress.json']},
        production_updated=False,measured_guard_passed=False,visual_status='pending',counts_not_anatomical_metrics=True))
    print(frame,'screen',len(proposals),len(keep),records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',nargs='+',default=FRAMES,choices=FRAMES);a=p.parse_args()
    torch.set_num_threads(2)
    for frame in a.frames:run(frame)
