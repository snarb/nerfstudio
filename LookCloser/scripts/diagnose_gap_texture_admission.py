"""Replay every source visibility gate at newly black moving-view gap pixels.

No RGB repair: distinguishes source-mask veto, mesh footprint, incidence and
measured-depth evidence at the actual new surface intersections.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from study_multiview_face_prior import read,save,sha
from study_train_gap_subfaces import ROOT as STUDY,FRAME
from joint_temporal_texture import cameras,project
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from native_texture_footprint import snap_centers,relevant_tap,sample_native
from temporal_texture_view_prior import angle_weights
from view_consistent_source_quality import quality
from study_confidence_depth_prior import load_real,project_integer
from review_full_block_transfer import ROOT as DEPTH_ROOT
from prune_measured_free_surface import near_tap_evidence

ROOT=Path('/mnt/data/dec5_gap_texture_admission')


def gates(depth,uv,z):
    q=snap_centers(torch.tensor(uv[:,None],device='cuda'));zq=torch.tensor(z,device='cuda')
    tensor=torch.tensor(depth[:,None],device='cuda')
    d=sample_native(tensor,q)[:,0,0]
    border=(q[:,0,:,0]>2)&(q[:,0,:,0]<1917)&(q[:,0,:,1]>2)&(q[:,0,:,1]<1077)
    center=(zq>0)&(d>0)&((d-zq).abs()<.0015*zq)&border
    taps=[];valid=[];relevant=[]
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        value=sample_native(tensor,q.floor()+q.new_tensor([dx,dy]))[:,0,0]
        rel=relevant_tap(q,dx,dy)
        good=((value>0)&((value-zq).abs()<.003*zq))|~rel
        taps.append(value.cpu().numpy());valid.append(good.cpu().numpy());relevant.append(rel.cpu().numpy())
    result=dict(interpolated=d.cpu().numpy(),border=border.cpu().numpy(),center=center.cpu().numpy(),
        taps=np.stack(taps,2),tap_valid=np.stack(valid,2),tap_relevant=np.stack(relevant,2),
        snapped_uv=q[:,0].cpu().numpy())
    result['final']=result['center']&result['tap_valid'].all(2)
    del tensor;return result


def main():
    assert not ROOT.exists();ROOT.mkdir()
    before=STUDY/'refined'/FRAME/'rgb/moving';after=STUDY/'carved'/FRAME/'rgb/moving'
    q=read(after/'request.json');r=read(after/'frames'/FRAME/'result.json')
    bindings={}
    for folder in [before,after]:
        complete=read(folder/'frames'/FRAME/'complete.json')
        assert complete['request_sha256']==sha(folder/'request.json')
        for name,h in complete['hashes'].items():
            p=folder/'frames'/FRAME/name;assert sha(p)==h;bindings[str(p)]=h
        bindings[str(folder/'request.json')]=sha(folder/'request.json')
    for name,h in q['script_hashes'].items():assert sha(Path(__file__).with_name(name))==h
    recipe=q['recipe'];assert not recipe['static_registration'] and recipe['source_incidence_power']==2
    assert sha(r['mesh_path'])==r['mesh_sha256'];bindings[r['mesh_path']]=r['mesh_sha256']
    record=q['inventory'][0];spec=record['source_masks'];maskroot=Path(spec['root'])
    for name,key in [('complete.json','complete_sha256'),('masks.npz','masks_sha256'),('cameras.json','cameras_sha256')]:
        assert sha(maskroot/name)==spec[key];bindings[str(maskroot/name)]=spec[key]
    masknames=read(maskroot/'cameras.json');masks=np.load(maskroot/'masks.npz')['masks']
    rows,_,_=cameras(FRAME);assert [r['physical_camera'] for r in rows]==r['source_cameras']
    masks=np.stack([masks[masknames.index(row['physical_camera'])] for row in rows])
    a=np.array(Image.open(before/'frames'/FRAME/'prediction_native.png'))
    b=np.array(Image.open(after/'frames'/FRAME/'prediction_native.png'))
    black=(a.max(2)>0)&(b.max(2)==0);ys,xs=np.nonzero(black);assert len(xs)==5
    sources=np.array(Image.open(after/'frames'/FRAME/'source_ids.png'));assert (sources[black]==255).all()
    mesh=o3d.io.read_triangle_mesh(r['mesh_path']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
    scene=scene_for(v,t);d,ids,bary=camera_depth(scene,r['camera'])
    rendered=np.load(after/'frames'/FRAME/'target_depth.npz')['depth']
    np.testing.assert_array_equal(np.where(np.isfinite(d),d,0),rendered)
    weights=np.c_[1-bary[black].sum(1),bary[black]];faces=ids[black];points=(v[t[faces]]*weights[:,:,None]).sum(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        depth=np.stack(list(pool.map(lambda row:camera_depth(scene,row)[0],rows)))
    depth=np.where(np.isfinite(depth),depth,0);uv,z=project(points,rows)
    raw=gates(depth,uv,z);masked=gates(np.where(masks>0,depth,0),uv,z)
    tv=v[t[faces]];normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    centers=np.array([row['transform_matrix'] for row in rows],np.float32)[:,:3,3]
    direction=centers[:,None]-points;length=np.linalg.norm(direction,axis=-1);direction/=length[...,None]
    angle,angles=angle_weights(rows,r['camera'],recipe['target_angle_sigma_degrees'])
    qs=quality((direction*normal).sum(-1),length,'incidence2')*angle[:,None]
    assert not ((qs>0)&masked['final']).any(),'Production black source selection does not replay'
    realrows,realdepths,receipt=load_real(DEPTH_ROOT,FRAME)
    assert [r['physical_camera'] for r in realrows]==[r['physical_camera'] for r in rows]
    near=[];measured=[]
    for row,rd in zip(realrows,realdepths):
        u,zz=project_integer(row,points);near.append(near_tap_evidence(rd,u,zz,radius=0))
        xy=np.rint(u).astype(int);ok=(xy>=0).all(1)&(xy<[1920,1080]).all(1)&(zz>0)
        val=np.full(len(points),np.nan);j=np.flatnonzero(ok);val[j]=rd[xy[j,1],xy[j,0]]-zz[j];measured.append(val)
    np.savez_compressed(ROOT/'evidence.npz',points=points,landscape_xy=np.c_[xs,ys],face_ids=faces,
        uv=uv,z=z,quality=qs,angles=angles,measured_near=np.stack(near),measured_delta=np.stack(measured),
        **{'raw_'+k:v for k,v in raw.items()},**{'masked_'+k:v for k,v in masked.items()})
    pixel_records=[]
    for j,(x,y) in enumerate(zip(xs,ys)):
        pixel_records.append(dict(landscape_xy=[int(x),int(y)],portrait_xy=[int(y),int(1919-x)],
            raw_center_views=int(raw['center'][:,j].sum()),raw_final_views=int(raw['final'][:,j].sum()),
            masked_center_views=int(masked['center'][:,j].sum()),masked_final_views=int(masked['final'][:,j].sum()),
            measured_near_views=int(np.stack(near)[:,j].sum()),
            raw_valid_cameras=[rows[i]['physical_camera'] for i in np.flatnonzero(raw['final'][:,j])],
            center_valid_cameras=[rows[i]['physical_camera'] for i in np.flatnonzero(masked['center'][:,j])]))
    save(ROOT/'result.json',dict(pixels=pixel_records,source_cameras=[r['physical_camera'] for r in rows],
        source_mask_spec=spec,depth_receipt=receipt,input_hashes=bindings,
        script_sha256=sha(__file__),evidence_sha256=sha(ROOT/'evidence.npz'),prediction_changed=False,
        production_changed=False,mask_and_mesh_gates_replayed=True,zero_registration=True))
    print(pixel_records,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with torch.inference_mode():main()
