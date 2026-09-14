"""Explain real-target-camera texture eligibility without changing rendering."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
import open3d as o3d
import torch
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,project,sample
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from temporal_texture_view_prior import angle_weights
import study_forearm_plane_transfer_v3 as prior


def run(root,frame,output):
    torch.set_num_threads(2);prior.configure();name=prior.v2.v1.NAMES[1]
    dest=root/'rgb'/frame/name/'guarded';folder=dest/'frames'/frame
    request=read(dest/'request.json');receipt=read(folder/'complete.json')
    if sha(dest/'request.json')!=receipt['request_sha256']:raise ValueError('Changed target request')
    for p,h in receipt['hashes'].items():
        if sha(folder/p)!=h:raise ValueError('Changed rendered data')
    spec=request['inventory'][0];rows,_,_=cameras(frame);own=next(i for i,r in enumerate(rows) if r['physical_camera']==name)
    mesh=o3d.io.read_triangle_mesh(spec['mesh'])
    if sha(spec['mesh'])!=spec['mesh_sha256']:raise ValueError('Changed mesh')
    v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32);tv=v[t];centroid=tv.mean(1)
    normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    scene=scene_for(v,t);ms=spec['source_masks'];maskroot=Path(ms['root'])
    if sha(maskroot/'masks.npz')!=ms['masks_sha256']:raise ValueError('Changed source masks')
    lookup=dict(zip(read(maskroot/'cameras.json'),np.load(maskroot/'masks.npz')['masks']))
    def depth(row):
        d,_,_=camera_depth(scene,row);d[lookup[row['physical_camera']]==0]=np.inf
        return np.where(np.isfinite(d),d,0)
    with ThreadPoolExecutor(max_workers=4) as pool:depths=torch.tensor(np.stack(list(pool.map(depth,rows)))[:,None],device='cuda')
    uv,z=project(centroid,rows);q=torch.tensor(uv[:,None],device='cuda');zq=torch.tensor(z,device='cuda')
    d=sample(depths,q)[:,0,0];visible=(zq>0)&(d>0)&((d-zq).abs()<.0015*zq)
    visible&=(q[:,0,:,0]>2)&(q[:,0,:,0]<1917)&(q[:,0,:,1]>2)&(q[:,0,:,1]<1077)
    centers=np.array([r['transform_matrix'] for r in rows],np.float32)[:,:3,3]
    direction=centers[:,None]-centroid;length=np.linalg.norm(direction,axis=-1);direction/=length[...,None]
    quality=np.abs((direction*normal).sum(-1))**8/length.clip(.01)**2*visible.cpu().numpy()
    admitted=np.where(quality>=quality.max(0)*.12,quality,0)
    weights,angles=angle_weights(rows,spec['camera'],request['recipe']['target_angle_sigma_degrees'])
    early=quality*weights[:,None];late=admitted*weights[:,None]
    labels=np.load(folder/'face_source_labels.npy')
    td,ids,_=camera_depth(scene,spec['camera']);td[lookup[name]==0]=np.inf
    saved=np.load(folder/'target_depth.npz')['depth']
    if not np.allclose(np.where(np.isfinite(td),td,0),saved,atol=1e-6):raise ValueError('Target replay differs')
    skin=prior.v2.v1.masks(frame)[name];hit=skin&np.isfinite(td);f=ids[hit]
    lost=visible.cpu().numpy()[own]&(admitted[own]==0)
    summary=dict(skin_pixels=int(skin.sum()),geometry_pixels=len(f),own_centroid_visible_pixels=int(visible.cpu().numpy()[own,f].sum()),
        own_removed_by_incidence_prefilter_pixels=int(lost[f].sum()),
        own_late_prior_best_pixels=int((late.argmax(0)[f]==own).sum()),own_early_prior_best_pixels=int((early.argmax(0)[f]==own).sum()),
        own_saved_face_label_pixels=int((labels[f]==own).sum()),target_own_camera_angle_degrees=float(angles[own]))
    image=np.asarray(Image.open(folder/'prediction_native.png').convert('RGB'));overlay=image.copy()
    selected=np.zeros(skin.shape,bool);selected[hit]=lost[f];overlay[selected]=[255,30,30]
    panel=Image.new('RGB',(1080,550));draw=ImageDraw.Draw(panel)
    for i,im in enumerate([image,overlay]):panel.paste(Image.fromarray(np.rot90(im)).crop((0,1400,540,1920)),(540*i,30))
    draw.text((4,5),'current train-view render',fill='white');draw.text((544,5),'RED: own source visible but incidence-prefiltered',fill='white')
    out=output/frame;out.mkdir(parents=True,exist_ok=True);panel.save(out/'admission_native.png')
    atomic_json(out/'result.json',dict(frame=frame,summary=summary,script_sha256=sha(__file__),render_receipt_sha256=sha(folder/'complete.json'),
        panel_sha256=sha(out/'admission_native.png'),renderer_changed=False,geometry_changed=False,
        counterfactual_scope='rank only, not a new RGB render; centroid eligibility is not final pixel visibility'))
    print(frame,summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_color_qualified_curved'))
    p.add_argument('--frame',required=True);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_texture_admission'));a=p.parse_args();run(a.root,a.frame,a.output)
