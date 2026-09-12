"""Bake only added surface patches; preserve every retained original UV texel.

This opt-in export keeps the original hard-source atlas verbatim in the top
of a larger atlas. Only new chin/cylinder faces receive newly sampled texture.
Unsupported new texels use explicitly labeled within-part appearance completion,
not a fabricated claim of camera visibility or measured depth.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import open3d as o3d
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from joint_temporal_texture import ROOT,SOURCE,read,sha,atomic_json,load_frame,load_parameters,project,sample,bounded_warp,apply_response,display
from bake_joint_temporal_mesh import atlas_geometry,camera_depth,export_asset
from hard_surface_texture import select_surface_sources,gather_hard_rgb
from diffusion_mesh_repair import BASE,scene_for


def extend_uv(old_uv,patch_uv,old_size,patch_size):
    ow,oh=old_size;pw,ph=patch_size
    width=max(ow,pw);height=oh+16+ph
    old=np.array(old_uv,copy=True);old[:,0]*=ow/width;old[:,1]=1-(1-old[:,1])*oh/height
    patch=np.array(patch_uv,copy=True);patch[:,0]*=pw/width;patch[:,1]=1-(oh+16+(1-patch[:,1])*ph)/height
    return old,patch,(width,height)


def neutral_metal_samples(color):
    """Conservative material-color prior, not semantic or visibility evidence."""
    maximum=color.max(-1);minimum=color.min(-1)
    return (maximum>.25)&((maximum-minimum)<maximum*.23)


def bake(candidate,output,neutral_cylinder=False):
    if any(output.resolve().is_relative_to(p.resolve()) for p in [ROOT,SOURCE,BASE]):
        raise ValueError('Repair output must not overwrite immutable inputs or calibration')
    target=output/'frames'/'000973';target.mkdir(parents=True,exist_ok=True)
    base=dict(np.load(BASE/'atlas_geometry.npz'));base_ids=np.load(candidate/'base_face_ids.npy')
    mesh=o3d.io.read_triangle_mesh(str(candidate/'mesh.ply'));v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
    ops=read(candidate/'operations.json')
    if sha(candidate/'mesh.ply')!=ops['output_mesh_sha256']:raise ValueError('Candidate mesh checksum mismatch')
    if not np.array_equal(v[:len(base['vertices'])],base['vertices']):raise ValueError('Original vertices were moved')
    retained=base_ids>=0
    if not np.array_equal(t[retained],base['triangles'][base_ids[retained]]):raise ValueError('Original face mapping mismatch')
    if not retained[:retained.sum()].all():raise ValueError('Expected original faces followed by added patches')
    patch_faces=t[~retained];unique,inverse=np.unique(patch_faces,return_inverse=True)
    cylinder_count=ops.get('cylinder_triangles',0);is_cylinder=np.zeros(len(patch_faces),bool)
    if cylinder_count:is_cylinder[-cylinder_count:]=True
    patchmesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v[unique]),o3d.utility.Vector3iVector(inverse.reshape(-1,3)))
    o3d.io.write_triangle_mesh(str(target/'patch_input.ply'),patchmesh)
    patch=atlas_geometry(target/'patch_input.ply',target/'patch_atlas',resolution=1024)
    pw,ph=int(patch['width']),int(patch['height']);size=pw*ph
    data=load_frame(ROOT,'000973');rows=data['rows'];device=data['images'].device
    profile,static,residual=load_parameters(ROOT,'000973',device);gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    full_scene=scene_for(v,t)
    depths=[]
    for row in rows:
        d,_,_=camera_depth(full_scene,row);depths.append(np.where(np.isfinite(d),d,0))
    depth=torch.tensor(np.stack(depths)[:,None],device=device)
    tv=v[patch_faces];normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    camera_centers=np.array([r['transform_matrix'] for r in rows],np.float32)[:,:3,3]
    centroid=tv.mean(1);uv,z=project(centroid,rows)
    with torch.inference_mode():
        q=torch.tensor(uv[:,None],device=device);zq=torch.tensor(z,device=device);dq=sample(depth,q)[:,0,0]
        visible=(dq>0)&(zq>0)&((dq-zq).abs()<.0015*zq)
        visible&=(q[:,0,:,0]>2)&(q[:,0,:,0]<1917)&(q[:,0,:,1]>2)&(q[:,0,:,1]<1077)
        directions=camera_centers[:,None]-centroid;length=np.linalg.norm(directions,axis=-1);directions/=length[...,None]
        quality=np.abs((directions*normal).sum(-1))**8/length.clip(.01)**2*visible.cpu().numpy()
        quality=np.where(quality>=quality.max(0)*.12,quality,0)
        color=display(apply_response(sample(data['images'],q),profile)[:,:,0].permute(0,2,1).cpu().numpy(),gain)
        if neutral_cylinder:quality*=~is_cylinder[None]|neutral_metal_samples(color)
    labels,graph=select_surface_sources(color,quality,patch['triangles'],color_weight=.5)
    rgb=np.zeros((size,3),np.float32);support=np.zeros(size,np.uint8);source=np.full(size,255,np.uint8)
    with torch.inference_mode():
        for start in range(0,len(patch['pixels']),60000):
            stop=min(start+60000,len(patch['pixels']));ids=patch['face_ids'][start:stop];points=(tv[ids]*patch['barycentric'][start:stop,:,None]).sum(1)
            uv,z=project(points,rows);q=torch.tensor(uv[:,None],device=device);zq=torch.tensor(z,device=device)
            dq=sample(depth,q)[:,0,0];visible=(zq>0)&(dq>0)&((dq-zq).abs()<.0015*zq)
            visible&=(q[:,0,:,0]>2)&(q[:,0,:,0]<1917)&(q[:,0,:,1]>2)&(q[:,0,:,1]<1077)
            shifted=q+bounded_warp(static,residual,q);sd=sample(depth,shifted)[:,0,0];safe=(sd>0)&((sd-zq).abs()<.0025*zq)
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                bd=sample(depth,q.floor()+q.new_tensor([dx,dy]))[:,0,0]
                visible&=(bd>0)&((bd-zq).abs()<.003*zq)
                sd=sample(depth,shifted.floor()+q.new_tensor([dx,dy]))[:,0,0]
                safe&=(sd>0)&((sd-zq).abs()<.003*zq)
            shifted=torch.where(safe[:,None,:,None],shifted,q)
            directions=camera_centers[:,None]-points;length=np.linalg.norm(directions,axis=-1);directions/=length[...,None]
            quality=np.abs((directions*normal[ids]).sum(-1))**8/length.clip(.01)**2
            weights=torch.tensor(quality,device=device)*visible
            colors=apply_response(sample(data['images'],shifted),profile)[:,:,0].clamp_min(0)
            if neutral_cylinder:
                rr=colors*gain;rr=rr/(1+rr);dd=torch.where(rr<=.0031308,12.92*rr,1.055*rr.pow(1/2.4)-.055)
                maximum=dd.max(1).values;minimum=dd.min(1).values
                neutral=(maximum>.25)&((maximum-minimum)<maximum*.23)
                weights*=~torch.tensor(is_cylinder[ids],device=device)[None]|neutral
            chosen_rgb,chosen,_=gather_hard_rgb(colors,weights,torch.tensor(labels[ids],device=device))
            pixels=patch['pixels'][start:stop];rgb[pixels]=display(chosen_rgb.T.cpu().numpy(),gain)
            support[pixels]=(weights>0).sum(0).cpu().numpy().astype(np.uint8);source[pixels]=chosen.cpu().numpy().astype(np.uint8)
    # Local appearance prior is kept separate from true camera sample support.
    # Nearest copied texel, never a cross-camera average; only within chin or
    # cylinder material part, never from unrelated original face/hair triangles.
    parts=np.zeros(len(patch_faces),np.int8)
    if cylinder_count:parts[-cylinder_count:]=1
    completion=np.zeros(size,bool);pixels=patch['pixels'];yy,xx=np.divmod(pixels,pw)
    for part in np.unique(parts):
        in_part=parts[patch['face_ids']]==part;known=in_part&(support[pixels]>0);unknown=in_part&(support[pixels]==0)
        if unknown.any():
            if not known.any():raise ValueError('No camera-supported texture in local material part')
            tree=cKDTree(np.column_stack((xx[known],yy[known])));_,idx=tree.query(np.column_stack((xx[unknown],yy[unknown])))
            rgb[pixels[unknown]]=rgb[pixels[known][idx]];completion[pixels[unknown]]=True;source[pixels[unknown]]=254
    occupied=np.zeros(size,bool);occupied[pixels]=True;occupied=occupied.reshape(ph,pw)
    distance,nearest=distance_transform_edt(~occupied,return_indices=True);pad=(~occupied)&(distance<=8)
    patch_rgb=rgb.reshape(ph,pw,3);patch_rgb[pad]=patch_rgb[nearest[0][pad],nearest[1][pad]]
    original=np.asarray(Image.open(BASE/'texture_joint.png').convert('RGB'));oh,ow=original.shape[:2]
    old_uv,patch_uv,(width,height)=extend_uv(base['uv'],patch['uv'],(ow,oh),(pw,ph))
    texture=np.zeros((height,width,3),np.uint8);texture[:oh,:ow]=original
    texture[oh+16:,:pw]=np.rint(patch_rgb*255).clip(0,255).astype(np.uint8)
    if not np.array_equal(texture[:oh,:ow],original):raise ValueError('Original texture changed')
    Image.fromarray(texture).save(target/'texture_joint.png')
    combined={'vertices':v,'triangles':t,'mapping':np.r_[base['mapping'],unique[patch['mapping']]],
              'indices':np.concatenate((base['indices'][base_ids[retained]],patch['indices']+len(base['mapping']))),
              'uv':np.concatenate((old_uv,patch_uv)),'width':width,'height':height}
    if not np.array_equal(combined['mapping'][combined['indices']],t):raise ValueError('Combined UV inventory mismatch')
    np.savez_compressed(target/'atlas_geometry.npz',**combined)
    np.savez_compressed(target/'patch_texture_evidence.npz',support=support.reshape(ph,pw),source=source.reshape(ph,pw),completion=completion.reshape(ph,pw))
    np.save(target/'base_face_ids.npy',base_ids)
    export_asset(target,combined,'000973',None,tag='local_repaired')
    asset_manifest=read(target/'asset_manifest.json')
    asset_manifest.update(input_is_repaired_mesh=True,geometry_changed_vs_original=True,
                          geometry_changed_during_export=False)
    atomic_json(target/'asset_manifest.json',asset_manifest)
    shutil.copyfile(candidate/'mesh.ply',target/'mesh.ply');shutil.copyfile(candidate/'operations.json',target/'operations.json')
    atomic_json(target/'repair_bake_result.json',{'status':'baked_requires_visual_review','candidate_mesh_sha256':sha(candidate/'mesh.ply'),
                 'base_atlas_sha256':sha(BASE/'atlas_geometry.npz'),'base_texture_sha256':sha(BASE/'texture_joint.png'),
                 'parameters_sha256':sha(ROOT/'parameters.npz'),'exposure_sha256':sha(ROOT/'exposure.json'),
                 'original_texture_pixels_unchanged':True,'original_vertex_coordinates_unchanged':True,
                 'new_patch_triangles':len(patch_faces),'retained_original_triangles':int(retained.sum()),
                 'new_patch_texels':len(pixels),'appearance_prior_completed_texels':int(completion.sum()),
                 'appearance_prior':'nearest observed texel copied within the same new material part, not a measurement',
                 'real_camera_count':62,'synthetic_camera_rgb_baked':False,'graph':graph,
                 'neutral_cylinder_material_prior':neutral_cylinder,
                 'material_mask_is_not_visibility_evidence':True,
                 'scripts_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['bake_local_mesh_repair.py','local_mesh_repair.py','diffusion_mesh_repair.py']}})
    atomic_json(target/'complete.json',{'status':'artifact_valid_not_visually_approved',
                 'hashes':{str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file() and p.name!='complete.json'}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neutral-cylinder',action='store_true')
    a=p.parse_args();torch.set_num_threads(8);bake(a.candidate,a.output,a.neutral_cylinder)


if __name__=='__main__':main()
