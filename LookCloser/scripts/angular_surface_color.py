#!/usr/bin/env python3
"""Train-only, mesh-attached first-order angular RGB gain correction.

Fit log RGB differences versus viewing direction, regularized along mesh edges.
Rendering corrects the color of ONE reprojected source toward the requested view;
there is no source RGB averaging, new geometry, neural field or held-out RGB fit.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256


def calibrated_camera_rgb(rgb, row, model='rgb'):
    """Use the same camera response convention during fitting and rendering."""
    from patchmatch_color_calibration import apply_camera_gain
    if model=='rgb':return apply_camera_gain(rgb,row['rgb_gain'])
    if model=='spatial':return apply_camera_gain(rgb,row['exposure_gain'],row['spatial_log_gain_grid'])
    raise ValueError('Angular calibration supports rgb or spatial camera response')


def validate_camera_response(manifest, model):
    if model not in ('rgb','spatial') or manifest.get('camera_color_model','rgb')!=model:
        raise ValueError('Angular field camera response differs from renderer calibration mode')


def fit_angular_coefficients(log_rgb,directions,valid,vertices,triangles,held,*,smoothness=20.,ridge=.01,iterations=512):
    """Eliminate per-vertex albedo intercepts; solve a mesh-coupled linear system."""
    if log_rgb.shape!=directions.shape or valid.shape!=log_rgb.shape[:2] or log_rgb.shape[-1]!=3:
        raise ValueError('Expected CN3 colors/directions and CN visibility')
    if not bool(torch.isfinite(log_rgb).all()) or not bool(torch.isfinite(directions).all()) or smoothness<=0 or ridge<=0:
        raise ValueError('Invalid angular color inputs or regularization')
    weight=valid.float()*(~held)[None]
    total=weight.sum(0).clamp_min(1)
    mean_d=(directions*weight[...,None]).sum(0)/total[:,None]
    mean_rgb=(log_rgb*weight[...,None]).sum(0)/total[:,None]
    d=directions-mean_d;rgb=log_rgb-mean_rgb
    covariance=torch.einsum('cn,cnk,cnl->nkl',weight,d,d)
    rhs=torch.einsum('cn,cnk,cnl->nkl',weight,d,rgb)
    edge=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    edge=np.unique(np.sort(edge,axis=1),axis=0)
    a=torch.as_tensor(edge[:,0],device=log_rgb.device);b=torch.as_tensor(edge[:,1],device=log_rgb.device)
    length=np.linalg.norm(vertices[edge[:,0]]-vertices[edge[:,1]],axis=-1)
    ew=torch.as_tensor(np.clip((.0005/np.maximum(length,1e-7))**2,.05,20),device=log_rgb.device,dtype=log_rgb.dtype)*smoothness
    degree=torch.zeros(len(vertices),device=log_rgb.device).index_add_(0,a,ew).index_add_(0,b,ew)
    eye=torch.eye(3,device=log_rgb.device)[None]
    diagonal=covariance+(degree+ridge)[:,None,None]*eye
    inverse=torch.linalg.inv(diagonal)
    def matvec(x):
        out=torch.bmm(diagonal,x)
        out.index_add_(0,a,-ew[:,None,None]*x[b])
        out.index_add_(0,b,-ew[:,None,None]*x[a])
        return out
    def dot(x,y):return (x*y).sum((0,1),keepdim=True)
    x=torch.zeros_like(rhs);r=rhs.clone();z=torch.bmm(inverse,r);p=z.clone();rz=dot(r,z)
    norm=dot(rhs,rhs).sqrt().clamp_min(1e-12)
    for iteration in range(1,iterations+1):
        ap=matvec(p);alpha=rz/dot(p,ap).clamp_min(1e-30)
        x=x+alpha*p;r=r-alpha*ap;z=torch.bmm(inverse,r)
        next_rz=dot(r,z);p=z+(next_rz/rz.clamp_min(1e-30))*p;rz=next_rz
        if iteration%16==0 and bool((dot(r,r).sqrt()/norm<1e-4).all()):break
    residual=matvec(x)-rhs;relative=dot(residual,residual).sqrt()/norm
    return x,{'iterations':iteration,'max_relative_residual':float(relative.max()),'converged':bool((relative<.001).all()),
              'smoothness':smoothness,'ridge':ridge,'fit_vertices':int(((valid&~held).sum(0)>1).sum()),
              'held_vertices':int(held.sum()),'edge_count':len(edge),'direction_features':'unit_world_direction_xyz'}


class AngularSurfaceColor:
    def __init__(self,manifest_path,mesh_path):
        import open3d as o3d
        self.manifest=json.loads(Path(manifest_path).read_text());self.o3d=o3d
        if self.manifest['mesh_sha256']!=sha256(mesh_path) or self.manifest['uses_eval_rgb'] is not False:
            raise ValueError('Angular model mesh mismatch or eval RGB leakage')
        coefficients=Path(manifest_path).parent/self.manifest['coefficients']
        if sha256(coefficients)!=self.manifest['coefficients_sha256']:raise ValueError('Angular coefficient checksum mismatch')
        self.coefficients=np.load(coefficients)['coefficients']
        mesh=o3d.io.read_triangle_mesh(str(mesh_path));self.triangles=np.asarray(mesh.triangles)
        if self.coefficients.shape!=(len(mesh.vertices),3,3):raise ValueError('Angular field vertex inventory mismatch')
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=8)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def sample(self,world,support):
        shape=world.shape[:-1];points=world[support].astype(np.float32)
        out=np.zeros((*shape,3,3),np.float32)
        if not len(points):return out
        nearest=self.scene.compute_closest_points(self.o3d.core.Tensor(points))
        ids=nearest['primitive_ids'].numpy();uv=nearest['primitive_uvs'].numpy()
        weights=np.c_[1-uv.sum(-1),uv]
        values=(self.coefficients[self.triangles[ids]]*weights[:,:,None,None]).sum(1)
        distance=np.linalg.norm(points-nearest['points'].numpy(),axis=-1)
        values[distance>.001]=0  # no angular extrapolation far beyond measured mesh
        out[support]=values
        return out


def angular_log_gain(coefficients,world,target_center,source_center):
    to_target=torch.nn.functional.normalize(target_center-world,dim=-1)
    to_source=torch.nn.functional.normalize(source_center-world,dim=-1)
    return torch.einsum('...k,...kc->...c',to_target-to_source,coefficients).clamp(-math.log(2),math.log(2))


def main():
    import open3d as o3d
    from render_patchmatch_camera_path import normalize_frame
    from render_mesh_image_blend import load_rgb,grid_sample
    from patchmatch_color_calibration import decode_exposed_linear,encode_exposed_linear
    from mesh_texture_visibility import MeshVisibility
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','mesh','mesh-metadata','camera-color-calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoothness',type=float,default=20.);p.add_argument('--ridge',type=float,default=.01)
    p.add_argument('--camera-color-model',choices=('rgb','spatial'),default='rgb')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    payload=json.loads((args.data/'transforms.json').read_text());meta=json.loads(args.mesh_metadata.read_text())
    calibration=json.loads(args.camera_color_calibration.read_text());train=set(payload['train_filenames'])
    frames=[normalize_frame(f,payload,meta) for f in payload['frames'] if f['file_path'] in train]
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if len(frames)!=62 or len({f['physical_camera'] for f in frames})!=62 or forbidden&{f['physical_camera'] for f in frames}:
        raise ValueError('Requires 62 unique train cameras and no held-out RGB')
    if any(f.get('mask_path') for f in frames) or calibration['uses_eval_rgb'] is not False:
        raise ValueError('Semantic masks/eval RGB forbidden')
    mesh=o3d.io.read_triangle_mesh(str(args.mesh));vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
    if sha256(args.mesh)!=meta['output_sha256'] or calibration['mesh_sha256']!=meta['output_sha256']:
        raise ValueError('Geometry/color calibration mesh mismatch')
    points=torch.as_tensor(vertices,device='cuda',dtype=torch.float32);visibility=MeshVisibility(args.mesh)
    colors=[];directions=[];valid=[];source_hashes={}
    with torch.inference_mode():
        for index,f in enumerate(frames):
            path=(args.data/f['file_path']).resolve();physical=f['physical_camera'];row=calibration['cameras'][physical]
            digest=sha256(path)
            if digest!=row['image_sha256']:raise ValueError('Source RGB changed since camera color calibration')
            source_hashes[physical]=digest
            pose=torch.tensor(f['transform_matrix'],device='cuda',dtype=torch.float32)
            q=(points-pose[:3,3])@pose[:3,:3];z=-q[:,2]
            u=f['fl_x']*q[:,0]/z+f['cx']-.5;v=-f['fl_y']*q[:,1]/z+f['cy']-.5
            rgb=calibrated_camera_rgb(load_rgb(path,torch.device('cuda')),row,args.camera_color_model)
            sampled=grid_sample(rgb,u[None],v[None])[:,0].T
            projectable=(z>0)&(u>=3)&(u<f['w']-4)&(v>=3)&(v<f['h']-4)
            seen,_=visibility.visible(vertices,pose[:3,3].cpu().numpy(),projectable.cpu().numpy())
            seen=torch.as_tensor(seen,device='cuda')&(sampled>.1).all(-1)&(sampled<.9).all(-1)
            colors.append(sampled);directions.append(torch.nn.functional.normalize(pose[:3,3]-points,dim=-1));valid.append(seen)
            print(f'source={index+1}/62 valid={int(seen.sum())}',flush=True)
        rgb=torch.stack(colors);direction=torch.stack(directions);visible=torch.stack(valid)
        linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
        log_rgb=(linear/(1-linear).clamp_min(1e-6)).clamp_min(1e-7).log()
        blocks=np.floor(vertices/.004).astype(np.int64)
        held=torch.as_tensor((((blocks[:,0]*73856093)^(blocks[:,1]*19349663)^(blocks[:,2]*83492791))%5)==0,device='cuda')
        coefficients,stats=fit_angular_coefficients(log_rgb,direction,visible,vertices,triangles,held,smoothness=args.smoothness,ridge=args.ridge)
        if not stats['converged']:raise RuntimeError(f'Angular coefficient fit did not converge: {stats}')
        centers=np.array([np.asarray(f['transform_matrix'])[:3,3] for f in frames]);near=np.argsort(np.linalg.norm(centers[:,None]-centers[None],axis=-1),axis=1)[:,1:9]
        pairs=sorted({tuple(sorted((i,int(j)))) for i,js in enumerate(near) for j in js});rows=[];all_before=[];all_after=[]
        for i,j in pairs:
            check=visible[i]&visible[j]&held
            if int(check.sum())<20:continue
            gains=torch.einsum('nk,nkc->nc',direction[j,check]-direction[i,check],coefficients[check]).clamp(-math.log(2),math.log(2))
            corrected=encode_exposed_linear(decode_exposed_linear(rgb[i,check].cpu().numpy())*gains.exp().cpu().numpy())
            before=np.abs(rgb[i,check].cpu().numpy()-rgb[j,check].cpu().numpy()).mean(-1)
            after=np.abs(corrected-rgb[j,check].cpu().numpy()).mean(-1)
            all_before.extend(before);all_after.extend(after)
            rows.append({'i':frames[i]['physical_camera'],'j':frames[j]['physical_camera'],'held_samples':len(before),
                         'l1_before':float(np.median(before)),'l1_after':float(np.median(after))})
        file=args.output/'coefficients.npz';np.savez_compressed(file,coefficients=coefficients.cpu().numpy())
        result={'schema_version':1,'method':'mesh_attached_first_order_angular_log_gain','uses_eval_rgb':False,
                'uses_semantic_masks':False,'source_averaging':False,'camera_color_model':args.camera_color_model,'vertices':len(vertices),
                'mesh_sha256':sha256(args.mesh),'mesh_metadata_sha256':sha256(args.mesh_metadata),
                'camera_color_calibration_sha256':sha256(args.camera_color_calibration),'data_sha256':sha256(args.data/'transforms.json'),
                'source_hashes':source_hashes,'coefficients':file.name,'coefficients_sha256':sha256(file),
                'script_sha256':sha256(Path(__file__)),'fit':stats,'held_block_size_normalized':.004,'pairs':rows,
                'held_pair_display_l1_before':float(np.median(all_before)),'held_pair_display_l1_after':float(np.median(all_after))}
        atomic_json(args.output/'angular_color_manifest.json',result)
        print(json.dumps({k:v for k,v in result.items() if k in ('fit','held_pair_display_l1_before','held_pair_display_l1_after')}),flush=True)


if __name__=='__main__':main()
