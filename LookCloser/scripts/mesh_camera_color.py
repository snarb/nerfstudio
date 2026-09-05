#!/usr/bin/env python3
"""Train-only, mesh-attached per-camera low-frequency log RGB gain fields.

Fit pairwise color differences at common visible surface points, regularizing
the gain fields along mesh edges. Shared albedo/detail cancels from the fit.
Rendering still uses ONE projected train RGB, multiplied by that source's smooth
gain; it never renders the fitted common color or an average of source images.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256


def fit_mesh_camera_gains(log_rgb,valid,held,vertices,triangles,*,smoothness=64.,ridge=.01,
                          iterations=1536,tolerance=1e-7):
    """Solve (camera-centering projector + mesh Laplacian + ridge) gain = -P RGB."""
    if (log_rgb.ndim!=3 or log_rgb.shape[-1]!=3 or valid.shape!=log_rgb.shape[:2]
            or held.shape!=(log_rgb.shape[1],) or len(vertices)!=log_rgb.shape[1]):
        raise ValueError('Expected CN3 colors, CN visibility, N held flags and mesh vertices')
    if (not bool(torch.isfinite(log_rgb).all()) or not np.isfinite(vertices).all()
            or not math.isfinite(smoothness) or smoothness<=0 or not math.isfinite(ridge) or ridge<=0
            or not math.isfinite(tolerance) or tolerance<=0 or iterations<1):
        raise ValueError('Invalid finite mesh-response inputs or regularization')
    device=log_rgb.device;n=len(vertices)
    rgb=log_rgb.permute(1,0,2).double()
    weight=(valid&~held[None]).T.double()
    total=weight.sum(1,keepdim=True).clamp_min(1)
    def centered(value):
        mean=(value*weight[...,None]).sum(1,keepdim=True)/total[...,None]
        return weight[...,None]*(value-mean)
    edge=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    edge=np.unique(np.sort(edge,axis=1),axis=0)
    if not len(edge) or edge.min()<0 or edge.max()>=n:raise ValueError('Invalid/empty triangle graph')
    length=np.linalg.norm(vertices[edge[:,0]]-vertices[edge[:,1]],axis=-1)
    ew=np.clip((.0005/np.maximum(length,1e-7))**2,.05,20)*smoothness
    indices=torch.as_tensor(np.r_[edge,edge[:,::-1]].T,device=device)
    values=torch.as_tensor(np.r_[ew,ew],device=device,dtype=torch.float64)
    adjacency=torch.sparse_coo_tensor(indices,values,(n,n)).coalesce()
    degree=torch.sparse.sum(adjacency,dim=1).to_dense()
    diagonal=weight*(1-weight/total)+degree[:,None]+ridge
    def matvec(value):
        neighbors=torch.sparse.mm(adjacency,value.reshape(n,-1)).reshape_as(value)
        return centered(value)+(degree[:,None,None]+ridge)*value-neighbors
    # Subtract one visible reference before centering to cancel shared albedo
    # exactly, rather than turning rounding of identical RGB into a tiny RHS.
    reference=rgb[torch.arange(n,device=device),weight.argmax(1)]
    rhs=-centered(rgb-reference[:,None])
    # P and the shared mesh Laplacian preserve the zero-camera-mean subspace.
    # Projecting the preconditioner removes a numerically excited gauge mode;
    # it does not change the unique full-system solution (ridge > 0).
    def zero_mean(value):return value-value.mean(1,keepdim=True)
    rhs=zero_mean(rhs)
    # All channels/cameras are coupled by P; one CG system per RGB channel.
    def dot(a,b):return (a*b).sum((0,1),keepdim=True)
    x=torch.zeros_like(rhs);r=rhs.clone();z=zero_mean(r/diagonal[...,None]);p=z.clone();rz=dot(r,z)
    norm=dot(rhs,rhs).sqrt().clamp_min(1e-12)
    for iteration in range(1,iterations+1):
        ap=matvec(p);alpha=rz/dot(p,ap).clamp_min(1e-30)
        x+=alpha*p;r-=alpha*ap;z=zero_mean(r/diagonal[...,None])
        new_rz=dot(r,z);p=z+(new_rz/rz.clamp_min(1e-30))*p;rz=new_rz
        if iteration%16==0 and bool((dot(r,r).sqrt()/norm<tolerance).all()):break
    residual=matvec(x)-rhs;relative=dot(residual,residual).sqrt()/norm
    required=tolerance*5
    stats={'iterations':iteration,'requested_iterations':iterations,'max_relative_residual':float(relative.max()),
           'required_true_relative_residual':required,'converged':bool((relative<required).all()),
           'smoothness':smoothness,'ridge':ridge,'dtype':'float64','edge_count':len(edge),
           'fit_vertices_with_two_sources':int((weight.sum(1)>=2).sum()),'held_vertices':int(held.sum()),
           'maximum_abs_mean_camera_log_gain':float(x.mean(1).abs().max()),
           'gain_clamp_fraction':float((x.abs()>math.log(2)).double().mean()),
           'gauge':'zero mean camera log gain at each vertex before clipping'}
    return x.permute(1,0,2).float().clamp(-math.log(2),math.log(2)),stats


class MeshCameraColor:
    def __init__(self,manifest_path,mesh_path,calibration_path,camera_color_model):
        import open3d as o3d
        self.o3d=o3d;self.manifest=json.loads(Path(manifest_path).read_text())
        m=self.manifest
        if (m['uses_eval_rgb'] is not False or m['uses_semantic_masks'] is not False
                or m['output_source_averaging'] is not False or m['mesh_sha256']!=sha256(mesh_path)
                or m['camera_color_calibration_sha256']!=sha256(calibration_path)
                or m['camera_color_model']!=camera_color_model):
            raise ValueError('Mesh camera-color input, response mode or provenance mismatch')
        self.names=m['source_cameras'];self.index={name:i for i,name in enumerate(self.names)}
        forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
        if len(self.names)!=len(self.index) or forbidden&set(self.names):raise ValueError('Duplicate/held-out source camera')
        file=Path(manifest_path).parent/m['coefficients']
        if sha256(file)!=m['coefficients_sha256']:
            raise ValueError('Mesh camera-color coefficient checksum mismatch')
        with np.load(file,allow_pickle=False) as coefficients:
            self.fields=coefficients['log_gains']
        mesh=o3d.io.read_triangle_mesh(str(mesh_path));self.triangles=np.asarray(mesh.triangles)
        if (self.fields.shape!=(len(self.names),len(mesh.vertices),3) or not np.isfinite(self.fields).all()
                or np.max(np.abs(self.fields))>math.log(2)+1e-6 or m['fit']['converged'] is not True):
            raise ValueError('Mesh camera-color coefficient inventory/values invalid')
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=8)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def bind(self,world,support):
        self.shape=world.shape[:-1];self.support=support
        points=world[support].astype(np.float32)
        if not len(points):
            self.ids=np.empty(0,np.int64);self.weights=np.empty((0,3),np.float32);self.near=np.empty(0,bool)
            return
        nearest=self.scene.compute_closest_points(self.o3d.core.Tensor(points))
        self.ids=nearest['primitive_ids'].numpy();uv=nearest['primitive_uvs'].numpy()
        self.weights=np.c_[1-uv.sum(-1),uv]
        self.near=np.linalg.norm(points-nearest['points'].numpy(),axis=-1)<=.001

    def sample(self,physical_camera):
        values=(self.fields[self.index[physical_camera],self.triangles[self.ids]]*self.weights[:,:,None]).sum(1)
        values[~self.near]=0
        result=np.zeros((*self.shape,3),np.float32);result[self.support]=values
        return result


def main():
    import open3d as o3d
    from render_patchmatch_camera_path import normalize_frame
    from render_mesh_image_blend import load_rgb,grid_sample
    from patchmatch_color_calibration import apply_camera_gain,decode_exposed_linear,encode_exposed_linear
    from mesh_texture_visibility import MeshVisibility
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','mesh','mesh-metadata','camera-color-calibration','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoothness',type=float,default=64.);p.add_argument('--ridge',type=float,default=.01)
    p.add_argument('--solver-iterations',type=int,default=4096)
    p.add_argument('--camera-color-model',choices=['rgb','spatial','spatial-rgb'],default='spatial')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous fits')
    payload=json.loads((a.data/'transforms.json').read_text());meta=json.loads(a.mesh_metadata.read_text())
    calibration=json.loads(a.camera_color_calibration.read_text());train=set(payload['train_filenames'])
    frames=[normalize_frame(f,payload,meta) for f in payload['frames'] if f['file_path'] in train]
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if len(frames)!=62 or len({f['physical_camera'] for f in frames})!=62 or forbidden&{f['physical_camera'] for f in frames}:
        raise ValueError('Requires 62 unique fixed train cameras, no held-out sources')
    if any(f.get('mask_path') for f in frames) or calibration['uses_eval_rgb'] is not False or calibration['uses_semantic_masks'] is not False:
        raise ValueError('Masks/eval RGB forbidden')
    if sha256(a.mesh)!=meta['output_sha256'] or calibration['mesh_sha256']!=meta['output_sha256']:
        raise ValueError('Mesh/color calibration geometry mismatch')
    mesh=o3d.io.read_triangle_mesh(str(a.mesh));vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
    points=torch.as_tensor(vertices,device='cuda',dtype=torch.float32);visibility=MeshVisibility(a.mesh)
    colors=[];valid=[];source_hashes={}
    with torch.inference_mode():
        for i,f in enumerate(frames):
            path=(a.data/f['file_path']).resolve();physical=f['physical_camera'];row=calibration['cameras'][physical]
            source_hashes[physical]=sha256(path)
            if source_hashes[physical]!=row['image_sha256']:raise ValueError('Source RGB changed since calibration')
            pose=torch.tensor(f['transform_matrix'],device='cuda',dtype=torch.float32)
            q=(points-pose[:3,3])@pose[:3,:3];z=-q[:,2]
            u=f['fl_x']*q[:,0]/z+f['cx']-.5;v=-f['fl_y']*q[:,1]/z+f['cy']-.5
            gain=row['exposure_gain'] if a.camera_color_model=='spatial' else row['rgb_gain']
            grid=row['spatial_log_gain_grid'] if a.camera_color_model=='spatial' else row['spatial_rgb_log_gain_grid'] if a.camera_color_model=='spatial-rgb' else None
            rgb=apply_camera_gain(load_rgb(path,torch.device('cuda')),gain,grid)
            sampled=grid_sample(rgb,u[None],v[None])[:,0].T
            projectable=(z>0)&(u>=3)&(u<f['w']-4)&(v>=3)&(v<f['h']-4)
            seen,_=visibility.visible(vertices,pose[:3,3].cpu().numpy(),projectable.cpu().numpy())
            seen=torch.as_tensor(seen,device='cuda')&(sampled>.1).all(-1)&(sampled<.9).all(-1)
            colors.append(sampled);valid.append(seen)
            print(f'source={i+1}/62 valid={int(seen.sum())}',flush=True)
        rgb=torch.stack(colors);visible=torch.stack(valid)
        linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
        log_rgb=(linear/(1-linear).clamp_min(1e-6)).clamp_min(1e-7).log()
        blocks=np.floor(vertices/.004).astype(np.int64)
        held=torch.as_tensor((((blocks[:,0]*73856093)^(blocks[:,1]*19349663)^(blocks[:,2]*83492791))%5)==0,device='cuda')
        fields,stats=fit_mesh_camera_gains(log_rgb,visible,held,vertices,triangles,smoothness=a.smoothness,ridge=a.ridge,iterations=a.solver_iterations)
        if not stats['converged']:raise RuntimeError(f'Failed true-residual gate: {stats}')
        corrected=encode_exposed_linear(decode_exposed_linear(rgb.cpu().numpy())*fields.exp().cpu().numpy())
        centers=np.array([np.asarray(f['transform_matrix'])[:3,3] for f in frames]);near=np.argsort(np.linalg.norm(centers[:,None]-centers[None],axis=-1),axis=1)[:,1:9]
        pairs=sorted({tuple(sorted((i,int(j)))) for i,js in enumerate(near) for j in js});rows=[];before=[];after=[]
        for i,j in pairs:
            check=(visible[i]&visible[j]&held).cpu().numpy()
            if check.sum()<20:continue
            b=np.abs(rgb[i].cpu().numpy()[check]-rgb[j].cpu().numpy()[check]).mean(-1)
            c=np.abs(corrected[i,check]-corrected[j,check]).mean(-1)
            before.extend(b);after.extend(c)
            rows.append({'i':frames[i]['physical_camera'],'j':frames[j]['physical_camera'],'held_samples':int(check.sum()),
                         'before':float(np.median(b)),'after':float(np.median(c))})
        a.output.mkdir(parents=True);file=a.output/'camera_log_gains.npz';np.savez_compressed(file,log_gains=fields.cpu().numpy())
        result={'method':'mesh_attached_camera_log_rgb_gain_fields','uses_eval_rgb':False,'uses_semantic_masks':False,
            'output_source_averaging':False,'renders_latent_consensus_rgb':False,'geometry_changed':False,
            'camera_color_model':a.camera_color_model,'source_cameras':[f['physical_camera'] for f in frames],
            'source_hashes':source_hashes,'mesh_sha256':sha256(a.mesh),'mesh_metadata_sha256':sha256(a.mesh_metadata),
            'camera_color_calibration_sha256':sha256(a.camera_color_calibration),'data_sha256':sha256(a.data/'transforms.json'),
            'coefficients':file.name,'coefficients_sha256':sha256(file),'script_sha256':sha256(Path(__file__)),
            'color_helper_sha256':sha256(Path(__file__).with_name('patchmatch_color_calibration.py')),
            'pixel_center_offset':.5,'exact_mesh_visibility':True,'held_block_size_normalized':.004,
            'fit':stats,'held_pair_display_l1_before':float(np.median(before)),'held_pair_display_l1_after':float(np.median(after)),
            'held_pair_samples':len(before),'pairs':rows,'interpretation':__doc__}
        atomic_json(a.output/'mesh_camera_color_manifest.json',result)
        print(json.dumps({k:result[k] for k in ['fit','held_pair_display_l1_before','held_pair_display_l1_after','held_pair_samples']}),flush=True)


if __name__=='__main__':main()
