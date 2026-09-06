#!/usr/bin/env python3
"""Opt-in common mesh-attached display-RGB base plus one-camera detail.

Each train camera gets an independently smoothed observed RGB base. A separate
base fits the per-vertex visible-camera mean. Rendered RGB is source RGB minus
its base plus the common base. This is explicit low-frequency color replacement,
not pointwise unchanged reprojection, a log-gain field, or a view-specific blend.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256

HELD={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}


def fit_surface_bases(rgb,visible,vertices,triangles,*,smoothness=64.,ridge=1e-4,iterations=4096,tolerance=1e-7):
    """Same mesh low-pass operator for observed source bases and common RGB."""
    vertices=np.asarray(vertices,np.float64);triangles=np.asarray(triangles)
    if (rgb.ndim!=3 or rgb.shape[-1]!=3 or visible.shape!=rgb.shape[:2] or visible.dtype!=torch.bool
            or vertices.shape!=(rgb.shape[1],3) or triangles.ndim!=2 or triangles.shape[1]!=3
            or not np.issubdtype(triangles.dtype,np.integer) or not len(triangles)
            or triangles.min()<0 or triangles.max()>=len(vertices) or not np.isfinite(vertices).all()):
        raise ValueError('Invalid camera/vertex/color/triangle inventory')
    if (not np.isfinite([smoothness,ridge,tolerance]).all() or min(smoothness,ridge,tolerance)<=0 or iterations<1
            or not bool(visible.any(1).all()) or not bool(torch.isfinite(rgb[visible]).all())
            or bool(((rgb[visible]<0)|(rgb[visible]>1)).any())):
        raise ValueError('Need finite visible display RGB and positive solver settings')
    clean=torch.where(visible[...,None],rgb,0).double()
    count=visible.sum(0);common=clean.sum(0)/count.clamp_min(1)[:,None]
    values=torch.cat((clean,common[None]),0)
    valid=torch.cat((visible,(count>0)[None]),0).double()
    means=(values*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
    target=(values-means[:,None]).permute(1,0,2)
    weight=valid.T;n=len(vertices)
    edges=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    length=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
    ew=smoothness*np.clip((.0005/np.maximum(length,1e-7))**2,.05,20.)
    index=torch.as_tensor(np.r_[edges,edges[:,::-1]].T,device=rgb.device)
    adjacency=torch.sparse_coo_tensor(index,torch.as_tensor(np.r_[ew,ew],device=rgb.device),(n,n)).coalesce()
    degree=torch.sparse.sum(adjacency,dim=1).to_dense()
    diagonal=weight+degree[:,None]+ridge
    def matvec(value):
        return diagonal[...,None]*value-torch.sparse.mm(adjacency,value.reshape(n,-1)).reshape_as(value)
    rhs=weight[...,None]*target
    def dot(a,b):return (a*b).sum(0,keepdim=True)
    x=torch.zeros_like(rhs);r=rhs.clone();z=r/diagonal[...,None];direction=z.clone();rz=dot(r,z)
    # A constant field has roundoff-only RHS; use an explicit absolute floor
    # instead of requiring a relative solve of floating-point cancellation.
    norm=dot(rhs,rhs).sqrt().clamp_min(1e-6)
    for iteration in range(1,iterations+1):
        product=matvec(direction);alpha=rz/dot(direction,product).clamp_min(1e-30)
        x+=alpha*direction;r-=alpha*product;z=r/diagonal[...,None]
        next_rz=dot(r,z);direction=z+(next_rz/rz.clamp_min(1e-30))*direction;rz=next_rz
        if iteration%16==0 and bool((dot(r,r).sqrt()/norm<tolerance).all()):break
    residual=matvec(x)-rhs;relative=dot(residual,residual).sqrt()/norm
    bases=x.permute(1,0,2)+means[:,None]
    if not bool(torch.isfinite(bases).all()):raise RuntimeError('Nonfinite base solution')
    stats=dict(iterations=iteration,max_relative_residual=float(relative.max()),converged=bool((relative<5*tolerance).all()),
        required_true_relative_residual=5*tolerance,residual_normalization_floor=1e-6,
        smoothness=smoothness,ridge=ridge,edges=len(edges),
        all_train_observations_used=True,unseen_rgb_ignored=True,held_vertices=0,
        vertices_with_any_source=int((count>0).sum()),vertices_with_two_sources=int((count>=2).sum()),
        base_min=float(bases.min()),base_max=float(bases.max()),domain='display RGB',
        camera_count=len(rgb),common_base='mesh low-pass of visible train-camera arithmetic mean',
        detail_base='same low-pass fit separately to each train camera; prior is its visible mean')
    return bases[:-1].float(),bases[-1].float(),stats


class CanonicalSurfaceBase:
    # Reuse the established, tested geometric binding and interpolation only.
    from mesh_camera_color import MeshCameraColor as _Sampler
    bind=_Sampler.bind
    sample_offset=_Sampler.sample

    def __init__(self,manifest_path,mesh_path,calibration_path):
        import open3d as o3d
        self.o3d=o3d;self.manifest=json.loads(Path(manifest_path).read_text());m=self.manifest
        if (m['uses_eval_rgb'] is not False or m['uses_semantic_masks'] is not False or m['query_dependent_base'] is not False
                or m['mesh_sha256']!=sha256(mesh_path) or m['camera_color_calibration_sha256']!=sha256(calibration_path)
                or m['camera_color_model']!='spatial' or m['fit']['converged'] is not True):
            raise ValueError('Canonical base provenance/solver mismatch')
        self.names=m['source_cameras'];self.index={name:i for i,name in enumerate(self.names)}
        if len(self.names)!=len(self.index) or set(self.names)&HELD:raise ValueError('Duplicate or held source camera')
        file=Path(manifest_path).parent/m['coefficients']
        if sha256(file)!=m['coefficients_sha256']:raise ValueError('Canonical base coefficient checksum mismatch')
        with np.load(file,allow_pickle=False) as values:
            source=values['source_bases'];common=values['common_base'];self.fields=common[None]-source
        mesh=o3d.io.read_triangle_mesh(str(mesh_path));self.triangles=np.asarray(mesh.triangles)
        if source.shape!=(len(self.names),len(mesh.vertices),3) or common.shape!=(len(mesh.vertices),3) or not np.isfinite(self.fields).all():
            raise ValueError('Invalid finite source/common coefficient inventory')
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=8)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))


def main():
    import open3d as o3d
    from render_patchmatch_camera_path import normalize_frame
    from render_mesh_image_blend import load_rgb,grid_sample
    from patchmatch_color_calibration import apply_camera_gain
    from mesh_texture_visibility import MeshVisibility
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','mesh','mesh-metadata','camera-color-calibration','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoothness',type=float,default=64.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous fits; no marker-only resume')
    data=json.loads((a.data/'transforms.json').read_text());meta=json.loads(a.mesh_metadata.read_text())
    calibration=json.loads(a.camera_color_calibration.read_text());train=set(data['train_filenames'])
    frames=[normalize_frame(f,data,meta) for f in data['frames'] if f['file_path'] in train]
    names=[f['physical_camera'] for f in frames]
    if len(names)!=len(set(names)) or len(names)!=62 or set(names)&HELD:raise ValueError('Need 62 unique train-only cameras')
    if any(f.get('mask_path') for f in data['frames']) or calibration['uses_eval_rgb'] or calibration['uses_semantic_masks']:
        raise ValueError('Masks/held RGB forbidden')
    if sha256(a.mesh)!=meta['output_sha256'] or calibration['mesh_sha256']!=meta['output_sha256']:
        raise ValueError('Mesh calibration mismatch')
    files=[a.data/'transforms.json',a.mesh,a.mesh_metadata,a.camera_color_calibration,Path(__file__),
        Path(__file__).with_name('mesh_camera_color.py'),Path(__file__).with_name('mesh_texture_visibility.py'),
        Path(__file__).with_name('render_mesh_image_blend.py'),Path(__file__).with_name('patchmatch_color_calibration.py')]
    a.output.mkdir(parents=True)
    atomic_json(a.output/'request.json',dict(uses_eval_rgb=False,uses_semantic_masks=False,query_dependent_base=False,
        smoothness=a.smoothness,ridge=1e-4,all_train_observations_used=True,domain='display RGB',
        input_hashes={str(path.resolve()):sha256(path) for path in files},source_cameras=names))
    mesh=o3d.io.read_triangle_mesh(str(a.mesh));vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
    points=torch.as_tensor(vertices,device='cuda',dtype=torch.float32);visibility=MeshVisibility(a.mesh)
    colors=[];valid=[];hashes={}
    with torch.inference_mode():
        for i,f in enumerate(frames):
            path=(a.data/f['file_path']).resolve();row=calibration['cameras'][names[i]];hashes[names[i]]=sha256(path)
            if hashes[names[i]]!=row['image_sha256']:raise ValueError('Changed train JPEG')
            pose=torch.tensor(f['transform_matrix'],device='cuda',dtype=torch.float32);q=(points-pose[:3,3])@pose[:3,:3];depth=-q[:,2]
            u=f['fl_x']*q[:,0]/depth+f['cx']-.5;v=-f['fl_y']*q[:,1]/depth+f['cy']-.5
            rgb=apply_camera_gain(load_rgb(path,torch.device('cuda')),row['exposure_gain'],row['spatial_log_gain_grid'])
            sampled=grid_sample(rgb,u[None],v[None])[:,0].T
            projected=(depth>0)&(u>=3)&(u<f['w']-4)&(v>=3)&(v<f['h']-4)
            seen,_=visibility.visible(vertices,pose[:3,3].cpu().numpy(),projected.cpu().numpy())
            seen=torch.as_tensor(seen,device='cuda')&torch.isfinite(sampled).all(-1)
            colors.append(sampled);valid.append(seen)
            print(f'source={i+1}/62 visible={int(seen.sum())}',flush=True)
        rgb=torch.stack(colors);visible=torch.stack(valid)
        np.savez_compressed(a.output/'observations.npz',rgb=rgb.cpu().numpy(),visible=visible.cpu().numpy())
        source,common,stats=fit_surface_bases(rgb,visible,vertices,triangles,smoothness=a.smoothness)
        if not stats['converged']:raise RuntimeError('Canonical base true-residual gate failed: '+str(stats))
        coefficients=a.output/'bases.npz';np.savez_compressed(coefficients,source_bases=source.cpu().numpy(),common_base=common.cpu().numpy())
        atomic_json(a.output/'base_manifest.json',dict(method='mesh_common_display_base_plus_single_source_detail',
            uses_eval_rgb=False,uses_semantic_masks=False,query_dependent_base=False,camera_color_model='spatial',
            source_rgb_averaging='common low-frequency base only; hard-source detail retained',source_cameras=names,source_hashes=hashes,
            mesh_sha256=sha256(a.mesh),mesh_metadata_sha256=sha256(a.mesh_metadata),
            camera_color_calibration_sha256=sha256(a.camera_color_calibration),request_sha256=sha256(a.output/'request.json'),
            coefficients=coefficients.name,coefficients_sha256=sha256(coefficients),observations_sha256=sha256(a.output/'observations.npz'),fit=stats,
            accepted_surface_recipe=False))
        print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
