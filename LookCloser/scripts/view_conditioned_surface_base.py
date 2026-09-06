#!/usr/bin/env python3
"""Opt-in directional display-RGB base, with train-camera interpolation gate.

Unlike a gain-difference field this fits the low-frequency source RGB itself.
The query base is continuous in viewing direction. Original hard source detail
can be added after subtracting that source's independently smoothed RGB base.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from canonical_surface_base import HELD


def direction_features(direction, degree):
    """Fixed bounded polynomial features; not a physical reflectance model."""
    if degree not in (0, 1, 2) or direction.shape[-1] != 3 or not bool(torch.isfinite(direction).all()):
        raise ValueError('Need finite 3D directions and degree 0, 1 or 2')
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if bool((norm <= 1e-8).any()):
        raise ValueError('Zero camera direction')
    x, y, z = (direction/norm).unbind(-1)
    features = [torch.ones_like(x)]
    if degree >= 1:
        features += [x, y, z]
    if degree == 2:
        features += [x*y, x*z, y*z, x*x-y*y, 3*z*z-1]
    return torch.stack(features, -1)


def fit_directional_base(bases, directions, visible, vertices, triangles, *, degree,
                         smoothness=1., ridge=.02, iterations=4096, tolerance=1e-7):
    """Mesh-regularized low-band radiance, one normalized data weight per vertex."""
    vertices = np.asarray(vertices, np.float64); triangles = np.asarray(triangles)
    if (bases.ndim != 3 or bases.shape[-1] != 3 or bases.shape != directions.shape
            or visible.shape != bases.shape[:2] or visible.dtype != torch.bool
            or vertices.shape != (bases.shape[1], 3) or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.min() < 0 or triangles.max() >= len(vertices)
            or not np.isfinite([smoothness, ridge, tolerance]).all()
            or min(smoothness, ridge, tolerance) <= 0 or iterations < 1
            or not bool(visible.any()) or not bool(torch.isfinite(bases[visible]).all())):
        raise ValueError('Invalid visible bases, mesh or solver settings')
    feature = direction_features(directions.double(), degree)
    values = torch.where(visible[..., None], bases, 0).double()
    count = visible.sum(0); weight = visible.double()/count.clamp_min(1)[None]
    mean = values.sum((0, 1))/visible.sum()
    covariance = torch.einsum('cn,cnk,cnl->nkl', weight, feature, feature)
    rhs = torch.einsum('cn,cnk,cnl->nkl', weight, feature, values-mean)
    n, k = rhs.shape[:2]
    edges = np.concatenate([triangles[:,[0,1]], triangles[:,[1,2]], triangles[:,[2,0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    length = np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]], axis=1)
    ew = smoothness*np.clip((.0005/np.maximum(length, 1e-7))**2, .05, 20)
    index = torch.as_tensor(np.r_[edges, edges[:,::-1]].T, device=bases.device)
    adjacency = torch.sparse_coo_tensor(index, torch.as_tensor(np.r_[ew,ew], device=bases.device), (n,n)).coalesce()
    degree_sum = torch.sparse.sum(adjacency, dim=1).to_dense()
    prior = torch.tensor([1e-4]+([ridge]*3 if degree>=1 else [])+([ridge*4]*5 if degree==2 else []),
                         device=bases.device, dtype=torch.float64)
    diagonal = covariance + torch.diag_embed(degree_sum[:,None]+prior[None])
    inverse = torch.linalg.inv(diagonal)
    def matvec(value):
        return torch.bmm(diagonal, value)-torch.sparse.mm(adjacency, value.reshape(n,-1)).reshape_as(value)
    def dot(a,b):
        return (a*b).sum((0,1), keepdim=True)
    x = torch.zeros_like(rhs); r = rhs.clone(); z = torch.bmm(inverse,r); direction=z.clone(); rz=dot(r,z)
    norm=dot(rhs,rhs).sqrt().clamp_min(1e-6)
    for iteration in range(1,iterations+1):
        product=matvec(direction); alpha=rz/dot(direction,product).clamp_min(1e-30)
        x+=alpha*direction; r-=alpha*product; z=torch.bmm(inverse,r)
        next_rz=dot(r,z); direction=z+(next_rz/rz.clamp_min(1e-30))*direction; rz=next_rz
        if iteration%16==0 and bool((dot(r,r).sqrt()/norm<tolerance).all()):
            break
    residual=matvec(x)-rhs; relative=dot(residual,residual).sqrt()/norm
    x[:,0]+=mean
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError('Nonfinite directional coefficients')
    return x.float(), dict(degree=degree,iterations=iteration,smoothness=smoothness,ridge=ridge,
        max_relative_residual=float(relative.max()),converged=bool((relative<5*tolerance).all()),
        required_relative_residual=5*tolerance,residual_normalization_floor=1e-6,
        vertices_with_observation=int((count>0).sum()),visible_observations=int(visible.sum()),
        basis_size=k,domain='display RGB base',data_weights='normalized visible camera count per vertex')


def interpolation_gate(rows):
    """Fixed camera-balanced low-band gate; not held-view surface acceptance."""
    grouped = {row['degree']: row for row in rows}
    if set(grouped) != {0,1,2}:
        raise ValueError('Need degree-zero baseline plus both directional controls')
    names = [r['camera'] for r in grouped[0]['cameras']]
    if len(names) != len(set(names)) or not names:
        raise ValueError('Empty/duplicate held train cameras')
    original = np.array([r['mean_abs_rgb'] for r in grouped[0]['cameras']])
    choices=[]
    for degree in (1,2):
        row=grouped[degree]
        if [r['camera'] for r in row['cameras']] != names or [r['vertices'] for r in row['cameras']] != [r['vertices'] for r in grouped[0]['cameras']]:
            raise ValueError('Changing camera/vertex inventory')
        candidate=np.array([r['mean_abs_rgb'] for r in row['cameras']])
        if not np.isfinite(np.r_[original,candidate]).all() or (np.r_[original,candidate]<0).any():
            raise ValueError('Nonfinite/negative held errors')
        relative=(np.median(original)-np.median(candidate))/max(np.median(original),1e-12)
        improved=float(np.mean(candidate<original))
        choices.append(dict(degree=degree,relative_median_improvement=float(relative),improved_camera_fraction=improved,
                            eligible=bool(relative>=.10 and improved>=.60 and row['fit']['converged']),
                            median_camera_mean_abs_rgb=float(np.median(candidate))))
    eligible=[r for r in choices if r['eligible']]
    # Prefer lower degree unless quadratic reduces held error another 5%.
    selected=None
    if eligible:
        selected=min(eligible,key=lambda r:r['degree'])
        if len(eligible)==2 and eligible[1]['median_camera_mean_abs_rgb'] < .95*selected['median_camera_mean_abs_rgb']:
            selected=eligible[1]
    return dict(choices=choices,selected_degree=None if selected is None else selected['degree'],
        eligible_for_render=selected is not None,uses_eval_rgb=False,accepted_surface_recipe=False)


def main():
    import open3d as o3d
    from render_patchmatch_camera_path import normalize_frame
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-base-manifest','data','mesh','mesh-metadata','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing control; output must not exist')
    base=json.loads(a.source_base_manifest.read_text()); source_root=a.source_base_manifest.parent
    payload=json.loads((a.data/'transforms.json').read_text()); meta=json.loads(a.mesh_metadata.read_text())
    train=set(payload['train_filenames'])
    frames=[normalize_frame(f,payload,meta) for f in payload['frames'] if f['file_path'] in train]
    names=[f['physical_camera'] for f in frames]
    if (len(names)!=len(set(names)) or len(names)!=62 or names!=base['source_cameras'] or set(names)&HELD
            or base['uses_eval_rgb'] is not False or base['uses_semantic_masks'] is not False
            or any(f.get('mask_path') for f in payload['frames'])
            or sha256(a.mesh)!=base['mesh_sha256'] or sha256(a.mesh_metadata)!=base['mesh_metadata_sha256']):
        raise ValueError('Need identical mesh and 62 ordered unique train-only source bases')
    coefficients=source_root/base['coefficients']; observations=source_root/'observations.npz'
    if sha256(coefficients)!=base['coefficients_sha256'] or sha256(observations)!=base['observations_sha256']:
        raise ValueError('Source base checksum mismatch')
    files=[a.source_base_manifest,a.data/'transforms.json',a.mesh,a.mesh_metadata,coefficients,observations,Path(__file__),
           Path(__file__).with_name('canonical_surface_base.py'),Path(__file__).with_name('render_patchmatch_camera_path.py')]
    held_names=sorted(names)[::5]; held_indices=[names.index(name) for name in held_names]
    hashes={str(path.resolve()):sha256(path) for path in files}
    for f in frames:
        path=(a.data/f['file_path']).resolve();digest=sha256(path)
        if digest!=base['source_hashes'][f['physical_camera']]:raise ValueError('Changed train source RGB')
        hashes[str(path)]=digest
    a.output.mkdir(parents=True)
    atomic_json(a.output/'request.json',dict(input_hashes=hashes,source_cameras=names,held_train_cameras=held_names,
        degrees=[0,1,2],smoothness=1.,ridge=.02,minimum_relative_improvement=.10,minimum_improved_camera_fraction=.60,
        quadratic_extra_improvement=.05,uses_eval_rgb=False,uses_semantic_masks=False,
        validation_scope='Directional fit withholds 13 train cameras; earlier camera response is fixed and was fitted on all train cameras',
        output='mesh-attached continuous direction-dependent RGB base, then hard source detail'))
    mesh=o3d.io.read_triangle_mesh(str(a.mesh));vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
    with np.load(coefficients) as values:bases=torch.as_tensor(values['source_bases'],device='cuda')
    with np.load(observations) as values:visible=torch.as_tensor(values['visible'],device='cuda')
    centers=np.array([np.asarray(f['transform_matrix'])[:3,3] for f in frames])
    directions=torch.as_tensor(centers[:,None]-vertices[None],device='cuda',dtype=torch.float32)
    fit_visible=visible.clone();fit_visible[held_indices]=False
    check=(fit_visible.sum(0)>=2)[None]&visible[held_indices]
    rows=[]
    with torch.inference_mode():
        for degree in (0,1,2):
            coeff,stats=fit_directional_base(bases,directions,fit_visible,vertices,triangles,degree=degree)
            if not stats['converged']:raise RuntimeError('Directional low-base fit did not converge: '+str(stats))
            predicted=torch.einsum('cnk,nkl->cnl',direction_features(directions[held_indices],degree),coeff)
            cameras=[]
            for i,(index,name) in enumerate(zip(held_indices,held_names)):
                error=(predicted[i,check[i]]-bases[index,check[i]]).abs().mean(-1)
                if not len(error):raise ValueError('No eligible held-camera vertices')
                cameras.append(dict(camera=name,vertices=len(error),mean_abs_rgb=float(error.mean()),median_abs_rgb=float(error.median())))
            np.savez_compressed(a.output/f'held_degree{degree}.npz',coefficients=coeff.cpu().numpy())
            rows.append(dict(degree=degree,fit=stats,cameras=cameras))
            atomic_json(a.output/'held_scores.json',rows)
            print(json.dumps(dict(stage='held_fit',degree=degree,fit=stats,median_camera_error=float(np.median([r['mean_abs_rgb'] for r in cameras])))),flush=True)
        gate=interpolation_gate(rows);atomic_json(a.output/'selection_before_render.json',gate)
        if gate['eligible_for_render']:
            coeff,stats=fit_directional_base(bases,directions,visible,vertices,triangles,degree=gate['selected_degree'])
            if not stats['converged']:raise RuntimeError('All-train directional fit did not converge')
            output=a.output/'coefficients.npz';np.savez_compressed(output,coefficients=coeff.cpu().numpy())
            atomic_json(a.output/'directional_base_manifest.json',dict(method='direction_conditioned_mesh_display_RGB_base',
                source_base_manifest=str(a.source_base_manifest.resolve()),source_base_manifest_sha256=sha256(a.source_base_manifest),
                mesh_sha256=sha256(a.mesh),mesh_metadata_sha256=sha256(a.mesh_metadata),source_cameras=names,
                coefficients=output.name,coefficients_sha256=sha256(output),fit=stats,uses_eval_rgb=False,uses_semantic_masks=False,
                degree=gate['selected_degree'],query_dependent_base=True,accepted_surface_recipe=False,
                request_sha256=sha256(a.output/'request.json'),selection_sha256=sha256(a.output/'selection_before_render.json')))
        print(json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':main()
