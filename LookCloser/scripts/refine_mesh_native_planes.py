#!/usr/bin/env python3
"""Opt-in, one-step normal refinement of a TSDF mesh using train depth planes.

No RGB, semantic regions, query cameras or query image coordinates are inputs.
All scalar displacements use the same native-plane consensus rule. Connectivity
is retained; local step damping prevents triangle inversion/area collapse.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from carve_patchmatch_mesh_free_space import train_frames
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from render_patchmatch_camera_path import normalize_frame


def native_plane_votes(depth, points, normals, camera):
    """Vectorized counterpart of audit_native_plane_displacement, float64."""
    if (depth.ndim != 2 or points.ndim != 2 or points.shape[1] != 3
            or normals.shape != points.shape or not bool(torch.isfinite(points).all())
            or not bool(torch.isfinite(normals).all())
            or not bool((torch.linalg.norm(normals,dim=1)-1).abs().lt(1e-6).all())):
        raise ValueError('Need depth and finite points/unit normals')
    dev=points.device; p=points.double(); normals=normals.double()
    pose=torch.as_tensor(camera['transform_matrix'],device=dev,dtype=torch.float64)
    q=(p-pose[:3,3])@pose[:3,:3];z=-q[:,2]
    safe=torch.where(z>0,z,torch.ones_like(z))
    u=camera['fl_x']*q[:,0]/safe+camera['cx']-.5
    v=-camera['fl_y']*q[:,1]/safe+camera['cy']-.5
    x,y=u.round().long(),v.round().long();h,w=depth.shape
    inside=(z>0)&(x>=2)&(x<w-2)&(y>=2)&(y<h-2)
    ids=torch.where(inside)[0]
    result=torch.full((len(p),),float('nan'),device=dev,dtype=torch.float64)
    if not len(ids):
        return result
    offsets=torch.arange(-2,3,device=dev)
    dy,dx=torch.meshgrid(offsets,offsets,indexing='ij');dx=dx.flatten();dy=dy.flatten()
    design=torch.stack((dx,dy,torch.ones_like(dx)),1).double()
    values=depth[y[ids,None]+dy,x[ids,None]+dx].double()
    positive=torch.isfinite(values)&(values>0)
    middle=torch.nanquantile(torch.where(positive,values,float('nan')),.5,dim=1)
    use=positive&((values-middle[:,None]).abs()<=.005)
    inverse=torch.where(positive,values.reciprocal(),0.)
    def fit(mask):
        weights=mask.double()
        lhs=torch.einsum('ni,ij,ik->njk',weights,design,design)
        rhs=(weights*inverse)@design
        supported=mask.sum(1)>=20
        lhs=torch.where(supported[:,None,None],lhs,torch.eye(3,device=dev,dtype=torch.float64))
        coeff=torch.linalg.solve(lhs,rhs.unsqueeze(-1)).squeeze(-1)
        return coeff,supported
    for _ in range(3):
        coeff,supported=fit(use)
        pred_inv=coeff@design.T
        pred_z=torch.where(pred_inv>0,pred_inv.reciprocal(),float('inf'))
        use=use&supported[:,None]&((pred_z-values).abs()<=.0005)
    coeff,supported=fit(use)
    aa,bb,cc=coeff.unbind(1)
    nc=torch.stack((aa*camera['fl_x'],bb*camera['fl_y'],
        cc+aa*(camera['cx']-.5-x[ids])+bb*(camera['cy']-.5-y[ids])),1)
    length=torch.linalg.norm(nc,dim=1)
    good=supported&(length>0)
    length=length.clamp_min(1e-20)
    nw=(nc*torch.tensor([1,-1,-1],device=dev))@pose[:3,:3].T/length[:,None]
    offset=1/length+(nw*pose[:3,3]).sum(1)
    cosine=(nw*normals[ids]).sum(1)
    displacement=(offset-(nw*p[ids]).sum(1))/torch.where(cosine!=0,cosine,torch.ones_like(cosine))
    good &= (cosine.abs()>=.5)&(displacement.abs()<=.004)&torch.isfinite(displacement)
    result[ids[good]]=displacement[good]
    return result


def consensus_votes(votes):
    """Largest 1 mm interval, >=3 views and >=60% of qualified plane votes."""
    if votes.ndim!=2 or votes.shape[0]<2:
        raise ValueError('Need camera by point votes')
    values=torch.sort(torch.where(torch.isfinite(votes),votes,float('inf')).T.contiguous(),dim=1).values
    total=torch.isfinite(values).sum(1)
    end=torch.searchsorted(values,values+.001,right=True)
    starts=torch.arange(votes.shape[0],device=votes.device)
    counts=torch.where(torch.isfinite(values),end-starts,0)
    count,left=counts.max(1); rows=torch.arange(len(values),device=votes.device)
    low=left+((count-1).clamp_min(0)//2);high=left+(count//2)
    selected=(values[rows,low]+values[rows,high])/2
    eligible=(count>=3)&(count>=.6*total)
    target=torch.where(eligible,selected,0.)
    return target,eligible,count,total


def smooth_supported_displacements(vertices, triangles, target, eligible, counts):
    """Laplacian regularization with exactly fixed zero unsupported vertices."""
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.linalg import cg
    n=len(vertices);edges=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    length=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
    ew=.2*np.clip((.0005/np.maximum(length,1e-7))**2,.05,20.)
    adjacency=coo_matrix((np.r_[ew,ew],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    weight=np.minimum(counts,8)/8
    matrix=diags(weight+np.asarray(adjacency.sum(1)).ravel()+1e-4)-adjacency
    ix=np.flatnonzero(eligible);result=np.zeros(n)
    if not len(ix):
        return result,dict(converged=True,relative_residual=0.,eligible_vertices=0)
    a=matrix[ix][:,ix];rhs=(weight*target)[ix]
    x,info=cg(a,rhs,rtol=1e-9,atol=1e-12,maxiter=4096)
    residual=float(np.linalg.norm(a@x-rhs)/max(np.linalg.norm(rhs),1e-9))
    if info or residual>1e-7 or not np.isfinite(x).all():
        raise RuntimeError(f'Normal displacement solve did not converge: {info}, {residual}')
    result[ix]=x
    return result,dict(converged=True,relative_residual=residual,eligible_vertices=len(ix))


def damp_noninverting(vertices, triangles, offsets):
    if (vertices.shape != offsets.shape or not np.isfinite(vertices).all()
            or not np.isfinite(offsets).all()):
        raise ValueError('Need matching finite vertex/displacement inventories')
    original=vertices[triangles]
    before=np.cross(original[:,1]-original[:,0],original[:,2]-original[:,0])
    area=np.linalg.norm(before,axis=1)
    if np.any(area<=0):
        raise ValueError('Input mesh contains zero-area triangles')
    scale=np.ones(len(vertices)); damped=np.zeros(len(vertices),bool)
    for iteration in range(33):
        trial=(vertices+offsets*scale[:,None])[triangles]
        after=np.cross(trial[:,1]-trial[:,0],trial[:,2]-trial[:,0])
        bad=((before*after).sum(1)<=0)|(np.linalg.norm(after,axis=1)<.1*area)
        if not bad.any():
            break
        ids=np.unique(triangles[bad]);damped[ids]=True
        scale[ids]*=.5
    else:
        raise RuntimeError('Could not preserve triangle orientation/area')
    return vertices+offsets*scale[:,None],dict(iterations=iteration,damped_vertices=int(damped.sum()),
        minimum_scale=float(scale.min()),inverted_or_collapsed_triangles=0,
        self_intersections_checked=False)


def main():
    import open3d as o3d
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('depth-data','mesh','mesh-metadata','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    payload=json.loads((a.depth_data/'transforms.json').read_text())
    meta=json.loads(a.mesh_metadata.read_text());frames=train_frames(payload)
    if sha256(a.mesh)!=meta['output_sha256'] or any(f.get('mask_path') for f in payload['frames']):
        raise ValueError('Wrong mesh receipt or forbidden mask')
    paths=[a.mesh,a.mesh_metadata,a.depth_data/'transforms.json',Path(__file__),
        Path(__file__).with_name('carve_patchmatch_mesh_free_space.py'),
        Path(__file__).with_name('render_patchmatch_camera_path.py')]
    paths += [a.depth_data/f['depth_file_path'] for f in frames]
    request=dict(method='native_plane_normal_consensus_one_step',uses_rgb=False,uses_eval_rgb=False,
        uses_semantic_masks=False,uses_target_coordinates=False,train_camera_count=62,
        topology_changes=False,passes=1,native_footprint=5,minimum_native_taps=20,
        plane_depth_residual=.0005,maximum_normal_shift=.004,minimum_plane_mesh_cosine=.5,
        consensus_interval=.001,minimum_consensus_views=3,minimum_consensus_fraction=.6,
        laplacian_smoothness=.2,unsupported_vertices_fixed=True,
        input_hashes={str(f.resolve()):sha256(f) for f in paths},
        device=a.device,accepted_surface_recipe=False)
    a.output.mkdir(parents=True);atomic_json(a.output/'request.json',request)
    mesh=o3d.io.read_triangle_mesh(str(a.mesh));mesh.compute_vertex_normals()
    vertices=np.asarray(mesh.vertices).copy();triangles=np.asarray(mesh.triangles).copy()
    normals=np.asarray(mesh.vertex_normals).copy()
    points=torch.as_tensor(vertices,device=a.device);n=torch.as_tensor(normals,device=a.device)
    votes=[];rows=[]
    with torch.inference_mode():
        for i,frame in enumerate(frames):
            with gzip.open(a.depth_data/frame['depth_file_path'],'rb') as stream:
                raw=np.load(stream,allow_pickle=False)
            if raw.shape!=(1080,1920) or not np.isfinite(raw).all() or (raw<0).any():
                raise ValueError('Invalid full-resolution geometric depth')
            depth=torch.as_tensor(raw.astype(np.float64)*float(meta['dataparser_scale']),device=a.device)
            f=normalize_frame(frame,payload,meta)
            vote=native_plane_votes(depth,points,n,f);votes.append(vote)
            row=dict(camera=f['physical_camera'],qualified_planes=int(torch.isfinite(vote).sum()),
                depth_coverage=float((raw>0).mean()))
            rows.append(row);print(json.dumps(dict(stage='native_planes',source=i+1,**row)),flush=True)
        all_votes=torch.stack(votes)
        target,eligible,count,total=consensus_votes(all_votes)
        target,eligible,count,total=(x.cpu().numpy() for x in (target,eligible,count,total))
        np.savez_compressed(a.output/'native_votes.npz',votes=all_votes.cpu().numpy(),target=target,
            eligible=eligible,agreeing_views=count,qualified_views=total,normals=normals)
    delta,solve=smooth_supported_displacements(vertices,triangles,target,eligible,count)
    refined,guard=damp_noninverting(vertices,triangles,normals*delta[:,None])
    distance=np.linalg.norm(refined-vertices,axis=1)
    if not np.any(distance>1e-9):
        raise RuntimeError('No actual geometry change')
    mesh.vertices=o3d.utility.Vector3dVector(refined);mesh.compute_vertex_normals()
    output=a.output/'refined.ply'
    if not o3d.io.write_triangle_mesh(str(output),mesh,write_ascii=False):
        raise OSError('Failed mesh write')
    np.savez_compressed(a.output/'applied_displacements.npz',offsets=refined-vertices,smoothed_normal_delta=delta)
    _,components,_=mesh.cluster_connected_triangles()
    result=deepcopy(meta)
    result.update(output=str(output),output_sha256=sha256(output),vertices=len(vertices),triangles=len(triangles),
        connected_components=len(components),component_triangles=sorted(map(int,components),reverse=True),
        artifact_type='TSDF-derived mesh with native train-depth normal refinement; not raw TSDF volume',
        native_plane_refinement=dict(request_sha256=sha256(a.output/'request.json'),
            geometry_only=True,topology_unchanged=True,solve=solve,triangle_guard=guard,
            displacement_quantiles=np.quantile(distance,[0,.5,.9,.99,1]).tolist(),
            vertices_moved_over_one_voxel=int((distance>.0005).sum()),
            per_camera=rows,evidence_sha256=sha256(a.output/'native_votes.npz'),
            applied_displacements_sha256=sha256(a.output/'applied_displacements.npz')),
        accepted_surface_recipe=False)
    atomic_json(a.output/'refined.json',result)
    print(json.dumps(dict(stage='complete',solve=solve,guard=guard,
        displacement_quantiles=result['native_plane_refinement']['displacement_quantiles'])),flush=True)


if __name__=='__main__':
    main()
