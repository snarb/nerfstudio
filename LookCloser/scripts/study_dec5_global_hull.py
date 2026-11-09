#!/usr/bin/env python3
"""Global train-matte silhouette envelope; inferred occupancy, not measured depth.

Two fixed alpha thresholds expose the opaque core and a permissive coverage
envelope. Unknown crop/FOV samples never carve geometry. This is independent
of local TSDF hole membranes and of HairGS/scalp reconstruction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
import torch
import torch.nn.functional as F

from dec5_hair import cv_pose, VALIDATION, TEST
from run_dec5_hair_campaign import atomic_json, sha


def portrait_crop_uv(uv, width, crop):
    """Continuous pixel centers after numpy.rot90 and portrait crop."""
    return np.stack((uv[..., 1]-crop[0], width-uv[..., 0]-crop[1]), -1)


def insert_three(a, b, c, value):
    """Keep the three smallest distances without a camera x volume tensor."""
    first = torch.minimum(a, value)
    rest = torch.maximum(a, value)
    second = torch.minimum(b, rest)
    third = torch.minimum(c, torch.maximum(b, rest))
    return first, second, third


def carve_mesh(root, output):
    """Remove globally contradicted faces, then allow a fresh target raycast."""
    import open3d as o3d
    from dec5_hair import project
    output.mkdir(parents=True,exist_ok=False)
    source=root/'center_tsdf_base/mesh.ply'
    mesh=o3d.io.read_triangle_mesh(str(source))
    vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    points=np.concatenate((vertices,vertices[faces].mean(1)))
    payload=json.loads((root/'dataset/transforms.json').read_text())
    rows=[f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(rows)!=59 or any(f['physical_camera'] in VALIDATION+TEST for f in rows):
        raise ValueError('Only train59 can carve the mesh')
    known=np.zeros(len(points),np.uint16);outside=np.zeros_like(known);records=[]
    started=time.monotonic()
    for index,row in enumerate(rows):
        path=root/'matting_v3'/(row['physical_camera']+'.npz')
        data=np.load(path);binary=data['alpha'].astype(np.float32)>=.05
        sdf=(distance_transform_edt(binary)-distance_transform_edt(~binary)).astype(np.float32)
        uv,z=project(points,row);uv=portrait_crop_uv(uv,row['w'],data['portrait_crop'])
        h,w=sdf.shape
        valid=(z>0)&(uv[:,0]>=.5)&(uv[:,0]<=w-.5)&(uv[:,1]>=.5)&(uv[:,1]<=h-.5)
        values=map_coordinates(sdf,(uv[:,::-1]-.5).T,order=1,mode='constant',cval=0,prefilter=False)
        known+=valid;outside+=valid&(values < -2.)
        records.append(dict(camera=row['physical_camera'],matte_sha256=sha(path)))
        atomic_json(output/'progress.json',dict(stage='carving',completed=index+1,total=59))
    contradicted=(known>=12)&(outside>=3)
    remove=contradicted[:len(vertices)][faces].all(1)&contradicted[len(vertices):]
    if remove.all():raise ValueError('All head geometry contradicted; inspect calibration and mattes')
    candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces[~remove]))
    candidate.compute_vertex_normals();o3d.io.write_triangle_mesh(str(output/'mesh.ply'),candidate)
    np.savez_compressed(output/'evidence.npz',known=known,outside=outside,removed_face_ids=np.flatnonzero(remove))
    atomic_json(output/'complete.json',dict(source_mesh_sha256=sha(source),mesh_sha256=sha(output/'mesh.ply'),
        removed_faces=int(remove.sum()),retained_faces=int((~remove).sum()),input_records=records,
        policy='all vertices AND centroid outside person alpha .05 by >2 native pixels in >=3 views; >=12 known',
        unchanged_vertices=True,unknown_is_not_background=True,train_only=True,
        seconds=time.monotonic()-started,implementation_sha256=sha(__file__)))
    atomic_json(output/'progress.json',dict(stage='complete',completed=59,total=59))


def mesh_fields(output):
    from skimage.measure import marching_cubes
    import open3d as o3d
    if (output/'complete.json').exists():
        raise FileExistsError('Hull result is immutable')
    request=json.loads((output/'request.json').read_text())
    data=np.load(output/'field.npz')
    fields,known,lower,spacing=[data[k] for k in ('field','known','lower','spacing')]
    spacing=float(spacing);upper=np.asarray(request['upper'])
    started=time.monotonic();meshes=[]
    for channel,name in enumerate(('coverage','opaque')):
        value=fields[...,channel]
        if not (value.min()<0<value.max()):
            raise ValueError('Empty or unbounded hull: '+name)
        vertices,faces,_,_=marching_cubes(value,level=0,spacing=(spacing,)*3)
        vertices=vertices.astype(np.float64)+lower
        boundary=((vertices<=lower+1.5*spacing)|(vertices>=upper-1.5*spacing)).any(-1)
        supported=map_coordinates(known.astype(np.float32),((vertices-lower)/spacing).T,order=0,mode='nearest')>=12
        keep=~boundary[faces].any(-1)&supported[faces].all(-1)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces[keep]))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(output/(name+'.ply')),mesh)
        meshes.append(dict(name=name,vertices=len(vertices),triangles=int(keep.sum()),removed_box_or_unknown_faces=int((~keep).sum()),
                           sha256=sha(output/(name+'.ply'))))
    atomic_json(output/'complete.json',dict(meshes=meshes,field_sha256=sha(output/'field.npz'),
        meshing_seconds=time.monotonic()-started,known_view_range=[int(known.min()),int(known.max())],
        request_sha256=sha(output/'request.json'),geometry_ground_truth=False))
    atomic_json(output/'progress.json',dict(stage='complete',completed=int(known.size),total=int(known.size)))


def build(root, output, resolution=384):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8*(1 << 30)/torch.cuda.get_device_properties(0).total_memory)
    payload = json.loads((root/'dataset/transforms.json').read_text())
    rows = [f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(rows) != 59 or any(f['physical_camera'] in VALIDATION+TEST for f in rows):
        raise ValueError('Requires the split-clean train59 inventory')
    bound_path = root/'center_tsdf_base/mesh.json'
    bounds = np.asarray(json.loads(bound_path.read_text())['bounds'])
    lower, upper = bounds[0]-.02, bounds[1]+.02
    spacing = float((upper-lower).max()/(resolution-1))
    shape = np.ceil((upper-lower)/spacing).astype(int)+1
    upper = lower+(shape-1)*spacing
    inputs, records = [], []
    for row in rows:
        path = root/'matting_v3'/(row['physical_camera']+'.npz')
        data = np.load(path)
        alpha = data['alpha'].astype(np.float32)
        fields = []
        for threshold in (.05, .5):
            binary = alpha >= threshold
            if not binary.any() or binary.all():
                raise ValueError('Nontrivial head matte required: '+row['physical_camera'])
            fields.append(distance_transform_edt(binary)-distance_transform_edt(~binary))
        tensor = torch.as_tensor(np.stack(fields).astype(np.float32), device='cuda')[None]
        pose = torch.as_tensor(cv_pose(row), device='cuda', dtype=torch.float32)
        crop = data['portrait_crop'].astype(int)
        inputs.append((row, crop, tensor, pose))
        records.append(dict(camera=row['physical_camera'], matte_sha256=sha(path), crop=crop.tolist()))
    atomic_json(output/'request.json', dict(input_records=records, dataset_sha256=sha(root/'dataset/transforms.json'),
        bounds_source_sha256=sha(bound_path), lower=lower.tolist(), upper=upper.tolist(), spacing=spacing,
        shape=shape.tolist(), alpha_thresholds=[.05,.5], order_statistic=3, minimum_known_views=12,
        classification='inferred person silhouette occupancy; not hair depth or scalp',
        train_only=True, uses_validation_rgb=False, uses_test_rgb=False, implementation_sha256=sha(__file__)))
    count = int(np.prod(shape))
    fields = np.empty((count, 2), np.float32)
    known = np.empty(count, np.uint8)
    started = time.monotonic()
    step = 262144
    with torch.inference_mode():
        for start in range(0, count, step):
            end = min(start+step, count)
            indices = torch.arange(start, end, device='cuda')
            xyz = torch.stack((indices//int(shape[1]*shape[2]), (indices//int(shape[2]))%int(shape[1]), indices%int(shape[2])), -1)
            xyz = xyz.float()*spacing+torch.as_tensor(lower, device='cuda', dtype=torch.float32)
            a = torch.full((end-start, 2), float('inf'), device='cuda')
            b, c = a.clone(), a.clone()
            available = torch.zeros(end-start, dtype=torch.int16, device='cuda')
            for row, crop, tensor, pose in inputs:
                q = (xyz-pose[:3,3]) @ pose[:3,:3]
                z = q[:,2]
                u = row['fl_x']*q[:,0]/z.clamp_min(1e-8)+row['cx']
                v = row['fl_y']*q[:,1]/z.clamp_min(1e-8)+row['cy']
                x, y = v-float(crop[0]), row['w']-u-float(crop[1])
                height, width = tensor.shape[-2:]
                valid = (z>0) & (x>=.5) & (x<=width-.5) & (y>=.5) & (y<=height-.5)
                grid = torch.stack((2*x/width-1, 2*y/height-1), -1)[None,None]
                value = F.grid_sample(tensor, grid, align_corners=False, padding_mode='zeros')[0,:,0].T
                value *= z[:,None]/np.sqrt(row['fl_x']*row['fl_y'])
                value = torch.where(valid[:,None], value, float('inf'))
                a,b,c = insert_three(a,b,c,value)
                available += valid.to(torch.int16)
            c[available<12] = -spacing*2
            fields[start:end] = c.cpu().numpy()
            known[start:end] = available.cpu().numpy().astype(np.uint8)
            atomic_json(output/'progress.json', dict(stage='volume', completed=end, total=count, seconds=time.monotonic()-started))
    fields = fields.reshape((*shape, 2)); known = known.reshape(shape)
    np.savez_compressed(output/'field.npz', field=fields, known=known, lower=lower, spacing=spacing)
    mesh_fields(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resolution',type=int,default=384)
    p.add_argument('--stage',choices=('build','mesh','carve'),default='build')
    args=p.parse_args()
    if not 64<=args.resolution<=768: p.error('Resolution must be 64..768')
    if args.stage=='carve':carve_mesh(args.root,args.output)
    elif args.stage=='mesh':mesh_fields(args.output)
    else:build(args.root,args.output,args.resolution)


if __name__=='__main__':
    main()
