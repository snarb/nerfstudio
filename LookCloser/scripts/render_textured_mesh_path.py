#!/usr/bin/env python3
"""Raycast a fixed OBJ texture atlas at calibration-only camera path positions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from render_patchmatch_camera_path import normalize_frame,calibration_path,ANCHORS


def load_obj(path):
    vertices=[];uv=[];triangles=[];triangle_uv=[];materials=[];names=[];active=None
    texture_paths={}
    for line in path.read_text().splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0]=='mtllib':
            material=None
            for raw in (path.parent/parts[1]).read_text().splitlines():
                fields=raw.split()
                if not fields:continue
                if fields[0]=='newmtl':material=fields[1]
                elif fields[0]=='map_Kd':texture_paths[material]=path.parent/fields[1]
        elif parts[0]=='v':vertices.append(list(map(float,parts[1:4])))
        elif parts[0]=='vt':uv.append(list(map(float,parts[1:3])))
        elif parts[0]=='usemtl':
            active=parts[1]
            if active not in names:names.append(active)
        elif parts[0]=='f':
            if len(parts)!=4 or active is None:raise ValueError('Expected triangular textured OBJ')
            ids=[list(map(int,item.split('/')[:2])) for item in parts[1:]]
            if min(min(i) for i in ids)<1:raise ValueError('Only positive OBJ indices supported')
            triangles.append([i[0]-1 for i in ids]);triangle_uv.append([i[1]-1 for i in ids]);materials.append(names.index(active))
    textures=[np.asarray(Image.open(texture_paths[n]).convert('RGB')) for n in names]
    return (np.asarray(vertices,np.float32),np.asarray(triangles,np.int32),
            np.asarray(uv,np.float32)[np.asarray(triangle_uv)],np.asarray(materials),textures,texture_paths)


def sample_atlas(texture,uv):
    # texrecon divides zero-indexed pixel coordinates by size without adding .5.
    # Undo that exact atlas convention, not a generic OpenGL half-texel mapping.
    h,w=texture.shape[:2]
    x=np.clip(uv[:,0]*w,0,w-1);y=np.clip((1-uv[:,1])*h,0,h-1)
    x0=x.astype(int);y0=y.astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    dx=(x-x0)[:,None];dy=(y-y0)[:,None]
    return ((1-dx)*(1-dy)*texture[y0,x0]+dx*(1-dy)*texture[y0,x1]+
            (1-dx)*dy*texture[y1,x0]+dx*dy*texture[y1,x1])


def geometry_audit(reference,vertices,triangles):
    """Require the same oriented triangles; tolerate OBJ six-decimal serialization."""
    from scipy.spatial import cKDTree
    rv=np.asarray(reference.vertices)
    # texrecon preserves vertex order; prefer it over ambiguous nearest matches
    # when TSDF vertices are coincident or closer than OBJ decimal precision.
    if vertices.shape==rv.shape and np.linalg.norm(vertices-rv,axis=1).max()<=1e-6:
        ids=np.arange(len(rv));dist=np.linalg.norm(vertices-rv,axis=1)
    else:
        dist,ids=cKDTree(rv).query(vertices)
    if dist.max()>1e-6:raise ValueError('Texturing changed mesh vertex positions')
    def canonical(t):
        t=np.asarray(t)
        starts=t.argmin(1)
        rolled=np.take_along_axis(t,(starts[:,None]+np.arange(3))%3,axis=1)
        return rolled[np.lexsort(rolled.T[::-1])]
    expected=canonical(np.asarray(reference.triangles));actual=canonical(ids[triangles])
    if expected.shape!=actual.shape or not np.array_equal(expected,actual):
        raise ValueError('Texturing changed oriented mesh triangle inventory')
    return {'triangles':len(triangles),'vertices_in_obj':len(vertices),'reference_vertices':len(rv),
            'vertex_position_max_error':float(dist.max()),'oriented_triangle_inventory_equal':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('obj','reference-mesh','mesh-metadata','calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--samples-per-segment',type=int,default=1)
    p.add_argument('--anchors',nargs='+',default=list(ANCHORS))
    a=p.parse_args()
    import open3d as o3d
    v,t,uv,material,textures,paths=load_obj(a.obj)
    audit=geometry_audit(o3d.io.read_triangle_mesh(str(a.reference_mesh)),v,t)
    mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v),o3d.core.Tensor(t))
    scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(mesh)
    calibration=json.loads(a.calibration.read_text());metadata=json.loads(a.mesh_metadata.read_text())
    targets=[normalize_frame(f,calibration,metadata) for f in calibration_path(calibration,a.anchors,a.samples_per_segment)]
    a.output.mkdir(parents=True,exist_ok=False)
    request={'obj_sha256':sha256(a.obj),'reference_mesh_sha256':sha256(a.reference_mesh),
             'mesh_metadata_sha256':sha256(a.mesh_metadata),'calibration_sha256':sha256(a.calibration),
             'script_sha256':sha256(Path(__file__)),'geometry_audit':audit,'targets':targets,
             'atlas_sha256':{str(p):sha256(p) for p in paths.values()},'uses_eval_rgb':False,
             'source_selection_camera_dependent':False,'screen_space_hole_fill':False}
    atomic_json(a.output/'request.json',request)
    rows=[]
    for i,f in enumerate(targets):
        pose=np.asarray(f['transform_matrix']);ext=np.linalg.inv(pose@np.diag([1.,-1.,-1.,1.])).astype(np.float32)
        K=np.array([[f['fl_x'],0,f['cx']],[0,f['fl_y'],f['cy']],[0,0,1]],np.float32)
        rays=scene.create_rays_pinhole(o3d.core.Tensor(K),o3d.core.Tensor(ext),int(f['w']),int(f['h']))
        hit=scene.cast_rays(rays);ids=hit['primitive_ids'].numpy();bary=hit['primitive_uvs'].numpy()
        valid=ids!=np.iinfo(np.uint32).max
        tri=ids[valid];b=bary[valid];weights=np.stack((1-b.sum(-1),b[:,0],b[:,1]),-1)
        pixel_uv=(uv[tri]*weights[...,None]).sum(1)
        rgb=np.zeros((len(tri),3));labels=material[tri]
        for j,texture in enumerate(textures):
            select=labels==j
            if select.any():rgb[select]=sample_atlas(texture,pixel_uv[select])
        output=np.zeros((*ids.shape,3),np.uint8);output[valid]=np.rint(rgb.clip(0,255)).astype(np.uint8)
        path=a.output/f'view_{i:04d}.png';Image.fromarray(output).save(path)
        row={'index':i,'physical_camera':f['physical_camera'],'render':str(path),'sha256':sha256(path),
             'geometric_hit_fraction':float(valid.mean())};rows.append(row)
        atomic_json(a.output/'result.json',{'state':'complete' if i==len(targets)-1 else 'running','views':rows,'geometry_audit':audit})
        print(f'view={i+1}/{len(targets)} camera={f["physical_camera"]}',flush=True)


if __name__=='__main__':main()
