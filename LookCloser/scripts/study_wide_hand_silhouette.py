"""Six-vs-twelve camera envelopes with explicit unknown-domain face rejection."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy import ndimage
from joint_temporal_texture import read,sha,atomic_json,project
from local_silhouette_volume import combine_silhouettes,remove_box_caps
from silhouette_domain_surface import availability_bits,stable_domain_faces
import study_hand_silhouette_volume as base

ROOT=Path('/mnt/data/dec5_hand_silhouette_wide')
EXTRA=Path('/mnt/data/dec5_hand_silhouette_extra')
ORIGINAL=base.ROOT


def stage_extra():
    base.ROOT=EXTRA;base.OBS=Path('/mnt/data/dec5_wrist_wide_observations');base.stage()


def volume():
    import open3d as o3d
    from skimage.measure import marching_cubes
    original=ORIGINAL/base.FRAME;extra=EXTRA/base.FRAME;ROOT.mkdir(exist_ok=False)
    q=read(original/'request.json');r=read(extra/'request.json')
    assert q['settings']==r['settings'] and q['lower']==r['lower'] and q['upper']==r['upper']
    for request,root in [(q,original),(r,extra)]:
        for p,h in request['dependencies'].items():assert sha(p)==h
        assert sha(root/'silhouettes.npz')==request['silhouettes_sha256']
    data=dict(np.load(original/'silhouettes.npz'));data.update(dict(np.load(extra/'silhouettes.npz')))
    lower=np.array(q['lower']);upper=np.array(q['upper']);spacing=q['settings']['voxel_spacing']
    axes=[np.arange(a,b+spacing/2,spacing) for a,b in zip(lower,upper)]
    shape=tuple(len(a) for a in axes);count=int(np.prod(shape));actual_upper=[a[-1] for a in axes]
    request=dict(original_request_sha256=sha(original/'request.json'),extra_request_sha256=sha(extra/'request.json'),
        original=str(original),extra=str(extra),settings=q['settings'],lower=q['lower'],upper=q['upper'],
        grids_shape=list(shape),script_sha256=sha(__file__),domain_helper_sha256=sha(Path(__file__).with_name('silhouette_domain_surface.py')),
        heldout_used=False,unknown_domain_not_empty=True,production_updated=False)
    atomic_json(ROOT/'request.json',request);summaries=[]
    for name,records in [('six',q['cameras']),('twelve',q['cameras']+r['cameras'])]:
        rows=[r['camera'] for r in records];fields=[data[r['physical_camera']+'_field'] for r in rows]
        domains=[data[r['physical_camera']+'_domain'] for r in rows]
        field=np.empty(count,np.float32);bits=np.empty(count,np.uint16)
        for start in range(0,count,100000):
            end=min(start+100000,count);ijk=np.array(np.unravel_index(np.arange(start,end),shape)).T
            uv,z=project(lower+ijk*spacing,rows)
            sdf,_,_=combine_silhouettes(uv,z,fields,domains,3,0.)
            field[start:end]=sdf;bits[start:end]=availability_bits(uv,z,domains)
        field=field.reshape(shape);bits=bits.reshape(shape)
        v,t,_,_=marching_cubes(field,0,spacing=(spacing,)*3);v+=lower
        box=remove_box_caps(v,t,lower,actual_upper,spacing)
        stable=stable_domain_faces(v,t,lower,spacing,bits,3)
        for mode,keep in [('raw',box),('domain_safe',box&stable)]:
            dest=ROOT/name/mode;dest.mkdir(parents=True)
            mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[keep]))
            mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'hull.ply'),mesh)
            result=dict(cameras=[r['physical_camera'] for r in rows],mode=mode,vertices=len(mesh.vertices),triangles=len(mesh.triangles),
                removed_box_faces=int((~box).sum()),domain_edge_faces=int((box&~stable).sum()),mesh_sha256=sha(dest/'hull.ply'),
                occupancy_voxels=int((field>=0).sum()),surface_is_inferred_envelope=True,visual_status='pending')
            atomic_json(dest/'result.json',result);summaries.append(dict(name=name,**result))
            print(name,mode,len(mesh.triangles),'removed domain',result['domain_edge_faces'],flush=True)
        np.savez_compressed(ROOT/name/'field.npz',field=field,bits=bits,lower=lower,spacing=spacing)
    atomic_json(ROOT/'result.json',dict(request_sha256=sha(ROOT/'request.json'),variants=summaries,video_updated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage_extra','volume']);a=p.parse_args()
    {'stage_extra':stage_extra,'volume':volume}[a.action]()
