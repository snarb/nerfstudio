"""Camera-independent bounded 3D contour closure pilot, never production defaults.

Geometric proposals are inferred, not new PatchMatch measurements. The selected
camera/spot is used only afterward for diagnosis, never for proposing triangles.
"""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
import mapbox_earcut
from shapely.geometry import Polygon
from PIL import Image,ImageDraw
from boundary_cycle_blocks import cyclic_boundary_blocks
from joint_temporal_texture import read,sha,atomic_json,cameras
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

PARENT=Path('/mnt/data/dec5_elevated_camera_dynamic_150')
PHASE=Path('/mnt/data/dec5_phase30_dynamic_150')
FRAMES=['001193','001195']
SETTINGS=dict(max_edges=18,max_extent=.003,max_gap=.0015,min_detour_ratio=2.5,
              max_plane_rmse=.0002,min_head_x=-.03,max_occlusion_depth=.001)


def propose(v,t,settings=SETTINGS):
    loops,_=cyclic_boundary_blocks(t)
    existing_edges=set(map(tuple,np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)))
    existing_faces=set(map(tuple,np.sort(t,axis=1)))
    candidates=[]
    for li,loop in enumerate(loops):
        if len(loop)<settings['max_edges']:continue
        for start in range(len(loop)):
            for length in range(4,settings['max_edges']+1):
                arc=loop[(start+np.arange(length))%len(loop)];points=v[arc]
                if points[:,0].min()<=settings['min_head_x'] or np.ptp(points,axis=0).max()>settings['max_extent']:break
                gap=np.linalg.norm(points[-1]-points[0])
                if gap<1e-8 or gap>settings['max_gap']:continue
                path=np.linalg.norm(np.diff(points,axis=0),axis=1).sum();ratio=path/gap
                if ratio<settings['min_detour_ratio']:continue
                chord=tuple(sorted((int(arc[0]),int(arc[-1]))))
                if chord in existing_edges:continue
                centered=points-points.mean(0);_,_,basis=np.linalg.svd(centered,full_matrices=False)
                rmse=float(np.sqrt(np.mean((centered@basis[2])**2)))
                if rmse>settings['max_plane_rmse']:continue
                xy=centered@basis[:2].T;polygon=Polygon(xy)
                if not polygon.is_valid or polygon.area<1e-10:continue
                faces=arc[mapbox_earcut.triangulate_float64(np.ascontiguousarray(xy),np.array([length],np.uint32)).reshape(-1,3)]
                if any(tuple(f) in existing_faces for f in np.sort(faces,axis=1)):continue
                directed=set(map(tuple,faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)))
                if (arc[0],arc[1]) in directed:faces=faces[:,::-1]
                edges=set(map(tuple,np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)))
                arc_edges={tuple(sorted((int(a),int(b)))) for a,b in zip(arc[:-1],arc[1:])}
                if (edges-arc_edges)&existing_edges:continue
                candidates.append((ratio,li,start,length,arc,faces,edges,arc_edges,rmse,gap))
    candidates.sort(key=lambda c:(-c[0],c[1],c[2],c[3]));used=set();added=[];notes=[]
    for ratio,li,start,length,arc,faces,edges,arc_edges,rmse,gap in candidates:
        if used&arc_edges or (edges-arc_edges)&existing_edges:continue
        used|=arc_edges;existing_edges|=edges;added.append(faces)
        notes.append(dict(loop=li,start=start,length=length,vertices=arc.tolist(),
            ratio=float(ratio),plane_rmse=rmse,gap=float(gap),triangles=len(faces)))
    tt=np.concatenate([t,*added]) if added else t.copy()
    edges,counts=np.unique(np.sort(tt[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    _,oldcounts=np.unique(np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    if (counts>2).sum()>(oldcounts>2).sum():raise ValueError('New nonmanifold edges')
    if not np.array_equal(tt[:len(t)],t):raise ValueError('Original triangles altered')
    return tt,notes


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    parent=read(PARENT/'request.json');phase=read(PHASE/'request.json')
    request=dict(frames=FRAMES,settings=SETTINGS,source_request_sha256=sha(PARENT/'request.json'),
        phase_request_sha256=sha(PHASE/'request.json'),script_sha256=sha(__file__),
        helper_sha256=sha(Path(__file__).with_name('boundary_cycle_blocks.py')),
        geometry_proposals_use_camera=False,inferred_geometry=True,production_changed=False)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Request mismatch')
    atomic_json(output/'request.json',request)
    spots={r['frame_id']:r for r in read('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')['selected_components']}
    for frame in FRAMES:
        root=output/frame;root.mkdir(exist_ok=True)
        row=next(r for r in parent['inventory'] if r['frame_id']==frame)
        if sha(row['mesh'])!=row['mesh_sha256']:raise ValueError('Changed source')
        mesh=o3d.io.read_triangle_mesh(row['mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
        tt,notes=propose(v,t)
        candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(tt))
        candidate.compute_vertex_normals();o3d.io.write_triangle_mesh(str(root/'candidate.ply'),candidate)
        old=scene_for(v,t);new=scene_for(v,tt)
        pr=next(r for r in phase['inventory'] if r['frame_id']==frame)
        train,_,_=cameras(frame)
        center=np.array(row['camera']['transform_matrix'])[:3,3]
        nearby=sorted(train,key=lambda c:np.linalg.norm(np.array(c['transform_matrix'])[:3,3]-center))[:3]
        controls=[('old_moving',row['camera']),('phase_moving',pr['camera']),*[(c['physical_camera'],c) for c in nearby]]
        records=[]
        for name,cam in controls:
            d0,ids0,_=camera_depth(old,cam);d1,ids1,_=camera_depth(new,cam)
            valid0=np.isfinite(d0);valid1=np.isfinite(d1);newvis=valid1&~valid0
            occlusion=valid0&valid1&(d1<d0-SETTINGS['max_occlusion_depth'])
            normals=np.asarray(candidate.triangle_normals);valid=ids1<len(tt)
            color=np.zeros((*ids1.shape,3),np.uint8)
            lighting=np.abs(normals@np.array([.3,.4,.866]));color[valid]=(60+170*lighting[ids1[valid],None]).astype(np.uint8)
            color[valid&(ids1>=len(t))]=[240,60,50]
            portrait=np.rot90(color);Image.fromarray(portrait).save(root/f'{name}_clay_added.png')
            rec=dict(camera=name,newly_visible=int(newvis.sum()),old_occlusions_gt_threshold=int(occlusion.sum()),
                added_surface_pixels=int((valid&(ids1>=len(t))).sum()))
            if name=='old_moving':
                x0,y0,x1,y1=spots[frame]['bbox_inclusive'];b0=np.rot90(valid0);b1=np.rot90(valid1)
                rec['selected_spot_old_misses']=int((~b0[y0:y1+1,x0:x1+1]).sum())
                rec['selected_spot_remaining_misses']=int((~b1[y0:y1+1,x0:x1+1]).sum())
                crop=(x0-65,y0-65,x1+66,y1+66)
                Image.fromarray(portrait).crop(crop).save(root/'spot_added_native.png')
            records.append(rec)
        atomic_json(root/'result.json',dict(source_mesh_sha256=row['mesh_sha256'],candidate_sha256=sha(root/'candidate.ply'),
            original_vertices_and_triangles_unchanged=True,added_triangles=len(tt)-len(t),proposals=notes,
            controls=records,production_accepted=False,visual_status='pending'))
        print(frame,len(notes),len(tt)-len(t),records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_jaw_3d_boundary_notches'))
    run(p.parse_args().output)
