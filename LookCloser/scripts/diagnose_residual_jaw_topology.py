"""Trace already-confirmed jaw misses to 3D boundary topology; no mesh edits."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, project
from local_mesh_repair import boundary_loops
from boundary_cycle_blocks import cyclic_boundary_blocks

PARENT = Path('/mnt/data/dec5_elevated_camera_dynamic_150')
SPOTS = Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')


def analyze(output):
    parent = read(PARENT/'request.json')
    request = dict(parent_sha256=sha(PARENT/'request.json'), spots_sha256=sha(SPOTS),
                   script_sha256=sha(__file__), cycle_helper_sha256=sha(Path(__file__).with_name('boundary_cycle_blocks.py')), geometry_changed=False)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json') != request:
        raise ValueError('Diagnostic request mismatch')
    atomic_json(output/'request.json', request)
    results = []
    for spot in read(SPOTS)['selected_components']:
        frame = spot['frame_id']; row = next(r for r in parent['inventory'] if r['frame_id']==frame)
        if sha(row['mesh']) != row['mesh_sha256']:
            raise ValueError('Changed source mesh')
        mesh = o3d.io.read_triangle_mesh(row['mesh'])
        v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        uv, z = project(v, [row['camera']])
        portrait = np.column_stack((uv[0,:,1], 1919-uv[0,:,0]))
        edges, counts = np.unique(np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),
                                  axis=0, return_counts=True)
        boundary = edges[counts==1]
        loops, rejected = boundary_loops(t)
        labels = np.full(len(v), -1, int)
        components = [*loops, *[np.array(c) for c in rejected]]
        for i, comp in enumerate(components): labels[comp] = i
        x0,y0,x1,y1 = spot['bbox_inclusive']; center = np.array([(x0+x1)/2,(y0+y1)/2])
        midpoint = portrait[boundary].mean(1)
        distance = np.linalg.norm(midpoint-center,axis=1)
        near = np.argsort(distance)[:40]
        cycle_loops, cycle_stats = cyclic_boundary_blocks(t)
        nearby_cycles=[]
        for i,loop in enumerate(cycle_loops):
            d=float(np.linalg.norm(portrait[loop]-center,axis=1).min())
            if d<25:
                nearby_cycles.append(dict(index=i,vertex_ids=loop.tolist(),extent=np.ptp(v[loop],axis=0).tolist(),
                    minimum_projected_distance=d,center=v[loop].mean(0).tolist()))
        candidates=[]
        for i in np.unique(labels[boundary[near].ravel()]):
            if i<0: continue
            comp=components[i]; pts=v[comp]
            candidates.append(dict(component=int(i),simple_loop=bool(i<len(loops)),
                vertices=len(comp),vertex_ids=comp.tolist(),extent=np.ptp(pts,axis=0).tolist(),
                center=pts.mean(0).tolist(),minimum_projected_distance=float(np.linalg.norm(portrait[comp]-center,axis=1).min()),
                projected_bbox=[*portrait[comp].min(0).tolist(),*portrait[comp].max(0).tolist()]))
        crop=[x0-65,y0-65,x1+66,y1+66]
        source=PARENT/'frames'/frame/'frame.png'
        rgb=Image.open(source).convert('RGB').crop(crop)
        panel=Image.new('RGB',(rgb.width*2,rgb.height));panel.paste(rgb,(0,0));panel.paste(rgb,(rgb.width,0))
        draw=ImageDraw.Draw(panel); palette=['cyan','magenta','yellow','lime','orange','red']
        for e in near:
            a,b=boundary[e]; color=palette[int(labels[a])%len(palette)]
            q=portrait[[a,b]]-np.array(crop[:2])+np.array([rgb.width,0])
            draw.line([tuple(p) for p in q],fill=color,width=1)
        panel.save(output/f'{frame}_boundary_native.png')
        record=dict(frame_id=frame,mesh=row['mesh'],mesh_sha256=row['mesh_sha256'],
            render_sha256=sha(source),spot=spot,crop=crop,remaining_simple_loops=len(loops),
            rejected_components=len(rejected),near_boundary_components=candidates,
            nearest_boundary_midpoint_distance=float(distance.min()),
            cycle_stats=cycle_stats,nearby_cycles=nearby_cycles,
            panel_sha256=sha(output/f'{frame}_boundary_native.png'))
        atomic_json(output/f'{frame}.json',record);results.append(record)
        print(frame,[(c['component'],c['simple_loop'],c['vertices'],round(c['minimum_projected_distance'],2)) for c in candidates],flush=True)
        print('cycles',[(c['index'],len(c['vertex_ids']),c['extent'],c['minimum_projected_distance']) for c in nearby_cycles],flush=True)
    atomic_json(output/'result.json',dict(request_sha256=sha(output/'request.json'),records=results,
        note='2D proximity alone cannot establish surface depth or authorize filling; inspect 3D and train support next.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_residual_jaw_topology'))
    analyze(p.parse_args().output)
