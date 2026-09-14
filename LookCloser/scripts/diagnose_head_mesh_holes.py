"""Read-only geometry-vs-texture and boundary-loop evidence for head defects."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,project
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from local_mesh_repair import boundary_loops

PARENT=Path('/mnt/data/dec5_screen_travel_dynamic_150_v2')
OUTPUT=Path('/mnt/data/dec5_expanded_head_repair_diagnosis')

def diagnose(frame):
    record=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    out=OUTPUT/frame;out.mkdir(parents=True,exist_ok=True)
    rgb=Image.open(PARENT/'frames'/frame/'frame.png').convert('RGB')
    panel=Image.new('RGB',(1620,984));draw=ImageDraw.Draw(panel)
    panel.paste(rgb.resize((540,960)),(0,24));draw.text((4,4),'Actual RGB',fill='white')
    reports={}
    for column,key in enumerate(['untrimmed_mesh','mesh'],1):
        mesh=o3d.io.read_triangle_mesh(record[key]);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
        depth,ids,_=camera_depth(scene_for(v,t),record['camera']);hit=np.isfinite(depth)
        normal=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])
        normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
        light=np.array(record['camera']['transform_matrix'])[:3,2]
        shade=(.25+.75*np.abs(normal@light))*230
        clay=np.zeros((*hit.shape,3),np.uint8);clay[hit]=shade[ids[hit],None].astype(np.uint8)
        portrait=Image.fromarray(np.rot90(clay));portrait.save(out/f'{key}_clay.png')
        loops,rejected=boundary_loops(t);overlay=rgb.copy();d=ImageDraw.Draw(overlay);entries=[]
        for i,loop in enumerate(loops):
            points=v[loop];uv,z=project(points,[record['camera']]);xy=np.column_stack((uv[0,:,1],1919-uv[0,:,0]))
            center=xy.mean(0)
            if len(loop)>15 and 400<center[1]<1200:
                d.line([tuple(p) for p in xy]+[tuple(xy[0])],fill='red',width=2)
                d.text(tuple(center),str(i),fill='yellow')
                entries.append(dict(loop=i,count=len(loop),vertices=loop.tolist(),center=points.mean(0).tolist(),extent=np.ptp(points,axis=0).tolist(),portrait_center=center.tolist(),portrait_extent=np.ptp(xy,axis=0).tolist()))
        overlay.save(out/f'{key}_loops.png');np.savez_compressed(out/f'{key}_depth.npz',depth=np.where(hit,depth,0))
        panel.paste(portrait.resize((540,960)),(column*540,24));draw.text((column*540+4,4),key,fill='white')
        reports[key]=dict(mesh=record[key],sha256=sha(record[key]),loops=entries,all_loops=len(loops),rejected_components=len(rejected))
    panel.save(out/'comparison.png');atomic_json(out/'diagnosis.json',reports)
    print(frame,[(k,len(v['loops'])) for k,v in reports.items()],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frames',nargs='+',default=['001083','001123']);a=p.parse_args()
    for frame in a.frames:diagnose(frame)
