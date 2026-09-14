"""Retiming-only visibility controls: unchanged periodic path, actual actor times."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import CALIBRATION, read, sha, atomic_json
from render_smooth_temporal_mesh_video import verify_request, calibration_pose
from render_patchmatch_camera_path import normalize_frame
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

PARENT=Path('/mnt/data/dec5_elevated_camera_dynamic_150')
OUTPUT=Path('/mnt/data/dec5_temporal_camera_phase_probe')
FRAMES=['000899','000973','001029','001083','001123','001191','001193','001195','001197']
PHASES=[-30,-15,0,15,30]


def shifted_camera(inventory,index,phase,calibration):
    target=inventory[index]; source=inventory[(index+phase)%len(inventory)]
    raw=calibration_pose(source['camera'],calibration,read(source['metadata']))
    camera=normalize_frame(raw,calibration,read(target['metadata']))
    if phase==0 and not np.allclose(camera['transform_matrix'],target['camera']['transform_matrix'],atol=1e-7):
        raise ValueError('Zero phase failed identity')
    return camera


def probe(output):
    request=verify_request(PARENT);cal=read(CALIBRATION)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'request.json',dict(parent_request_sha256=sha(PARENT/'request.json'),
        frames=FRAMES,phase_offsets=PHASES,actor_times_unchanged=True,
        identical_periodic_camera_path=True,mesh_changed=False,heldout_rgb_used=False,
        scope='CPU visibility canary, not full-video acceptance',script_sha256=sha(__file__)))
    results=[]
    for f in FRAMES:
        index=next(i for i,r in enumerate(request['inventory']) if r['frame_id']==f);r=request['inventory'][index]
        if sha(r['mesh'])!=r['mesh_sha256']:raise ValueError('Changed mesh')
        mesh=o3d.io.read_triangle_mesh(r['mesh']);mesh.compute_triangle_normals()
        scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles));normals=np.asarray(mesh.triangle_normals)
        dest=output/f;dest.mkdir();panel=Image.new('RGB',(540*len(PHASES),650));draw=ImageDraw.Draw(panel)
        for i,phase in enumerate(PHASES):
            camera=shifted_camera(request['inventory'],index,phase,cal)
            d,ids,_=camera_depth(scene,camera);hit=np.isfinite(d);rgb=np.zeros((*d.shape,3),np.uint8)
            shade=(.25+.75*np.abs(normals[ids[hit]]@np.array(camera['transform_matrix'])[:3,2])).clip(0,1)
            rgb[hit]=np.rint(shade[:,None]*np.array([205,210,216])).astype(np.uint8)
            im=Image.fromarray(np.rot90(rgb));name=f'phase_{phase:+d}'
            im.save(dest/(name+'.png'))
            # Save native head crop as evidence; overview alone is not the gate.
            im.crop((0,500,1080,1300)).save(dest/(name+'_head.png'))
            preview=im.crop((0,400,1080,1660)).resize((540,630),Image.Resampling.LANCZOS)
            panel.paste(preview,(540*i,20));draw.text((540*i+3,3),f'{f} camera phase {phase:+d}',fill='white')
            results.append(dict(frame=f,phase=phase,camera=camera,mesh_sha256=r['mesh_sha256'],
                clay_path=str(dest/(name+'.png')),clay_sha256=sha(dest/(name+'.png'))))
        panel.save(dest/'phase_overview.png');print('completed',f,flush=True)
    atomic_json(output/'results.json',dict(rows=results,visual_status='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    probe(p.parse_args().output)
