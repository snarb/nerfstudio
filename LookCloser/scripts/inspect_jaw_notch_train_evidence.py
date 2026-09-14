"""Real-train RGB around rejected notch proposals; diagnostic, not fitting."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,project,display,exr,ROOT
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from study_jaw_boundary_notches import PARENT,FRAMES
from guard_jaw_boundary_notches import SOURCE


def inspect():
    output=Path('/mnt/data/dec5_jaw_3d_train_evidence');output.mkdir(exist_ok=True)
    parent=read(PARENT/'request.json');profiles=read(ROOT/'camera_profiles.json')
    gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']));exposure=read(ROOT/'exposure.json')['fixed_exposure_gain']
    records=[]
    for frame in FRAMES:
        row=next(r for r in parent['inventory'] if r['frame_id']==frame)
        base=o3d.io.read_triangle_mesh(row['mesh']);candidate=o3d.io.read_triangle_mesh(str(SOURCE/frame/'candidate.ply'))
        v,t=np.asarray(candidate.vertices),np.asarray(candidate.triangles);n=len(base.triangles)
        admission=read(f'/mnt/data/dec5_jaw_3d_boundary_guarded_attribution/{frame}/spot_admission.json')
        selected=np.array([r['proposal']+n for r in admission['triangles']]);center=v[t[selected]].reshape(-1,3).mean(0)
        scene=scene_for(v,t);old=scene_for(np.asarray(base.vertices),np.asarray(base.triangles))
        train,_,_=cameras(frame);wanted={'M004_B005_12109O','H004_D005_1210SF','F004_E005_1210FP'}
        chosen=[c for c in train if c['physical_camera'] in wanted]
        panel=Image.new('RGB',(960,3*350));draw=ImageDraw.Draw(panel)
        for i,cam in enumerate(chosen):
            rgb=np.rint(display(exr(cam['file_path'])*np.array(gains[cam['physical_camera']]),exposure)*255).clip(0,255).astype(np.uint8)
            d0,_,_=camera_depth(old,cam);d1,ids,_=camera_depth(scene,cam)
            proposed=np.isin(ids,selected);overlay=rgb.copy();overlay[proposed]=[255,30,40]
            oldhit=np.isfinite(d0);comparison=rgb.copy();comparison[proposed&oldhit]=[0,200,255]
            uv,_=project(center[None],[cam]);x,y=np.rint([uv[0,0,1],1919-uv[0,0,0]]).astype(int)
            box=tuple(map(int,(x-160,y-160,x+160,y+160)))
            for j,im in enumerate([rgb,overlay,comparison]):
                panel.paste(Image.fromarray(np.rot90(im)).crop(box),(j*320,i*350+30))
            draw.text((4,i*350+5),cam['physical_camera']+'  GT | RED proposed surface | CYAN old-hit overlap',fill='white')
            records.append(dict(frame=frame,camera=cam,source_sha256=sha(cam['file_path']),crop=box,
                proposal_pixels=int(proposed.sum()),overlap_pixels=int((proposed&oldhit).sum())))
        path=output/f'{frame}.png';panel.save(path)
    atomic_json(output/'request.json',dict(script_sha256=sha(__file__),source_request_sha256=sha(SOURCE/'request.json'),
        profiles_sha256=sha(ROOT/'camera_profiles.json'),exposure_sha256=sha(ROOT/'exposure.json'),
        records=records,heldout_used=False,fit_or_geometry_changes=False))


if __name__=='__main__':inspect()
