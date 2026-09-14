"""Inspect real train silhouettes near the questioned moving-camera viewpoints."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import cameras,read,sha,exr,display,ROOT,atomic_json

def build():
    parent=Path('/mnt/data/dec5_screen_travel_dynamic_150_v2');out=Path('/mnt/data/dec5_expanded_head_repair_diagnosis');request=read(parent/'request.json')
    profiles=read(ROOT/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']));exposure=read(ROOT/'exposure.json')['fixed_exposure_gain']
    for frame in ['001083','001123']:
        record=next(r for r in request['inventory'] if r['frame_id']==frame);rows,_,_=cameras(frame)
        centers=np.array([r['transform_matrix'] for r in rows])[:,:3,3];target=np.array(record['camera']['transform_matrix'])[:3,3]
        selected=np.argsort(np.linalg.norm(centers-target,axis=1))[:3];panel=Image.new('RGB',(1620,984));draw=ImageDraw.Draw(panel);evidence=[]
        for j,i in enumerate(selected):
            row=rows[i];rgb=display(exr(row['file_path'])*gains[row['physical_camera']],exposure);im=Image.fromarray(np.rot90(np.rint(rgb*255).clip(0,255).astype('uint8')))
            path=out/frame/f'train_{j}.png';im.save(path);panel.paste(im.resize((540,960)),(j*540,24));draw.text((j*540+3,4),row['physical_camera'],fill='white')
            evidence.append(dict(camera=row['physical_camera'],source_sha256=sha(row['file_path']),path=str(path),sha256=sha(path)))
        panel.save(out/frame/'train_evidence.png');atomic_json(out/frame/'train_evidence.json',dict(evidence=evidence,heldout_rgb_used=False))

if __name__=='__main__':build()
