"""Isolate camera framing using identical static geometry; not a dynamic video."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION,cameras
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from diffusion_mesh_repair import BASE,scene_for,render_atlas


def run(output):
    old_root=Path('/mnt/data/dec5_replayed_4x4_dynamic_150_v2')
    old=verify_request(old_root);new=verify_request(output);cal=read(CALIBRATION)
    _,_,meta=cameras('000973');metadata=read(meta)
    atlas=dict(np.load(BASE/'atlas_geometry.npz'));texture=np.array(Image.open(BASE/'texture_joint.png').convert('RGB'))
    scene=scene_for(atlas['vertices'],atlas['triangles']);panel=Image.new('RGB',(1080,1032));draw=ImageDraw.Draw(panel)
    root=output/'framing_diagnosis';root.mkdir(exist_ok=True);variants={}
    for line,(name,request) in enumerate([('old_centered',old),('new_screen_travel',new)]):
        centroids=[];images=[]
        for j,index in enumerate([0,37,75,112]):
            record=request['inventory'][index]
            row=normalize_frame(calibration_pose(record['camera'],cal,read(record['metadata'])),cal,metadata)
            rgb,depth,_,_=render_atlas(scene,atlas,texture,row)
            portrait=Image.fromarray(np.rot90(rgb));path=root/f'{name}_{index:03d}.png';portrait.save(path)
            mask=np.rot90(np.isfinite(depth));yy,xx=np.nonzero(mask);centroids.append([float(xx.mean()),float(yy.mean())])
            x=j*270;y=line*516;panel.paste(portrait.resize((270,480)),(x,y+28))
            draw.text((x+3,y+4),f'{name} phase={index}/150',fill='white')
            images.append(dict(path=str(path),sha256=sha(path)))
        variants[name]=dict(static_mesh_hit_centroids=centroids,centroid_span_pixels=np.ptp(centroids,axis=0).tolist(),images=images)
    # Check actual old retained images, not only a search for the word crop.
    old_crop_checks=[]
    for index in [0,37,75,112,149]:
        frame=old['ordered_frame_ids'][index];path=old_root/'frames'/frame
        same=np.array_equal(np.array(Image.open(path/'frame.png')),np.rot90(np.array(Image.open(path/'prediction_native.png'))))
        if not same:raise ValueError('Old postprocessing is not rotation-only')
        old_crop_checks.append(dict(frame_id=frame,rotation_only=True))
    panel.save(root/'comparison.png')
    atomic_json(root/'result.json',dict(static_diagnostic_only=True,source_time='000973',
        old_request_sha256=sha(old_root/'request.json'),new_request_sha256=sha(output/'request.json'),
        atlas_sha256=sha(BASE/'atlas_geometry.npz'),texture_sha256=sha(BASE/'texture_joint.png'),
        variants=variants,old_rotation_only_checks=old_crop_checks,script_sha256=sha(__file__),visual_status='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_screen_travel_dynamic_150_v2'))
    run(p.parse_args().output)
