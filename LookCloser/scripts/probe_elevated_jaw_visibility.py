"""CPU-only camera-height canary for residual jaw holes; no mesh repair claim."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_fill_holes, label
from joint_temporal_texture import CALIBRATION, cameras, read, sha, atomic_json
from render_smooth_temporal_mesh_video import verify_request, calibration_pose
from render_patchmatch_camera_path import normalize_frame
from expanded_head_camera_flight import polygon_weights
from screen_travel_camera_flight import portrait_projection
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

PARENT=Path('/mnt/data/dec5_elevated_camera_dynamic_150')
OUTPUT=Path('/mnt/data/dec5_elevated_jaw_height_probe')
FRAMES=['000899','001191','001193','001195','001197']
HEIGHTS=[0.,.1,.25,.4]


def camera_at_height(record, report, delta):
    cal=read(CALIBRATION); metadata=read(record['metadata']); old=record['camera']
    xy=np.array(old['rig_offset_xy']); xy[1]+=delta
    lookup={r['physical_camera']:r for r in cal['frames']}
    anchors=[normalize_frame(lookup[n],cal,metadata) for n in report['anchors']]
    positions=np.array([r['transform_matrix'] for r in anchors])[:,:3,3]
    _,_,reference_metadata=cameras('000973')
    reference_pose=np.eye(4); reference_pose[:3,3]=report['fixed_target']
    raw=calibration_pose(dict(transform_matrix=reference_pose.tolist()),cal,read(reference_metadata))
    target=np.array(normalize_frame(raw,cal,metadata)['transform_matrix'])[:3,3]
    up=np.array(normalize_frame(next(r for r in cal['frames'] if r['physical_camera'].startswith('H004_C005')),cal,metadata)['transform_matrix'])[:3,0]
    weights=polygon_weights(xy); position=weights@positions
    z=position-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y)
    look=np.column_stack((np.cross(y,z),y,z));desired=np.array(old['scene_landmark_portrait_xy'])
    native=np.array([old['w']-1-desired[1],desired[0]])
    ray=np.array([(native[0]-old['cx'])/old['fl_x'],-(native[1]-old['cy'])/old['fl_y'],-1.]);ray/=np.linalg.norm(ray)
    pose=np.eye(4);pose[:3,3]=position
    pose[:3,:3]=look@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
    row=deepcopy(old);row.update(transform_matrix=pose.tolist(),rig_offset_xy=xy.tolist(),convex_weights=weights.tolist())
    if not np.allclose(portrait_projection(target,row),desired,atol=1e-6):raise ValueError('Composition shifted')
    if delta==0 and not np.allclose(np.array(old['transform_matrix']),pose,atol=1e-7):
        raise ValueError('Zero-height control does not reproduce published camera')
    return row


def run(output):
    parent=verify_request(PARENT);output.mkdir(parents=True,exist_ok=False)
    spec=dict(parent_request_sha256=sha(PARENT/'request.json'),frames=FRAMES,heights_in_rig_rows=HEIGHTS,
        hypothesis='small camera elevation can occlude tiny residual jaw holes without geometry edits',
        camera_movement_preserved=True,geometry_changed=False,heldout_rgb_used=False,
        intrinsics_changed=False,post_render_crop_for_movie=False,script_sha256=sha(__file__))
    atomic_json(output/'request.json',spec);results=[]
    for frame in FRAMES:
        record=next(r for r in parent['inventory'] if r['frame_id']==frame)
        if sha(record['mesh'])!=record['mesh_sha256']:raise ValueError('Changed geometry')
        mesh=o3d.io.read_triangle_mesh(record['mesh']);mesh.compute_triangle_normals()
        scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles));normal=np.asarray(mesh.triangle_normals)
        dest=output/frame;dest.mkdir();crops=[]
        for delta in HEIGHTS:
            camera=camera_at_height(record,parent['camera_path_report'],delta)
            d,ids,_=camera_depth(scene,camera);hit=np.isfinite(d);rgb=np.zeros((*hit.shape,3),np.uint8)
            light=np.array(camera['transform_matrix'])[:3,2]
            shade=(.25+.75*np.abs(normal[ids[hit]]@light)).clip(0,1)
            rgb[hit]=np.rint(shade[:,None]*np.array([205,210,216])).astype(np.uint8)
            portrait=Image.fromarray(np.rot90(rgb));name=f'height_{delta:g}'
            portrait.save(dest/(name+'.png'))
            crop=portrait.crop((350,930,710,1280));crop.save(dest/(name+'_jaw.png'));crops.append(crop)
            phit=np.rot90(hit);holes=binary_fill_holes(phit)&~phit
            labels,count=label(holes);components=[]
            for idx in range(1,count+1):
                yy,xx=np.where(labels==idx)
                if len(xx) and 350<=xx.mean()<710 and 930<=yy.mean()<1280:
                    components.append(dict(area=len(xx),bbox=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]))
            results.append(dict(frame=frame,height=delta,camera=camera,mesh_sha256=record['mesh_sha256'],
                jaw_enclosed_miss_components=components,not_anatomical_mask_or_quality_metric=True,
                clay_path=str(dest/(name+'.png')),clay_sha256=sha(dest/(name+'.png'))))
            print(frame,delta,components,flush=True)
        panel=Image.new('RGB',(360*4,374));draw=ImageDraw.Draw(panel)
        for i,(delta,im) in enumerate(zip(HEIGHTS,crops)):
            panel.paste(im,(360*i,24));draw.text((360*i+3,4),f'camera +{delta:g} rig rows',fill='white')
        panel.save(dest/'jaw_comparison.png')
    atomic_json(output/'results.json',dict(rows=results,visual_status='pending',rgb_rendered=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    run(p.parse_args().output)
