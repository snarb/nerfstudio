"""Lower-row stereo observations focused on the missing wrist/forearm.

Reuses fixed-profile RGB, fixed calibration, rectification, inference and the
spatial-anchor validator. Rough hand landmarks choose framing only, not depth.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from calibrated_depth_witness import load_images
from calibrated_stereo_rectification import rectify_local_pair

ROOT=Path('/mnt/data/dec5_foundation_lower_forearm')
FRAME='001037'
PAIRS=[('E004_D005_1210L4','F004_D005_1210KW'),('E004_E005_1210WX','F004_E005_1210FP')]


def stage():
    output=ROOT/FRAME; output.mkdir(parents=True,exist_ok=False)
    rows,_,metadata=cameras(FRAME);lookup={r['physical_camera']:r for r in rows}
    images,_,receipt=load_images(FRAME)
    landmarks=Path('/mnt/data/dec5_hand_landmark_triangulation')/FRAME/'evidence.npz'
    points=np.load(landmarks);assert points['good'][0] and points['good'][9]
    # Move framing toward the forearm, away from the fingertip mean used before.
    wrist=points['points'][0];focus=wrist+.5*(wrist-points['points'][9])
    guards=Path('/mnt/data/dec5_dynamic_grid_150_guard_v2/foreground_guard')/FRAME
    masks=np.load(guards/'masks.npz')['masks'];names=read(guards/'cameras.json')
    deps={str(p):sha(p) for p in [landmarks,Path(metadata),guards/'masks.npz',guards/'cameras.json']}
    observation=[];rgb={};skin={}
    for name in sorted({n for pair in PAIRS for n in pair}):
        im=np.rot90(images[name]).copy();rgb[name]=im
        uv,_=project(np.vstack((points['points'][points['good']],focus)),[lookup[name]])
        portrait=np.column_stack((uv[0,:,1],1919-uv[0,:,0]))
        lo=np.floor(portrait.min(0)-[100,100]).astype(int);hi=np.ceil(portrait.max(0)+[100,180]).astype(int)
        lo=np.maximum(lo,0);hi=np.minimum(hi,[1080,1920]);window=np.zeros(im.shape[:2],np.uint8)
        window[lo[1]:hi[1],lo[0]:hi[0]]=1
        warm=im[...,0].astype(float)-im[...,2].astype(float)>8
        mask=warm&np.rot90(masks[names.index(name)]).astype(bool)&window.astype(bool)
        mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        if count<=1:raise ValueError('Empty warm-object region: '+name)
        mask=labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        skin[name]=binary_fill_holes(mask).astype(np.uint8)&window
        Image.fromarray(im).save(output/(name+'.png'))
        over=im.copy();contours,_=cv2.findContours(skin[name],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over,contours,-1,(0,255,0),2)
        Image.fromarray(over).crop((0,1300,650,1920)).save(output/(name+'_mask_review.png'))
        observation.append(dict(camera=lookup[name],image_sha256=sha(output/(name+'.png')),
            source_sha256=receipt['source_rgb_hashes'][lookup[name]['file_path']],region_box=[*lo.tolist(),*hi.tolist()],
            region_pixels=int(skin[name].sum())))
    records=[]
    for left,right in PAIRS:
        cal,maps=rectify_local_pair(lookup[left],lookup[right],focus)
        dest=output/(left[:6]+'_'+right[:6]);dest.mkdir()
        rectified=[cv2.remap(rgb[n],*m,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for n,m in zip([left,right],maps)]
        regions=[cv2.remap(skin[n],*m,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT) for n,m in zip([left,right],maps)]
        for label,im in zip(['left','right'],rectified):Image.fromarray(im).save(dest/(label+'.png'))
        np.savez_compressed(dest/'calibration.npz',**{k:v for k,v in cal.items() if k!='size'},
            cropped_intrinsic=cal['P1'][:,:3],crop=np.array([0,0,768,768]),left_mask=regions[0],right_mask=regions[1],
            left_map_x=maps[0][0],left_map_y=maps[0][1],right_map_x=maps[1][0],right_map_y=maps[1][1])
        panel=Image.new('RGB',(1536,793));draw=ImageDraw.Draw(panel)
        for i,im in enumerate(rectified):panel.paste(Image.fromarray(im),(i*768,25))
        for y in range(25,793,80):draw.line((0,y,1535,y),fill='lime')
        draw.text((4,5),left+' / '+right+' lower-forearm rectification',fill='white')
        panel.save(dest/'rectification_review.png')
        records.append(dict(left=left,right=right,directory=str(dest),baseline=cal['baseline'],
            disparity_offset=cal['disparity_offset'],crop=[0,0,768,768],rough_landmarks_for_framing_only=True,
            hashes={n:sha(dest/n) for n in ['left.png','right.png','calibration.npz','rectification_review.png']}))
    atomic_json(output/'request.json',dict(frame=FRAME,pairs=records,source_hashes=deps,
        rgb_receipt=receipt,observations=observation,focus=focus.tolist(),
        script_sha256=sha(__file__),rectification_sha256=sha(Path(__file__).with_name('calibrated_stereo_rectification.py')),
        physical_cameras_fixed=True,heldout_used=False,production_updated=False,
        mask_use='Post-hoc warm hand/forearm region, not a precise anatomical segmentation; RGB model inputs unmasked',
        previous_crop_reused=False,visual_status='pending'))
    print('staged',[(r['left'],r['right']) for r in records],flush=True)


def infer():
    import infer_foundation_hand_stereo as engine
    engine.ROOT=ROOT/FRAME;engine.run()


def bias():
    import study_foundation_anchor_bias as engine
    engine.ROOT=ROOT/'bias';engine.SOURCES=[ROOT/FRAME];engine.run()
    atomic_json(ROOT/'bias_adapter.json',dict(script_sha256=sha(__file__),producer_sha256=sha(engine.__file__),
        stage_request_sha256=sha(ROOT/FRAME/'request.json'),bias_request_sha256=sha(ROOT/'bias/request.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['stage','infer','bias'])
    {'stage':stage,'infer':infer,'bias':bias}[parser.parse_args().action]()
