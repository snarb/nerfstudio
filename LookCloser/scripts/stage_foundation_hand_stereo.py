"""Rectified fixed-profile real stereo pairs for a research depth canary."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras
from calibrated_stereo_rectification import rectify_local_pair

ROOT=Path('/mnt/data/dec5_foundation_hand_stereo')
FRAME='001037'
PAIRS=[('F004_A005_12103K','G004_A005_121071'),('E004_C005_1210YM','F004_C005_121059')]


def stage():
    root=ROOT/FRAME;root.mkdir(parents=True,exist_ok=False)
    rows,_,metadata=cameras(FRAME);lookup={r['physical_camera']:r for r in rows}
    archives=[Path('/mnt/data/dec5_wrist_observations'),Path('/mnt/data/dec5_wrist_wide_observations')]
    observations={};images={};masks={};deps={str(metadata):sha(metadata)}
    for archive in archives:
        result=read(archive/FRAME/'result.json')
        deps[str(archive/FRAME/'result.json')]=sha(archive/FRAME/'result.json')
        for r in result['records']:
            n=r['camera']['physical_camera'];p=archive/FRAME/(n+'.png')
            if n not in {n for pair in PAIRS for n in pair}:continue
            assert sha(p)==r['image_sha256'] and sha(lookup[n]['file_path'])==r['source_sha256']
            for key in ['transform_matrix','fl_x','fl_y','cx','cy']:
                np.testing.assert_allclose(lookup[n][key],r['camera'][key],atol=1e-7,rtol=0)
            images[n]=np.array(Image.open(p));observations[n]=r
            deps[str(p)]=sha(p);deps[lookup[n]['file_path']]=r['source_sha256']
    for archive in [Path('/mnt/data/dec5_hand_silhouette_volume'),Path('/mnt/data/dec5_hand_silhouette_extra')]:
        p=archive/FRAME/'silhouettes.npz';deps[str(p)]=sha(p);a=np.load(p)
        for n in images:
            if n+'_mask' in a:masks[n]=np.rot90(a[n+'_mask']).astype(np.uint8)
    landmarks=Path('/mnt/data/dec5_hand_landmark_triangulation')/FRAME/'evidence.npz'
    a=np.load(landmarks);focus=a['points'][a['good']].mean(0);deps[str(landmarks)]=sha(landmarks)
    records=[]
    for pair in PAIRS:
        left,right=pair;cal,maps=rectify_local_pair(lookup[left],lookup[right],focus)
        dest=root/(left[:6]+'_'+right[:6]);dest.mkdir()
        rgb=[cv2.remap(images[n],*m,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for n,m in zip(pair,maps)]
        silhouettes=[cv2.remap(masks[n],*m,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT) for n,m in zip(pair,maps)]
        y0=0;y1=rgb[0].shape[0];width=rgb[0].shape[1]
        # Common crop retains horizontal disparity and changes only cy.
        k=cal['P1'][:,:3].copy();k[1,2]-=y0
        for label,im in zip(['left','right'],rgb):Image.fromarray(im[y0:y1]).save(dest/(label+'.png'))
        np.savez_compressed(dest/'calibration.npz',**{key:val for key,val in cal.items() if key!='size'},
                            cropped_intrinsic=k,crop=np.array([0,y0,width,y1]),
                            left_mask=silhouettes[0][y0:y1],right_mask=silhouettes[1][y0:y1],
                            left_map_x=maps[0][0][y0:y1],left_map_y=maps[0][1][y0:y1],
                            right_map_x=maps[1][0][y0:y1],right_map_y=maps[1][1][y0:y1])
        panel=Image.new('RGB',(2*width,y1-y0+25));draw=ImageDraw.Draw(panel)
        panel.paste(Image.fromarray(rgb[0][y0:y1]),(0,25))
        panel.paste(Image.fromarray(rgb[1][y0:y1]),(width,25))
        for y in range(25,panel.height,80):draw.line((0,y,2*width,y),fill='lime',width=1)
        draw.text((3,4),left+' / '+right+' local rectification (no resize)',fill='white')
        panel.save(dest/'rectification_review.png')
        record=dict(left=left,right=right,directory=str(dest),baseline=cal['baseline'],crop=[0,y0,width,y1],
                    disparity_offset=cal['disparity_offset'],rough_landmarks_for_framing_only=True,
                    hashes={n:sha(dest/n) for n in ['left.png','right.png','calibration.npz','rectification_review.png']})
        records.append(record);print('staged stereo',left,right,'baseline',cal['baseline'],'crop',record['crop'],flush=True)
    atomic_json(root/'request.json',dict(frame=FRAME,pairs=records,source_hashes=deps,script_sha256=sha(__file__),
        rectification_sha256=sha(Path(__file__).with_name('calibrated_stereo_rectification.py')),
        heldout_used=False,fixed_geometry_calibration=True,pose_optimized=False,source_color_adjusted=False,
        mask_use='post-hoc analysis and common crop only; RGB input is not masked',visual_status='pending'))


if __name__=='__main__':stage()
