"""Train-only mask additions from query-measured, multiview foreground evidence.

Only a 24-pixel exterior band is considered. Three other depth/color-compatible
foreground masks must corroborate a query depth. New seeds use the existing
native four-pixel dilation radius. No candidate mesh or eval RGB defines seeds.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,ROOT,exr,display
from study_confidence_depth_prior import load_real,unproject,project_integer
from forearm_rgb_witnesses import color_errors
from study_jaw_repair_transfer import PARENT

BASE=Path('/mnt/data/dec5_jaw_repair_transfer')
OUT=Path('/mnt/data/dec5_measured_foreground_override')


def add_certified_seeds(mask,seeds,radius=4,band_width=24):
    mask=np.asarray(mask,bool);seeds=np.asarray(seeds,bool)
    if mask.shape!=seeds.shape or radius<0 or band_width<radius:raise ValueError('Invalid mask expansion inputs')
    band=(~mask)&(distance_transform_edt(~mask)<=band_width)
    if (seeds&~band).any():raise ValueError('Seed outside allowed band')
    dilated=cv2.dilate(seeds.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
    return mask|(dilated&band)


def run(output,frame,name):
    folder=output/frame;folder.mkdir(parents=True,exist_ok=True)
    base=read(BASE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame)
    if receipt!=base['depth_receipt']:raise ValueError('Changed observed maps')
    source=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    maskroot=Path(source['source_masks']['root']);names=read(maskroot/'cameras.json');masks=np.load(maskroot/'masks.npz')['masks'].astype(bool)
    if sha(maskroot/'masks.npz')!=source['source_masks']['masks_sha256']:raise ValueError('Changed masks')
    index=next(i for i,r in enumerate(rows) if r['physical_camera']==name);camera=rows[index];mask=masks[names.index(name)]
    parameters=dict(exterior_band_px=24,seed_dilation_px=4,minimum_other_foreground_views=3,chroma_limit=.04,rgb_limit=.12)
    request=dict(frame=frame,camera=name,parameters=parameters,original_masks_sha256=sha(maskroot/'masks.npz'),
        mask_names_sha256=sha(maskroot/'cameras.json'),depth_receipt=receipt,
        profiles_sha256=sha(ROOT/'parameters.npz'),exposure_sha256=sha(ROOT/'exposure.json'),
        scripts={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,'forearm_rgb_witnesses.py',
            'diagnose_forearm_color_witnesses.py','study_confidence_depth_prior.py']},
        candidate_mesh_used=False,heldout_used=False,production_changed=False)
    if (folder/'request.json').exists() and read(folder/'request.json')!=request:raise ValueError('Frozen mask override mismatch')
    atomic_json(folder/'request.json',request)
    log_gain=np.load(ROOT/'parameters.npz')['log_gain'];gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,row=pair
        return row['physical_camera'],np.rint(display(exr(row['file_path'])*np.exp(log_gain[i]),gain)*255).clip(0,255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:images=dict(pool.map(load,enumerate(rows)))
    d=depths[index];band=(~mask)&(distance_transform_edt(~mask)<=24)
    y,x=np.nonzero(band&np.isfinite(d)&(d>0));points=unproject(camera,x,y,d[y,x]);qualified=np.zeros(len(x),np.uint8)
    # Bounded chunks avoid high peak memory on full-silhouette bands.
    for start in range(0,len(points),4096):
        q=points[start:start+4096];chroma,rgb=color_errors(q,camera,rows,depths,images)
        foreground=np.zeros(chroma.shape,bool)
        for i,row in enumerate(rows):
            uv,z=project_integer(row,q);xy=np.rint(uv).astype(int)
            good=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
            ids=np.flatnonzero(good)
            foreground[i,ids]=masks[names.index(row['physical_camera']),xy[ids,1],xy[ids,0]]
        qualified[start:start+len(q)]=((chroma<=.04)&(rgb<=.12)&foreground).sum(0)
    seeds=np.zeros(mask.shape,bool);seeds[y[qualified>=3],x[qualified>=3]]=True
    updated=add_certified_seeds(mask,seeds);np.save(folder/'mask.npy',updated)
    np.savez_compressed(folder/'evidence.npz',query_xy=np.column_stack([x,y]),query_points=points,
        other_qualified_foreground=qualified,seeds=seeds)
    image=images[name];overlay=image.copy();overlay[updated&~mask]=[255,0,255];overlay[seeds]=[0,255,255]
    # Whole image context plus fixed same-camera jaw crop; neither controls seeds.
    im=Image.fromarray(np.rot90(overlay));im.save(folder/'mask_overlay.png')
    panel=Image.new('RGB',(800,424));draw=ImageDraw.Draw(panel)
    for j,a in enumerate([np.rot90(image),np.rot90(overlay)]):
        panel.paste(Image.fromarray(a).crop((430,1020,830,1420)),(400*j,24))
    draw.text((2,3),'train RGB / cyan=measured seed, magenta=bounded expansion',fill='white');panel.save(folder/'jaw_native.png')
    atomic_json(folder/'result.json',dict(frame=frame,camera=name,request_sha256=sha(folder/'request.json'),
        original_masks_sha256=sha(maskroot/'masks.npz'),certified_seeds=int(seeds.sum()),added_mask_pixels=int((updated&~mask).sum()),
        hashes={n:sha(folder/n) for n in ['mask.npy','evidence.npz','mask_overlay.png','jaw_native.png']},
        source_rgb_sha256={r['file_path']:sha(r['file_path']) for r in rows},
        scene_segmentation_not_ground_truth=True,visual_status='requires_actual_review',production_accepted=False))
    print(frame,name,'measured candidates',len(points),'seeds',int(seeds.sum()),'added',int((updated&~mask).sum()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--frame',choices=['001083','001123','001193','001195'],required=True)
    p.add_argument('--camera',default='D004_D005_1210LZ');a=p.parse_args();run(a.output,a.frame,a.camera)
