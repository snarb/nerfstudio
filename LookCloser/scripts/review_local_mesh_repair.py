"""Independent real-camera review for local synthetic-prior mesh experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import cameras,read,sha,atomic_json,display,exr,ROOT,SOURCE,CALIBRATION,HELD_CAMERAS,project
from render_patchmatch_camera_path import normalize_frame
from diffusion_mesh_repair import BASE,OUTPUT,scene_for,render_atlas


def review(output,heldout=False,cylinder_asset='cylinder_asset',review_suffix=''):
    rows,_,meta=cameras('000973');by_name={r['physical_camera']:r for r in rows}
    if heldout:
        cal=read(CALIBRATION);raw=read(SOURCE/'000973'/'transforms.json')
        for row in cal['frames']:
            if row['physical_camera'] in HELD_CAMERAS:
                q=normalize_frame(row,cal,read(meta));source=next(r for r in raw['frames'] if r['physical_camera']==row['physical_camera'])
                q['file_path']=str(SOURCE/'000973'/source['file_path']);by_name[row['physical_camera']]=q
        names=sorted(HELD_CAMERAS)
    else:names=['I004_D005_1210Q7','K004_D005_121016','J004_C005_1210I4','H004_C005_1210SZ']
    assets={'before':BASE,'local':output/'local_asset'/'frames'/'000973','cylinder':output/cylinder_asset/'frames'/'000973'}
    loaded={}
    for label,path in assets.items():
        atlas=dict(np.load(path/'atlas_geometry.npz'));texture=np.asarray(Image.open(path/'texture_joint.png').convert('RGB'))
        loaded[label]=(scene_for(atlas['vertices'],atlas['triangles']),atlas,texture)
    tube=np.array(read(output/'request.json')['tube_seed']);gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    profile=read(ROOT/'camera_profiles.json');gains=dict(zip(profile['physical_cameras'],profile['rgb_gain']))
    root=output/(('heldout_review' if heldout else 'train_review')+review_suffix);records=[]
    chin_loop=next(r['vertices'] for r in read(output/'geometry_diagnosis.json')['chin_boundary_loops'] if r['loop']==103)
    chin_points=loaded['before'][1]['vertices'][chin_loop]
    ca=loaded['cylinder'][1];tube_points=ca['vertices'][np.unique(ca['triangles'][-384:])]
    for name in names:
        row=by_name[name];target=root/name;target.mkdir(parents=True,exist_ok=True)
        rgb=exr(row['file_path'])
        if not heldout:rgb=rgb*np.array(gains[name])
        gt=np.rint(display(rgb,gain)*255).clip(0,255).astype(np.uint8)
        images={'real RGB':gt};Image.fromarray(gt).save(target/'gt.png')
        for label,(scene,atlas,texture) in loaded.items():
            rgb,depth,_,_=render_atlas(scene,atlas,texture,row);images[label]=rgb;Image.fromarray(rgb).save(target/f'{label}.png')
        uv,_=project(tube[None],[row]);x,y=np.rint(uv[0,0]).astype(int)
        x0=int(np.clip(x-190,0,1920-512));y0=int(np.clip(y-180,0,1080-512));box=(x0,y0,x0+512,y0+512)
        overview=Image.new('RGB',(432*4,794));draw=ImageDraw.Draw(overview)
        crops={}
        for j,(label,rgb) in enumerate(images.items()):
            image=Image.fromarray(rgb)
            overview.paste(image.transpose(Image.Transpose.ROTATE_90).resize((432,768)),(j*432,26));draw.text((j*432+3,5),label,fill='white')
            crops[label]=image.crop(box).transpose(Image.Transpose.ROTATE_90).resize((1024,1024))
        overview.save(target/'overview.png')
        for label,roi in {'tube':(240,550,490,920),'chin':(480,820,1024,1015)}.items():
            ww,hh=roi[2]-roi[0],roi[3]-roi[1];panel=Image.new('RGB',(ww*4,hh+26));draw=ImageDraw.Draw(panel)
            for j,(variant,im) in enumerate(crops.items()):
                panel.paste(im.crop(roi),(j*ww,26));draw.text((j*ww+3,5),variant,fill='white')
            panel.save(target/f'{label}.png')
        # Oblique cameras move the chin relative to the tube. Track the reviewed
        # mesh boundary instead of allowing a fixed tube-relative crop to miss it.
        tracked_boxes={}
        for region,points in [('chin',chin_points),('tube',tube_points)]:
            tracked_uv,_=project(points,[row]);xy=tracked_uv[0]
            low=np.floor(xy.min(0)-30).astype(int);high=np.ceil(xy.max(0)+30).astype(int)
            low=np.maximum(low,[0,0]);high=np.minimum(high,[1920,1080]);tracked_box=[*low,*high]
            cc=[Image.fromarray(rgb).crop(tracked_box).transpose(Image.Transpose.ROTATE_90) for rgb in images.values()]
            if region=='tube':cc=[im.resize((im.width*2,im.height*2)) for im in cc]
            ww,hh=cc[0].size;panel=Image.new('RGB',(ww*4,hh+26));draw=ImageDraw.Draw(panel)
            for j,(label,im) in enumerate(zip(images,cc)):
                panel.paste(im,(j*ww,26));draw.text((j*ww+3,5),label,fill='white')
            panel.save(target/f'{region}_tracked.png');tracked_boxes[region]=list(map(int,tracked_box))
        records.append({'camera':name,'gt_sha256':sha(target/'gt.png'),'render_sha256':{n:sha(target/f'{n}.png') for n in loaded},
                        'source_sha256':sha(row['file_path']),'native_crop_xyxy':box,'tracked_native_crop_xyxy':tracked_boxes})
        print(f'review camera={name} heldout={heldout}',flush=True)
    atomic_json(root/'manifest.json',{'views':records,'heldout':heldout,'used_for_prediction':False,
                'train_reference_display':'fixed exposure plus frozen train-camera profile' if not heldout else 'fixed exposure only'})
    if heldout:
        import torch
        from score_colmap_patchmatch_tsdf_face import load_display_rgb,load_manual_face_mask,masked_display_metrics,LearnedPerceptualImagePatchSimilarity
        target=root/'F004_B005_1210O9';roi=ROOT/'config'/'face_roi_000973.json';gt=load_display_rgb(target/'gt.png')
        mask,_=load_manual_face_mask(roi,target/'gt.png',gt.shape[:2]);net=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
        gt=torch.tensor(gt.transpose(2,0,1),device='cuda');mask=torch.tensor(mask,device='cuda');metrics={}
        with torch.inference_mode():
            for label in loaded:
                pred=torch.tensor(load_display_rgb(target/f'{label}.png').transpose(2,0,1),device='cuda')
                metrics[label]=masked_display_metrics(pred,gt,mask,net)
        atomic_json(root/'face_metrics.json',{'frame':'000973','face_roi_sha256':sha(roi),'gt_sha256':sha(target/'gt.png'),
                    'variants':metrics,'no_full_frame_metrics':True,'same_protocol_as_joint_texture':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--heldout',action='store_true')
    p.add_argument('--cylinder-asset',default='cylinder_asset');p.add_argument('--review-suffix',default='')
    a=p.parse_args();review(a.output,a.heldout,a.cylinder_asset,a.review_suffix)


if __name__=='__main__':main()
