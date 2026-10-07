#!/usr/bin/env python3
"""Reviewable hair-fringe permissions; never an RGB or ray-sampling mask."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
import torch
from seed_colmap_from_nerfstudio import world_to_camera


# Polygons traced on the initial unmasked train RGB crops. These delimit a
# candidate hair domain, not a foreground extraction or an alpha target.
POLYGONS={
    11:[(669,110),(704,94),(729,105),(775,142),(790,177),(820,230),(846,263),
        (852,295),(715,295),(657,277),(656,234),(654,191),(663,167),(658,145)],
    97:[(645,50),(734,45),(797,41),(830,67),(869,108),(887,176),(880,254),
        (875,314),(465,314),(443,290),(440,231),(462,174),(490,122),(530,87),(598,65)],
    151:[(885,20),(1170,20),(1220,82),(1270,141),(1315,243),(1320,317),(1340,354),
         (1374,456),(1390,529),(1390,700),(1320,731),(1220,771),(990,771),(820,700),
         (780,657),(780,563),(807,458),(845,378),(862,314),(850,225),(850,140)]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def voxel_skin_veto(centers,cell_size,row,protected):
    """Conservatively exclude any voxel whose projected footprint touches skin."""
    if not len(centers):
        return np.zeros(0,dtype=bool)
    corners=np.array([(x,y,z) for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    world=centers[:,None,:]+corners[None,:,:]*cell_size
    w2c=world_to_camera(row)
    camera=world@w2c[:3,:3].T+w2c[:3,3]
    z=camera[...,2]
    in_front=(z>0).all(-1)
    # A camera-plane crossing has no bounded image footprint: deny it.
    blocked=(z>0).any(-1)&~in_front
    u=row['fl_x']*camera[...,0]/np.maximum(z,1e-8)+row['cx']-.5
    v=row['fl_y']*camera[...,1]/np.maximum(z,1e-8)+row['cy']-.5
    h,w=protected.shape
    x0=np.floor(u.min(-1).clip(0,w)).astype('int64')
    x1=(np.ceil(u.max(-1)).clip(-1,w-1)+1).astype('int64')
    y0=np.floor(v.min(-1).clip(0,h)).astype('int64')
    y1=(np.ceil(v.max(-1)).clip(-1,h-1)+1).astype('int64')
    integral=np.pad(protected.astype('int64'),((1,0),(1,0))).cumsum(0).cumsum(1)
    touched=integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0]
    blocked |= in_front&(x1>x0)&(y1>y0)&(touched>0)
    return blocked


def prepare(data,request_path,output):
    if (output/'annotation_manifest.json').exists():
        raise FileExistsError('Use a new annotation output')
    output.mkdir(parents=True,exist_ok=True)
    request=json.loads(request_path.read_text())
    meta=json.loads((data/'transforms.json').read_text())
    train=set(meta['train_filenames'])
    rows={r['camera_id']:r for r in meta['frames'] if r['file_path'] in train}
    receipts=[]
    contact=Image.new('RGB',(3*600,650),(30,30,30))
    for column,(cid,polygon) in enumerate(POLYGONS.items()):
        row=rows[cid];path=data/row['file_path']
        rgb=np.asarray(Image.open(path).convert('RGB'))
        domain=np.zeros(rgb.shape[:2],dtype=np.uint8)
        cv2.fillPoly(domain,[np.array(polygon,dtype=np.int32)],1)
        # The hand-drawn polygon only localizes the search. Exclude its own
        # artificial edge so a cropped domain cannot become a material boundary.
        domain=ndimage.binary_dilation(domain,iterations=15)
        dark=(rgb.max(-1)<90)&(domain>0)
        labels,count=ndimage.label(dark)
        sizes=np.bincount(labels.reshape(-1));sizes[0]=0
        component=labels==sizes.argmax()
        filled=ndimage.binary_fill_holes(component)
        # Three-pixel inside boundary band only; ornaments and dense interior
        # remain opaque. Color is used only to propose a permission domain.
        permissions=filled&~ndimage.binary_erosion(filled,iterations=3)&dark
        maximum=rgb.max(-1).astype('float32')
        saturation=(maximum-rgb.min(-1))/np.maximum(maximum,1)
        ornament=ndimage.binary_dilation((saturation>.25)&(maximum>40),iterations=4)
        permissions &= (saturation<.2)&~ornament&ndimage.binary_erosion(domain,iterations=5)
        protected=np.zeros_like(permissions)
        for label,box in request['roi_boxes'][Path(row['file_path']).name].items():
            if label not in ['face','hand']:
                continue
            x0,y0,x1,y1=box
            protected[max(0,y0-3):y1+3,max(0,x0-3):x1+3]=True
        permissions &= ~protected
        name=Path(row['file_path']).stem
        np.savez_compressed(output/(name+'.npz'),permissions=permissions,protected=protected)
        overlay=rgb.copy()
        overlay[permissions]=[0,255,0]
        # Cyan outlines identify the skin/hand veto without obscuring the RGB.
        outline=protected&~ndimage.binary_erosion(protected)
        overlay[outline]=[0,220,255]
        crop=Image.fromarray(overlay).crop((min(x for x,y in polygon)-20,max(0,min(y for x,y in polygon)-20),
                                          max(x for x,y in polygon)+20,max(y for x,y in polygon)+20))
        crop.thumbnail((590,600));contact.paste(crop,(column*600,30))
        ImageDraw.Draw(contact).text((column*600+5,8),f'train cam{cid:03d}: green permissions; cyan protected ROI',fill='white')
        receipts.append(dict(image=Path(row['file_path']).name,file=name+'.npz',camera_id=cid,
            source_sha256=sha(path),permissions_sha256=sha(output/(name+'.npz')),
            permitted_pixels=int(permissions.sum()),polygon=polygon))
    contact.save(output/'review.jpg')
    manifest=dict(train_only=True,reviewed=False,transforms_sha256=sha(data/'transforms.json'),
                  policy='three-pixel dark outer hair band; skin/hand veto; no RGB compositing or sampling exclusion',
                  maps=receipts)
    (output/'annotation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(review=str(output/'review.jpg'),permitted_pixels=[r['permitted_pixels'] for r in receipts])))


def build_grid(data,annotations,renders,output,resolution,reviewed):
    if not reviewed:
        raise ValueError('Inspect review.jpg before constructing the spatial grid')
    meta=json.loads((data/'transforms.json').read_text())
    annotation=json.loads((annotations/'annotation_manifest.json').read_text())
    if annotation['transforms_sha256']!=sha(data/'transforms.json'):
        raise ValueError('Semantic annotations differ from RGB/calibration')
    rows={Path(r['file_path']).name:r for r in meta['frames'] if r['file_path'] in meta['train_filenames']}
    metrics=json.loads((renders/'metrics.json').read_text())
    indices={r['image']:r['index'] for r in metrics['per_view'] if r['split']=='train'}
    aabb=np.array(meta['opacity_aabb'],dtype=np.float32)
    points=[];masks={}
    for receipt in annotation['maps']:
        name=receipt['image'];row=rows[name]
        if receipt['source_sha256']!=sha(data/row['file_path']):
            raise ValueError('Semantic RGB changed')
        if receipt['permissions_sha256']!=sha(annotations/receipt['file']):
            raise ValueError('Reviewed semantic permissions changed')
        with np.load(annotations/receipt['file'],allow_pickle=False) as p:
            permitted=p['permissions'];protected=p['protected']
        masks[name]=(permitted,protected)
        with np.load(renders/f"train_geometry_{indices[name]:04d}.npz",allow_pickle=False) as p:
            depth=p['depth'];alpha=p['accumulation']
        if depth.shape!=permitted.shape:
            raise ValueError('Semantic spatial seeds require native HD geometry')
        yy,xx=np.nonzero(permitted&(depth>0)&(alpha>.99))
        direction=np.stack([(xx+.5-row['cx'])/row['fl_x'],-(yy+.5-row['cy'])/row['fl_y'],-np.ones(len(xx))],-1)
        direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        pose=np.array(row['transform_matrix'])
        points.append(pose[:3,3]+(direction@pose[:3,:3].T)*depth[yy,xx,None])
    points=np.concatenate(points).astype(np.float32)
    votes=np.zeros(len(points),dtype=np.int16);veto=np.zeros(len(points),dtype=bool)
    for name,(permitted,protected) in masks.items():
        row=rows[name];w2c=world_to_camera(row)
        camera=points@w2c[:3,:3].T+w2c[:3,3]
        x=np.rint(row['fl_x']*camera[:,0]/np.maximum(camera[:,2],1e-8)+row['cx']-.5).astype('int64')
        y=np.rint(row['fl_y']*camera[:,1]/np.maximum(camera[:,2],1e-8)+row['cy']-.5).astype('int64')
        visible=(camera[:,2]>0)&(x>=0)&(y>=0)&(x<row['w'])&(y<row['h'])
        with np.load(renders/f"train_geometry_{indices[name]:04d}.npz",allow_pickle=False) as p:
            native_depth=p['depth']
        x=x.clip(0,row['w']-1);y=y.clip(0,row['h']-1)
        # A protected label behind a different visible surface is not a veto.
        ray_norm=np.sqrt(((x+.5-row['cx'])/row['fl_x'])**2+((y+.5-row['cy'])/row['fl_y'])**2+1)
        observed_z=native_depth[y,x]/ray_norm
        visible &= np.abs(observed_z-camera[:,2])<=np.maximum(.002,.01*camera[:,2])
        votes+=(visible&permitted[y,x]).astype(np.int16)
        veto |= visible&protected[y,x]
    accepted=points[(votes>=2)&~veto]
    uv=(accepted-aabb[0])/(aabb[1]-aabb[0])
    inside=((uv>=0)&(uv<1)).all(-1)
    ijk=np.floor(uv[inside]*resolution).astype('int64')
    grid=torch.zeros((resolution,)*3,dtype=torch.bool)
    if len(ijk):
        grid[ijk[:,0],ijk[:,1],ijk[:,2]]=True
    # Whole-voxel skin protection: project all eight corners, then veto any
    # protected pixel inside the conservative image footprint, not just its center.
    coordinates=torch.nonzero(grid).numpy()
    centers=aabb[0]+(coordinates+.5)/resolution*(aabb[1]-aabb[0])
    blocked=np.zeros(len(centers),dtype=bool)
    for name,(_,protected) in masks.items():
        blocked |= voxel_skin_veto(centers,(aabb[1]-aabb[0])/resolution,rows[name],protected)
    bad=coordinates[blocked]
    if len(bad):
        grid[bad[:,0],bad[:,1],bad[:,2]]=False
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(aabb=torch.from_numpy(aabb),permissions=grid,train_only=True,reviewed=True),output)
    receipt=dict(train_only=True,reviewed=True,grid_sha256=sha(output),annotation_manifest_sha256=sha(annotations/'annotation_manifest.json'),
                 permission_policy_version=2,builder_sha256=sha(Path(__file__)),
                 geometry_metrics_sha256=sha(renders/'metrics.json'),seed_points=len(points),supported_points=len(accepted),
                 permitted_voxels=int(grid.sum()),skin_vetoed_voxels=int(blocked.sum()),resolution=resolution)
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','grid'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--request',type=Path)
    p.add_argument('--renders',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--resolution',type=int,default=512)
    p.add_argument('--reviewed',action='store_true')
    args=p.parse_args()
    if args.action=='prepare':
        if args.request is None:p.error('--request is required')
        prepare(args.data,args.request,args.annotations)
    else:
        if args.renders is None or args.output is None:p.error('--renders and --output are required')
        build_grid(args.data,args.annotations,args.renders,args.output,args.resolution,args.reviewed)
