#!/usr/bin/env python3
"""One pixel-preserving collage of the cord region in all original cameras."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make(root, output, columns=15):
    if output.exists():
        raise FileExistsError('Preserve previous collages')
    meta_path = root/'data/transforms.json'
    audit = json.loads((root/'data/audit.json').read_text())
    if digest(meta_path) != audit['transforms_sha256']:
        raise ValueError('Calibration changed')
    frames = sorted(json.loads(meta_path.read_text())['frames'],key=lambda f:f['camera_id'])
    receipts = {r['camera_id']:r for r in audit['receipts']}
    if len(frames)!=165 or len(receipts)!=165:
        raise ValueError('Require all 165 original cameras')
    ref = next(f for f in frames if f['camera_id']==97)
    pose = np.array(ref['transform_matrix'])
    # Approximate spatial envelope used only to locate image crops. This is
    # not certified cord geometry, a depth prior, or an occlusion classifier.
    # It covers the braid, two ornaments and a margin below the second one.
    points = np.array([
        pose[:3,:3]@np.array([(x+.5-ref['cx'])/ref['fl_x'],
                            -(y+.5-ref['cy'])/ref['fl_y'],-1.])*depth+pose[:3,3]
        for x in [800,900] for y in [1120,1510] for depth in [1.06,1.09]])
    rows, crops = [], []
    for f in frames:
        receipt = receipts[f['camera_id']]
        source = Path(receipt['source'])
        if digest(source)!=receipt['source_sha256']:
            raise ValueError(f'Original photo changed: {source}')
        W,H = receipt['original_size']
        p = np.array(f['transform_matrix'])
        local = (points-p[:3,3])@p[:3,:3]
        z = -local[:,2]
        if (z<=0).any():
            raise ValueError('Crop envelope behind a camera')
        u = (f['fl_x']*local[:,0]/z+f['cx'])*W/f['w']
        v = (-f['fl_y']*local[:,1]/z+f['cy'])*H/f['h']
        requested = [math.floor(u.min())-12,math.floor(v.min())-12,
                     math.ceil(u.max())+12,math.ceil(v.max())+12]
        box = [max(0,requested[0]),max(0,requested[1]),
               min(W,requested[2]),min(H,requested[3])]
        visible = box[2]>box[0] and box[3]>box[1]
        crop = None
        if visible:
            with Image.open(source) as im:
                if im.size!=(W,H):
                    raise ValueError('Original dimensions differ')
                crop = im.convert('RGB').crop(box)
        rows.append(dict(camera=f['camera_id'],source=str(source),
            source_sha256=receipt['source_sha256'],original_size=[W,H],
            requested_box=requested,crop_box=box if visible else None,
            status=('edge' if requested!=box else 'inside') if visible else 'out_of_frame',
            crop_size=list(crop.size) if crop else None))
        crops.append(crop)
    output.mkdir(parents=True)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,23)
    title_font=ImageFont.truetype(font_path,30)
    header,pad,title=48,12,100
    count_rows=math.ceil(len(rows)/columns)
    col_width=[max([240]+[crops[i].width+pad*2 for i in range(c,len(rows),columns) if crops[i]])
               for c in range(columns)]
    row_height=[max([120]+[im.height for im in crops[r*columns:(r+1)*columns] if im])+header+pad*2
                for r in range(count_rows)]
    width=sum(col_width);height=title+sum(row_height)
    sheet=Image.new('RGB',(width,height),'#202124');draw=ImageDraw.Draw(sheet)
    draw.text((14,8),'FRAME 000470 | BELT CORD | ALL 165 CAMERAS | ORIGINAL JPEG CROPS, 1:1 PIXELS',font=title_font,fill='white')
    draw.text((14,48),'EDGE = photo boundary clips region. OUT OF FRAME = no region in photo. Back views may hide the cord.',font=font,fill='#bfc5cc')
    offsets_x=np.r_[0,np.cumsum(col_width)];offsets_y=title+np.r_[0,np.cumsum(row_height)]
    for i,(row,crop) in enumerate(zip(rows,crops)):
        col=i%columns;r=i//columns;x,y=int(offsets_x[col]),int(offsets_y[r])
        cw,ch=col_width[col],row_height[r]
        draw.rectangle((x,y,x+cw-1,y+ch-1),outline='#596069',width=2)
        label=f'cam {row["camera"]:03d}'
        if crop:
            label+=f' | {crop.width}x{crop.height}'
            if row['status']=='edge':label+=' | EDGE'
            px=x+(cw-crop.width)//2;py=y+header+(ch-header-crop.height)//2
            sheet.paste(crop,(px,py));row['collage_crop_box']=[px,py,px+crop.width,py+crop.height]
            # Verify every pasted crop keeps the original source pixels.
            if sheet.crop(tuple(row['collage_crop_box'])).tobytes()!=crop.tobytes():
                raise ValueError('Collage altered source pixels')
        else:
            draw.text((x+pad,y+header+20),'OUT OF FRAME',font=font,fill='#aeb5be')
        draw.text((x+pad,y+10),label,font=font,fill='white')
    path=output/'cord_all_165_fullres.png'
    sheet.save(path,compress_level=3)
    preview=sheet.copy();preview.thumbnail((2400,2400));preview.save(output/'cord_all_165_preview.jpg',quality=92)
    # Review pages are secondary previews only; the main PNG is never scaled.
    for start in range(0,len(rows),45):
        page=Image.new('RGB',(9*180,5*280),'#202124');pd=ImageDraw.Draw(page)
        pf=ImageFont.truetype(font_path,14)
        for j in range(45):
            i=start+j
            if i>=len(rows):break
            px=(j%9)*180;py=(j//9)*280
            pd.text((px+3,py+3),f'cam {rows[i]["camera"]:03d} {rows[i]["status"]}',font=pf,fill='white')
            crop=crops[i]
            if crop:
                thumb=crop.copy();thumb.thumbnail((174,248));page.paste(thumb,(px+(180-thumb.width)//2,py+25))
        page.save(output/f'review_{start//45+1}.jpg',quality=92)
    manifest=dict(frame='000470',cameras=165,native_crops=sum(im is not None for im in crops),
        out_of_frame=sum(im is None for im in crops),image_size=[width,height],
        png_sha256=digest(path),probe_sha256=digest(Path(__file__)),
        calibration_sha256=digest(meta_path),masks_used=False,rendered_images_used=False,
        source_pixels_preserved=True,resizing_in_main_collage=False,
        envelope_scope='Approximate camera97 crop volume, for cropping only; occlusion not inferred',
        reference_hd_box=[800,1120,900,1510],reference_camera_z_interval=[1.06,1.09],
        world_envelope=points.tolist(),photos=rows)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['cameras','native_crops','out_of_frame','image_size']})+
          f' png_bytes={path.stat().st_size}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/pavlo/renders/luster_000470_opacity'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();make(a.root,a.output)
