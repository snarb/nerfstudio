#!/usr/bin/env python3
"""Train-only translated-patch NCC audit of already warped source images."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--warps',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    images=[np.asarray(Image.open(f).convert('RGB')).astype(np.float32)/255 for f in sorted(a.warps.glob('source_*.png'))]
    valid=[np.asarray(Image.open(f))>0 for f in sorted(a.warps.glob('valid_*.png'))]
    gray=[im@np.array([.2126,.7152,.0722],np.float32) for im in images]
    rows=[]
    # Rectangles only localize an audit; they never enter prediction/scoring.
    for region,(x0,y0,x1,y1) in {'hand_neck':(400,390,800,780),'face':(650,400,1150,950)}.items():
        for source in range(1,len(images)):
            for y in range(y0+32,y1-32,24):
                for x in range(x0+32,x1-32,24):
                    ref=gray[0][y-24:y+24,x-24:x+24]
                    if not valid[0][y-24:y+24,x-24:x+24].all() or ref.std()<.008:continue
                    if not valid[source][y-32:y+32,x-32:x+32].all():continue
                    search=gray[source][y-32:y+32,x-32:x+32]
                    scores=cv2.matchTemplate(search,ref,cv2.TM_CCOEFF_NORMED)
                    v,u=np.unravel_index(scores.argmax(),scores.shape)
                    row={'region':region,'source_rank':source,'x':x,'y':y,'dx':int(u)-8,'dy':int(v)-8,
                         'ncc_at_zero':float(scores[8,8]),'ncc_best':float(scores[v,u]),'primary_std':float(ref.std())}
                    rows.append(row)
    summary=[]
    for region in ['hand_neck','face']:
        for source in range(1,len(images)):
            selected=[r for r in rows if r['region']==region and r['source_rank']==source]
            good=[r for r in selected if r['ncc_best']>.7 and abs(r['dx'])<8 and abs(r['dy'])<8]
            if not selected:continue
            summary.append({'region':region,'source_rank':source,'patches':len(selected),'reliable_matches':len(good),
                            'median_ncc_zero':float(np.median([r['ncc_at_zero'] for r in selected])),
                            'median_ncc_best':float(np.median([r['ncc_best'] for r in selected])),
                            'median_reliable_shift_xy':np.median([[r['dx'],r['dy']] for r in good],axis=0).tolist() if good else None,
                            'median_reliable_shift_length':float(np.median([np.hypot(r['dx'],r['dy']) for r in good])) if good else None})
    # Largest reliable shifts, not selected using held-out GT.
    selected=sorted([r for r in rows if r['region']=='hand_neck' and r['ncc_best']>.7 and abs(r['dx'])<8 and abs(r['dy'])<8],
                    key=lambda r:r['ncc_best']-r['ncc_at_zero'],reverse=True)[:8]
    for i,row in enumerate(selected):
        x,y,s=row['x'],row['y'],row['source_rank'];dx,dy=row['dx'],row['dy']
        crops=[images[0][y-32:y+32,x-32:x+32],images[s][y-32:y+32,x-32:x+32],images[s][y+dy-32:y+dy+32,x+dx-32:x+dx+32]]
        canvas=Image.new('RGB',(384,160))
        for j,(crop,label) in enumerate(zip(crops,['primary','projected',f'shift {dx},{dy}'])):
            im=Image.fromarray(np.rint(crop*255).astype(np.uint8)).rotate(90).resize((128,128),Image.Resampling.NEAREST)
            canvas.paste(im,(128*j,32));ImageDraw.Draw(canvas).text((128*j+2,5),label,fill='white')
        canvas.save(a.output/f'patch_{i:02d}.png')
    atomic_json(a.output/'audit.json',{'uses_eval_rgb':False,'applies_image_warp':False,'summary':summary,'patches':rows,
                 'visual_patch_rows':selected,'input_hashes':{f.name:sha256(f) for f in sorted(a.warps.glob('*.png'))}})
    print(json.dumps(summary))


if __name__=='__main__':main()
