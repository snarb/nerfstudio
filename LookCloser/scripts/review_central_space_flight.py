"""Native-detail preflight and chronological context for the central open flight."""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from central_space_temporal_flythrough import OUTPUT,PARENT

CANARY=['000899','000971','000973','001047','001197']
REGIONS={'face':(300,660,900,1240),'lipstick_hand':(280,970,730,1420),'ear_hair':(690,780,930,1160)}


def canary(output):
    ready=[f for f in CANARY if (output/'frames'/f/'complete.json').exists()]
    target=output/'canary_review';target.mkdir(exist_ok=True)
    for name,box in REGIONS.items():
        w,h=box[2]-box[0],box[3]-box[1]
        panel=Image.new('RGB',(w*3,(h+24)*2));draw=ImageDraw.Draw(panel)
        for i,frame in enumerate(ready):
            x,y=i%3*w,i//3*(h+24);panel.paste(Image.open(output/'frames'/frame/'frame.png').crop(box),(x,y+24))
            draw.text((x+4,y+4),frame,fill='white')
        panel.save(target/f'{name}.png')
    overview=Image.new('RGB',(540*len(ready),504));draw=ImageDraw.Draw(overview)
    for i,frame in enumerate(ready):
        for j,root in enumerate([PARENT,output]):overview.paste(Image.open(root/'frames'/frame/'frame.png').resize((270,480)),(i*540+j*270,24))
        draw.text((i*540+4,4),f'{frame}: previous view | new camera',fill='white')
    overview.save(target/'old_new_overview.png')
    atomic_json(target/'manifest.json',{'frames':ready,'expected_frames':CANARY,'all_canaries_ready':ready==CANARY,
                'render_hashes':{f:sha(output/'frames'/f/'frame.png') for f in ready},
                'evidence_hashes':{p.name:sha(p) for p in target.glob('*.png')},'status':'requires_actual_visual_review'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();canary(a.output)
