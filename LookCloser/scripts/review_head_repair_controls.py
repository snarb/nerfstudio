"""Matched-camera before/after head crops and geometry-miss attribution."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json

BASE=Path('/mnt/data/dec5_screen_travel_dynamic_150_v2')
NEW=Path('/mnt/data/dec5_head_repair_matched_v4')

def build(candidate=NEW):
    output=candidate/'comparison';output.mkdir(exist_ok=True);records=[]
    boxes={'001083':{'crown':[450,520,930,730],'jaw':[590,960,900,1170]},
           '001123':{'crown':[340,480,890,710],'jaw':[450,930,790,1140]}}
    for frame,regions in boxes.items():
        old=Image.open(BASE/'frames'/frame/'frame.png');new=Image.open(candidate/'frames'/frame/'frame.png')
        old_depth=np.rot90(np.load(BASE/'frames'/frame/'target_depth.npz')['depth']);new_depth=np.rot90(np.load(candidate/'frames'/frame/'target_depth.npz')['depth'])
        assert read(BASE/'frames'/frame/'result.json')['camera']==read(candidate/'frames'/frame/'result.json')['camera']
        for region,box in regions.items():
            x0,y0,x1,y1=box;w=x1-x0;h=y1-y0;panel=Image.new('RGB',(2*w,h+28));draw=ImageDraw.Draw(panel)
            panel.paste(old.crop(box),(0,28));panel.paste(new.crop(box),(w,28));draw.text((4,4),'BEFORE / same camera and time',fill='white');draw.text((w+4,4),'AFTER / local 3D completion',fill='white')
            path=output/f'{frame}_{region}.png';panel.save(path)
            old_black=np.asarray(old)[y0:y1,x0:x1].max(2)==0;new_black=np.asarray(new)[y0:y1,x0:x1].max(2)==0
            records.append(dict(frame=frame,region=region,box=box,path=str(path),sha256=sha(path),
                old_black_pixels_in_crop=int(old_black.sum()),old_black_with_depth=int((old_black&(old_depth[y0:y1,x0:x1]>0)).sum()),
                previously_black_now_colored=int((old_black&~new_black).sum()),new_depth_pixels=int(((old_depth[y0:y1,x0:x1]==0)&(new_depth[y0:y1,x0:x1]>0)).sum())))
    atomic_json(output/'evidence.json',dict(records=records,not_quality_metrics=True,compares_predictions_not_gt=True,visual_status='pending'))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,default=NEW);args=parser.parse_args();build(args.candidate)
