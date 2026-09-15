"""Post-hoc GT-referenced localization of the remaining jaw seam; no metrics."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from calibrated_depth_witness import load_images
from joint_temporal_texture import atomic_json, read, sha
from render_midsequence_jaw_completion import ROOT

VIEW='K004_B005_1210DS'
BOX=(300,960,820,1250)


def preview(frame):
    images, _, receipt=load_images(frame)
    gt=np.rot90(images[VIEW]); folder=ROOT/frame/'rgb'/VIEW/'completed'/'frames'/frame
    pred=np.asarray(Image.open(folder/'frame.png'))
    out=ROOT/frame/'seam_diagnosis'; out.mkdir(exist_ok=False)
    Image.fromarray(gt).save(out/'train_gt.png')
    panes=[]
    for name,array in [('real train GT',gt),('completed prediction',pred)]:
        im=Image.fromarray(array).crop(BOX); draw=ImageDraw.Draw(im)
        for x in range(350,820,50):
            draw.line([(x-BOX[0],0),(x-BOX[0],BOX[3]-BOX[1])],fill=(255,30,30),width=1)
            draw.text((x-BOX[0]+2,2),str(x),fill='white')
        for y in range(1000,1250,50):
            draw.line([(0,y-BOX[1]),(BOX[2]-BOX[0],y-BOX[1])],fill=(255,30,30),width=1)
            draw.text((2,y-BOX[1]+2),str(y),fill='white')
        pane=Image.new('RGB',(im.width,im.height+24)); pane.paste(im,(0,24))
        ImageDraw.Draw(pane).text((3,3),name,fill='white'); panes.append(pane)
    joined=Image.new('RGB',(sum(p.width for p in panes),panes[0].height)); joined.paste(panes[0],(0,0)); joined.paste(panes[1],(panes[0].width,0))
    joined.save(out/'coordinate_preview.png')
    atomic_json(out/'preview.json',dict(frame=frame,view=VIEW,box=BOX,source_receipt=receipt,
        prediction_sha256=sha(folder/'frame.png'),script_sha256=sha(__file__),
        GT_used_only_for_posthoc_diagnosis=True,quality_metrics=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=['001193','001195'])
    preview(p.parse_args().frame)
