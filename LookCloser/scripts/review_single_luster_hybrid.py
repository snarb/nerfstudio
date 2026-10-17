"""Evaluation-only native GT / LookCloser / hybrid crop export, without GPU use."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from luster_hybrid_common import geometry,read,sha

def review(root,name):
    protocol=read(root/'evaluation/protocol.json');out=root/'evaluation'/name;out.mkdir(exist_ok=True)
    for c in [7,12,95,150]:
        e=protocol['cameras'][str(c)];assert sha(e['source'])==e['source_sha256']
        gt=np.asarray(Image.open(e['source']).convert('RGB'));base=geometry(root,c)['rgb'];pred=np.load(root/f'candidates/{name}/cam{c:03d}.npy',mmap_mode='r')
        assert gt.shape==base.shape==pred.shape
        for region,box in e['boxes'].items():
            if box is None:continue
            x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1]
            tiles=[gt[sl],np.rint(base[sl].clip(0,1)*255).astype(np.uint8),np.rint(pred[sl].clip(0,1)*255).astype(np.uint8)]
            panel=Image.fromarray(np.concatenate(tiles,axis=1))
            if region!='person':panel.save(out/f'cam{c:03d}_{region}_GT_LookCloser_hybrid.png')
            panel.thumbnail((1500,900))
            labeled=Image.new('RGB',(panel.width,panel.height+24),(25,25,25));labeled.paste(panel,(0,24));draw=ImageDraw.Draw(labeled)
            for i,title in enumerate(['GT','LookCloser','hybrid']):draw.text((i*panel.width//3+5,5),title,fill='white')
            labeled.save(out/f'cam{c:03d}_{region}_preview.jpg',quality=95)
    print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();review(a.root,a.name)
