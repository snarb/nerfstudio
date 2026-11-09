"""Source-photo box diagnostic using explicitly labelled learned surface depth.

Never mark uncertain performer depth as outside. Raw field partitions remain
separate; this tool does not modify a rendered image or certify physical truth.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def main(a):
 r=json.loads((a.partition/'receipt.json').read_text());c=r['camera_runtime'];z=np.load(a.partition/'raw_partition.npz');h,w=c['h'],c['w'];dep=z['surface_depth'];alpha=z['full'][...,3].reshape(h,w);source=Image.open(a.source).convert('RGB');assert source.size==(w,h)
 yy,xx=np.mgrid[:h,:w];pose=np.array(c['transform_matrix']);d=np.stack([(xx+.5-c['cx'])/c['fl_x'],-(yy+.5-c['cy'])/c['fl_y'],-np.ones_like(xx)],-1)@pose[:3,:3].T;d/=np.linalg.norm(d,axis=-1,keepdims=True)
 valid=np.isfinite(dep)&(alpha>=a.minimum_opacity);xyz=pose[:3,3]+np.where(valid,dep,0)[...,None]*d;T=np.array(r['world_to_box']);q=xyz@T[:3,:3].T+T[:3,3];b=np.array(r['aabb']);inside=((q>=b[0])&(q<=b[1])).all(-1);fg=np.asarray(Image.open(a.mask))>=128;rgb=np.array(source);shown=np.zeros_like(rgb);shown[valid&inside]=rgb[valid&inside];unknown=fg&~valid;outside=fg&valid&~inside;shown[unknown]=(rgb[unknown]*.35+np.array([255,160,0])*.65).astype('uint8')
 a.output.mkdir(parents=True,exist_ok=True);Image.fromarray(shown).save(a.output/'source_depth_cut.png');marked=rgb.copy();marked[outside]=[255,0,70];marked[unknown]=(rgb[unknown]*.35+np.array([255,160,0])*.65).astype('uint8');Image.fromarray(marked).save(a.output/'source_flags.png');teacher=Image.open(a.partition/(a.camera+'_inside.png')).convert('RGB');contact=Image.new('RGB',(w*3,h+50),'#181818');dr=ImageDraw.Draw(contact)
 for i,(title,im)in enumerate([('SOURCE',source),('SOURCE + MODEL DEPTH: BLACK OUTSIDE / ORANGE UNKNOWN',Image.fromarray(shown)),('ACTUAL FIELD: INSIDE BOX ONLY',teacher)]):contact.paste(im,(i*w,50));dr.text((i*w+8,12),title,fill='white')
 contact.save(a.output/'comparison.png');preview=contact.copy();preview.thumbnail((1920,800));preview.save(a.output/'preview.jpg',quality=94)
 report=dict(foreground_pixels=int(fg.sum()),known_inside=int((fg&valid&inside).sum()),known_outside=int(outside.sum()),unknown=int(unknown.sum()),minimum_opacity=a.minimum_opacity,depth='learned first absolute opacity0.5 crossing; conditional on teacher quality',source=str(a.source),partition=str(a.partition),physical_containment_certified=False);(a.output/'diagnostics.json').write_text(json.dumps(report,indent=2));print(report)
 # Retain full calibrated positions only for supported actor pixels, not background.
 np.savez_compressed(a.output/'actor_depth_points.npz',model_xyz=xyz[fg&valid],pixel_xy=np.stack([xx,yy],-1)[fg&valid],inside=inside[fg&valid])

if __name__=='__main__':
 p=argparse.ArgumentParser();
 for k in ['partition','source','mask','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--camera',required=True);p.add_argument('--minimum-opacity',type=float,default=.9);main(p.parse_args())
