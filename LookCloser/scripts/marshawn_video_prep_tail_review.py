"""Tail source/hull/floor visual checks with explicitly limited depth proxies."""
from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image,ImageDraw
from render_temporal_aabb_review import voxel_scene,depths
from aabb_visualization import camera_in_box_frame,overlay_depth_aware
from audit_marshawn_temporal_box import rays
from prepare_marshawn_tuning import write,sha

def main(root):
 tail=root/'tail_native';meta=json.loads((tail/'geometry/common_box.json').read_text());bounds=np.asarray(meta['aabb']);old=np.asarray(json.loads((root/'frames/001660/data/box.json').read_text())['aabb']);T=np.asarray(meta['world_to_box']);out=tail/'review';out.mkdir(exist_ok=True);reports=[]
 identities=json.loads((tail/'reviewed_identities.json').read_text())['identities'];sheet=Image.new('RGB',(1440,2200))
 for j,frame in enumerate([1660,1900,1960,2020]):
  oldroot=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/temporal_2sec_floor_v1')
  geometry=tail/'geometry'/str(frame) if frame>=1960 else oldroot/'geometry_release'/str(frame)
  with np.load(geometry/'cells.npz') as z:scene=voxel_scene(z['lower'],z['upper'])
  for i,name in enumerate(['HC','DC','KA']):
   source_root=tail/'sources' if frame>=1960 else oldroot/'sources'
   r=json.loads((source_root/str(frame)/name/'receipt.json').read_text());c=camera_in_box_frame(r['camera_row'],T);im=Image.open(r['image']).convert('RGB');dep=depths(scene,c);ov,rec=overlay_depth_aware(im,bounds,c,dep,tolerance=.008,dash=16,gap=10);ov.save(out/f'{frame}_{name}_box.png')
   pose=np.asarray(c['transform_matrix']);yy,xx=np.mgrid[:c['h'],:c['w']];d=np.stack([(xx+.5-c['cx'])/c['fl_x'],-(yy+.5-c['cy'])/c['fl_y'],-np.ones_like(xx)],-1)@pose[:3,:3].T;d/=np.linalg.norm(d,axis=-1,keepdims=True);xyz=pose[:3,3]+dep[...,None]*d;inside=((xyz>=bounds[0])&(xyz<=bounds[1])).all(-1)&np.isfinite(dep);arr=np.array(im);arr[~inside]=0;black=Image.fromarray(arr);black.save(out/f'{frame}_{name}_inside_proxy.png')
   for k,panel in enumerate([ov,black]):
    tile=panel.copy();tile.thumbnail((480,253));sheet.paste(tile,(480*i,(j*2+k)*275+22));ImageDraw.Draw(sheet).text((480*i+4,(j*2+k)*275+4),f'{frame} {name} '+('BOX depthproxy' if k==0 else 'HULL/FLOOR proxy inside'),fill='white')
   if frame<1960:
    reports.append(dict(frame=frame,camera=name,overlay=rec));continue
   masks=np.load(tail/'geometry'/str(frame)/(name+'_selected_native.npz'));groups=identities[str(frame)][name]
   coverage={}
   for actor,ids in groups.items():
    y,x=np.where(masks[actor]);counts={'old':0,'proposed':0}
    for start in range(0,len(x),200000):
     for label,box in [('old',old),('proposed',bounds)]:counts[label]+=int((~rays(r['native_camera'],x[start:start+200000],y[start:start+200000],T,box)).sum())
    coverage[actor]=dict(pixels=len(x),misses=counts)
   reports.append(dict(frame=frame,camera=name,coverage=coverage,overlay=rec))
 sheet.save(out/'tail_common_box_contact.jpg',quality=94)
 write(out/'receipt.json',dict(proposal_sha256=sha(tail/'geometry/common_box.json'),records=reports,limitations='Ray intersection is necessary but not depth containment proof; blackvisualusespartialhull+floor proxy; greyfighter0threeviewSIFTanchors; crewdepthnotmodelled',no_training=True))
 print(json.dumps(reports),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
