import numpy as np,json
from pathlib import Path
from PIL import Image,ImageDraw
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106'));parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
B=args.campaign/'native_semantic_grid_20000';C=B/'corrected_v2';T=B.parent/'data';O=args.output;frames={f['physical_camera']:f for f in json.loads((T/'transforms.json').read_text())['frames']}
p=np.array(json.loads((C/'train_shoe_feature_depth.json').read_text())['points'][0]['xyz_floor']);z=np.load(C/'semantic_grid.npz');box=z['aabb'];pitch=(box[1]-box[0])/128;i=np.floor((p-box[0])/pitch).astype(int);lo=box[0]+i*pitch;corners=lo+np.array([[x,y,z]for x in [0,1]for y in [0,1]for z in [0,1]])*pitch
allrows=[];panels=[]
for folder in ['corrected_v2','margin_W32','margin_W16']:
 receipt=json.loads((B/folder/'grid_receipt.json').read_text())
 for row in receipt['sources']:
  f=frames[row['camera']];pose=np.array(f['transform_matrix']);q=(corners-pose[:3,3])@pose[:3,:3];d=-q[:,2];uv=q[:,:2]/d[:,None]*[f['fl_x'],-f['fl_y']]+[f['cx'],f['cy']];l=np.floor(uv.min(0)-1).astype(int);u=(np.ceil(uv.max(0))+1).astype(int)
  with np.load(row['archive'])as a:
   if folder=='corrected_v2':support=a['W64']>0;unknown=a['source_ineligible']
   else:support=a['support'];unknown=a['unknown']
  with np.load(row['semantic'])as a:valid=a['valid']
  full=(l>=0).all()and(u<=[f['w'],f['h']]).all()and(d>1e-5).all();sl=(slice(l[1],u[1]),slice(l[0],u[0]));clear=bool(full and not support[sl].any()and not unknown[sl].any()and valid[sl].all())
  if clear:
   v=(p-pose[:3,3])@pose[:3,:3];xy=v[:2]/-v[2]*[f['fl_x'],-f['fl_y']]+[f['cx'],f['cy']];allrows.append(dict(width=folder,camera=row['camera'],point_xy=xy.tolist(),rectangle=[*l.tolist(),*u.tolist()]))
   if folder=='margin_W16':
    src=Image.open(T/f['file_path']).convert('RGB');x,y=np.rint(xy).astype(int);crop=(x-60,y-60,x+60,y+60);im=src.crop(crop);dr=ImageDraw.Draw(im);dr.rectangle([l[0]-crop[0],l[1]-crop[1],u[0]-crop[0],u[1]-crop[1]],outline='red',width=1);dr.ellipse((58,58,62,62),outline='yellow');im=im.resize((240,240));panel=Image.new('RGB',(240,264));panel.paste(im,(0,24));ImageDraw.Draw(panel).text((2,3),row['camera']+' clears known toe W16',fill='white');panels.append(panel)
print(json.dumps(allrows));(O/'witness_votes.json').write_text(json.dumps(allrows,indent=2)+'\n')
if panels:
 cols=4;sheet=Image.new('RGB',(cols*240,264*((len(panels)+cols-1)//cols)))
 for n,im in enumerate(panels):sheet.paste(im,((n%cols)*240,(n//cols)*264))
 sheet.save(O/'witness_clear_votes.png')
