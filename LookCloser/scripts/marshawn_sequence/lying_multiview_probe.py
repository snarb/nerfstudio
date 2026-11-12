"""Conservative multi-view silhouette diagnostic for low-height render artifacts."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw


def probe(root,index,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);base=root/'operations/flying_strict_low_20261112'
 row=json.loads((base/'path.json').read_text())['records'][index];frame=row['source_frame'];pose=np.array(row['pose']);fx,fy,cx,cy=row['intrinsics']
 raw=np.load(base/'neural'/f'{index:03}'/'strict.npz');rgb=np.array(Image.open(base/'neural'/f'{index:03}'/'strict.png'));depth=raw['depth'][...,0];h,w=depth.shape
 y,x=np.mgrid[:h,:w];rays=np.stack([(x+.5-cx)/fx,-(y+.5-cy)/fy,-np.ones_like(x)],axis=-1);rays/=np.linalg.norm(rays,axis=-1,keepdims=True);world=pose[:3,3]+(rays@pose[:3,:3].T)*depth[...,None]
 inv=json.loads((root/'operations/flying_strict_20261112/camera_inventory.json').read_text())['cameras'];near=sorted(inv,key=lambda c:np.linalg.norm(np.array(c['position'])-pose[:3,3]))[:12]
 votes={d:np.zeros((h,w),np.uint8) for d in [8,24,48]};validvotes=np.zeros((h,w),np.uint8)
 for c in near:
  name=c['name'];maskfile=root/'source_archive/raid/lookcloser/masks8k'/f'{frame:06}'/f'{name}_{frame:06}.png';maskfile=maskfile if maskfile.exists() else out/'masks'/f'{frame:06}'/f'{name}_{frame:06}.png';mask=cv2.imread(str(maskfile),cv2.IMREAD_GRAYSCALE)
  if mask is None:raise FileNotFoundError(maskfile)
  p=np.array(c['pose']);v=c['view'];loc=(world-p[:3,3])@p[:3,:3];z=-loc[...,2];u=v['fl_x']*loc[...,0]/np.maximum(z,1e-8)+v['cx']+c['crop'][0];vv=-v['fl_y']*loc[...,1]/np.maximum(z,1e-8)+v['cy']+c['crop'][1]
  good=(z>0)&(u>=0)&(u<mask.shape[1])&(vv>=0)&(vv<mask.shape[0]);xx=np.clip(u.astype(int),0,mask.shape[1]-1);yy=np.clip(vv.astype(int),0,mask.shape[0]-1);validvotes+=good
  # Preserve low-opacity real edges; alpha>16 is evidence of foreground.
  binary=np.uint8(mask>16)
  for d in votes:
   dil=cv2.dilate(binary,np.ones((2*d+1,2*d+1),np.uint8));votes[d]+=good&(dil[yy,xx]==0)
 panels=[];Image.fromarray(rgb).save(out/f'{index:03}_baseline.png')
 for label,im in [('baseline',rgb)]+[(f'margin{d}px reject3',np.where(((vote>=3)&(validvotes>=6)&(world[...,2]<.19))[...,None],0,rgb)) for d,vote in votes.items()]:
  # All-person view plus native right/lower actor patch.
  panel=Image.fromarray(im).resize((960,540));ImageDraw.Draw(panel).text((10,10),f'{index} {frame} {label}',fill='white');panels.append(panel)
 canvas=Image.new('RGB',(1920,1080))
 for j,panel in enumerate(panels):canvas.paste(panel,((j%2)*960,(j//2)*540))
 canvas.save(out/f'{index:03}_compare.jpg')
 np.savez_compressed(out/f'{index:03}_votes.npz',**{f'votes{d}':v for d,v in votes.items()},z=world[...,2],valid=validvotes)
 (out/f'{index:03}_receipt.json').write_text(json.dumps(dict(index=index,frame=frame,cameras=[c['name'] for c in near],margins=list(votes),threshold=3,world_z_max=.19),indent=2));print('DONE',index,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();cv2.setNumThreads(4);probe(a.root,a.index,a.output)
