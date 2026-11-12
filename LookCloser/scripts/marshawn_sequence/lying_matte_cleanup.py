"""Local lying-actor matte intersection; never adds raw-render background."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from PIL import Image,ImageDraw

def run(root):
 r=Path(root)/'operations';src=r/'flying_strict_low_20261112/neural';m=r/'body_matte_cleanup_20261112';out=r/'lying_cleanup_20261112/matte';out.mkdir(parents=True,exist_ok=True)
 keys={0:(1200,500,1960,1120),10:(1200,500,1960,1120),20:(1250,500,1960,1120),30:(1300,550,1960,1120),35:(1300,550,1960,1120),40:(1400,550,1960,1120),45:(1650,700,1960,1120),49:(1850,900,1960,1120),50:(1960,1120,2000,1160),69:(1960,1120,2000,1160)}
 yy,xx=np.mgrid[:1080,:1920];rows=[];contact=Image.new('RGB',(1800,7*260))
 for i in range(70):
  rgb=cv2.imread(str(src/f'{i:03}'/'strict.png'));old=np.load(src/f'{i:03}'/'strict.npz')['accumulation'][...,0];ma=cv2.imread(str(m/'sequence/alpha'/f'{i:03}.png'),cv2.IMREAD_GRAYSCALE)/255
  lo=max(k for k in keys if k<=i);hi=min(k for k in keys if k>=i);t=0 if lo==hi else (i-lo)/(hi-lo);box=(1-t)*np.array(keys[lo])+t*np.array(keys[hi]);x0,y0,x1,y1=box
  roi=np.clip(np.minimum.reduce([xx-x0,x1-xx,yy-y0,y1-yy])/24,0,1)
  factor=1-roi*(1-np.minimum(1,ma/np.maximum(old,1e-4)))
  clean=np.uint8(np.clip(rgb*factor[...,None]+.5,0,255));cv2.imwrite(str(out/f'{i:03}.png'),clean);cv2.imwrite(str(out/f'{i:03}_factor.png'),np.uint16(np.clip(factor*65535+.5,0,65535)))
  assert np.all(clean<=rgb);assert np.array_equal(clean[roi==0],rgb[roi==0])
  rows.append(dict(index=i,box=box.tolist(),changed_pixels=int(np.any(clean!=rgb,axis=-1).sum()),unchanged_outside_roi=True,no_background_added=True))
  if i in [0,10,20,30,35,40,45]:
   j=[0,10,20,30,35,40,45].index(i)
   for col,(im,label) in enumerate([(rgb,'strict'),(clean,'intersection')]):
    patch=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop((1150,500,1920,1080)).resize((770,240));ImageDraw.Draw(patch).text((10,10),f'{i} {label}',fill='white');contact.paste(patch,(col*900,j*260))
 contact.save(out/'lying_comparison.jpg');(out/'receipt.json').write_text(json.dumps(dict(method='strictRGB * min(1,Mat2alpha/strictacc) inside feathered lying-actor region',rows=rows),indent=2));print('DONE70',sum(x['changed_pixels'] for x in rows))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
