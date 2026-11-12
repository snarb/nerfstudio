"""Restore confident body interior missing from the Mat2 native photographic tail."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(root):
 root=Path(root);op=root/'operations';base=op/'flying_previous_bodyclean_20261112';source=op/'flying_strict_20261112/gt_sources';mat=op/'matanyone2_hair_20261112/native_soft_prompt/alpha';out=op/'flying_bodyclean_seamfix_20261112';out.mkdir(exist_ok=True);(out/'gt').mkdir(exist_ok=True)
 for name in ['neural','codec']:
  if not (out/name).exists():(out/name).symlink_to(base/name,target_is_directory=True)
 (out/'path.json').write_bytes((base/'path.json').read_bytes());path=json.loads((out/'path.json').read_text());rows=[];panels=[]
 for row in path['records'][70:]:
  frame=row['source_frame'];im=cv2.imread(str(source/f'{frame:06}_rgb.png'));sam=cv2.imread(str(source/f'{frame:06}_alpha.png'),0);a=cv2.imread(str(mat/f'{frame:06}.png'),0).astype(np.float32)/255
  body=cv2.morphologyEx(np.uint8(sam>=128),cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
  dist=cv2.distanceTransform(body,cv2.DIST_L2,5)
  original_dist=cv2.distanceTransform(np.uint8(sam>=128),cv2.DIST_L2,5)
  yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]
  # Preserve the real narrow opening below the foreground elbow.
  dist=np.where((xx<1200)&(yy>1220),original_dist,dist)
  y=np.arange(a.shape[0])[:,None];weight=np.clip((dist-8)/8,0,1)*np.clip((y-620)/40,0,1)
  fixed=np.maximum(a,weight);assert np.array_equal(fixed[weight==0],a[weight==0]);assert np.all(fixed>=a)
  x0,y0,w,h=row['native_crop'];x0-=2665;y0-=650
  def compose(alpha):
   p=im[y0:y0+h,x0:x0+w].astype(np.float32)*alpha[y0:y0+h,x0:x0+w,None]
   if (w,h)!=(1920,1080):p=cv2.resize(p,(1920,1080),interpolation=cv2.INTER_LANCZOS4)
   return np.uint8(np.clip(p+.5,0,255))
  clean=compose(fixed);cv2.imwrite(str(out/'gt'/f'{row["index"]:03}.png'),clean);cv2.imwrite(str(out/'gt'/f'{row["index"]:03}_alpha.png'),np.uint8(fixed*255+.5))
  # Comparison against source, in native coordinates, before moving crop.
  if frame in [1254,1260,1267]:
   items=[]
   for label,p in [('source',im),('previous',np.uint8(im*a[...,None])),('restored',np.uint8(im*fixed[...,None]))]:
    p=p[600:1400,800:1750].copy();p=cv2.resize(p,(570,480));cv2.putText(p,f'{frame} {label}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1);items.append(p)
   panels.append(np.concatenate(items,axis=1))
  rows.append(dict(index=row['index'],frame=frame,restored_alpha_pixels=int((fixed>a).sum()),unchanged_outside_body_guard=True,source_sha256=sha(source/f'{frame:06}_rgb.png'),matte_sha256=sha(mat/f'{frame:06}.png'),sam_sha256=sha(source/f'{frame:06}_alpha.png')))
 cv2.imwrite(str(out/'seam_comparison.jpg'),np.concatenate(panels,axis=0))
 (out/'repair_receipt.json').write_text(json.dumps(dict(cause='MatAnyone2 false background on dark overlapping garments',method='max(Mat2, confident SAM body interior), native SAM>=128 union,41px close (no general hole fill), inward8..16px feather, bodyy620..660',neural_unchanged=True,rows=rows),indent=2))
 sys.path.insert(0,str(root/'tools'));from flying_delivery import assemble;from flying_encode import encode
 assemble(out);encode(out);old=out/'Marshawn908_1267_flying_strict_BC_slowmo_24fps.mp4';target=out/'Marshawn_body_clean_sleeve_seam_fixed.mp4';old.replace(target)
 receipt=json.loads((out/'delivery_receipt.json').read_text());receipt.update(video=str(target),repair_receipt_sha256=sha(out/'repair_receipt.json'));(out/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2));print('FIXED',target,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
