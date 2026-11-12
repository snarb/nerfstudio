"""MatAnyone reductions only at external silhouette boundaries, never interiors."""
import argparse,ast,json,sys,hashlib
from pathlib import Path
import cv2,numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw

def exterior_gate(alpha,band=3):
 mask=alpha>.1
 solid=ndi.binary_fill_holes(mask)
 exterior=~solid
 distance=ndi.distance_transform_edt(~exterior)
 gate=np.clip((band-distance)/max(band-1,1),0,1)
 thin=mask & ~cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))).astype(bool)
 protect=cv2.dilate(thin.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
 gate[ndi.distance_transform_edt(exterior)>band]=0
 gate[protect]=0
 return gate,solid,distance,protect

def run(root):
 root=Path(root);op=root/'operations';base=op/'flying_strict_low_20261112';out=op/'flying_exterior_only_20261112';out.mkdir(exist_ok=True)
 path=json.loads((base/'path.json').read_text());(out/'path.json').write_bytes((base/'path.json').read_bytes())
 if not (out/'codec').exists():(out/'codec').symlink_to(base/'codec',target_is_directory=True)
 # Reviewed actor and prop boxes from the independent body experiment.
 tree=ast.parse((root/'tools/body_matte_upper.py').read_text());K=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='K' for t in n.targets))
 yy,xx=np.mgrid[:1080,:1920];receipts=[];contact=[]
 for i in range(70):
  folder=base/'neural'/f'{i:03}';rgb=np.array(Image.open(folder/'strict.png'));rawrgb=np.array(Image.open(folder/'raw.png'));raw=np.load(folder/'raw.npz');alpha=np.load(folder/'strict.npz')['accumulation'][...,0].clip(0,1);ma=np.array(Image.open(op/'body_matte_cleanup_20261112/sequence/alpha'/f'{i:03}.png'))/255
  gate,solid,distance,thin=exterior_gate(alpha)
  lo=max(k for k in K if k<=i);hi=min(k for k in K if k>=i);t=0 if hi==lo else (i-lo)/(hi-lo);boxes=((1-t)*np.array(K[lo])+t*np.array(K[hi]))*4
  guard=thin.copy();overlap=np.zeros_like(guard)
  for x0,y0,x1,y1 in boxes[2:]:guard|=(xx>=x0-20)&(xx<=x1+20)&(yy>=y0-20)&(yy<=y1+20)
  # Conservative overlap of the two standing actors, plus other actor boxes.
  actorboxes=[boxes[0],boxes[1],boxes[5],np.array([1200,500,1920,1080])]
  for j,a in enumerate(actorboxes):
   for b in actorboxes[j+1:]:
    x0,y0=np.maximum(a[:2],b[:2])-24;x1,y1=np.minimum(a[2:],b[2:])+24
    if x1>x0 and y1>y0:overlap|=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
  guard|=overlap;gate[guard]=0
  factor=1-gate*(1-np.minimum(1,ma/np.maximum(alpha,1e-4)))
  clean=np.uint8(np.clip(rgb*factor[...,None]+.5,0,255));protected=(distance>=3)|guard
  assert np.array_equal(clean[protected],rgb[protected])
  # Separate repair of density-gate holes in overlapping torsos, using original raw RGB.
  envelope=ndi.binary_fill_holes(cv2.morphologyEx(np.uint8(alpha>.1),cv2.MORPH_CLOSE,np.ones((17,17),np.uint8)))
  inside=ndi.distance_transform_edt(envelope)>6
  row=path['records'][i];p=np.array(row['pose']);fx,fy,cx,cy=row['intrinsics'];rays=np.stack([(xx+.5-cx)/fx,-(yy+.5-cy)/fy,-np.ones_like(xx)],-1);rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
  z=(p[:3,3]+(rays@p[:3,:3].T)*raw['depth'])[...,2]
  repair=overlap & inside & (alpha<.98)&(raw['accumulation'][...,0]>.98)&(z>.18)&(z<.32)
  clean[repair]=rawrgb[repair]
  dest=out/'neural'/f'{i:03}';dest.mkdir(parents=True,exist_ok=True);Image.fromarray(clean).save(dest/'strict.png');Image.fromarray(np.uint8(repair)*255).save(dest/'interior_repair.png')
  receipts.append(dict(index=i,matting_protected_max_change=0,exterior_changed=int(np.any(np.uint8(np.clip(rgb*factor[...,None]+.5,0,255))!=rgb,axis=-1).sum()),separate_raw_torso_repair_pixels=int(repair.sum())))
  if i in [0,5,10,15,20,35,44,46,48,50,52,60]:
   previous=np.array(Image.open(op/'flying_previous_bodyclean_20261112/neural'/f'{i:03}'/'strict.png'));panel=Image.new('RGB',(1920,540))
   for col,(im,label) in enumerate([(previous,'previous'),(clean,'exterior only')]):
    q=Image.fromarray(im).resize((960,540));ImageDraw.Draw(q).text((10,10),f'{i} {label}',fill='white');panel.paste(q,(col*960,0))
   panel.save(out/f'compare_{i:03}.jpg')
 # Native photographic tail: SAM-defined union geometry, fully opaque interior;
 # Mat2 only in external band. No per-person alpha multiplication at overlaps.
 (out/'gt').mkdir(exist_ok=True);src=op/'flying_strict_20261112/gt_sources';tail=[]
 for row in path['records'][70:]:
  f=row['source_frame'];im=cv2.imread(str(src/f'{f:06}_rgb.png'));sam=cv2.imread(str(src/f'{f:06}_alpha.png'),0);ma=cv2.imread(str(op/'matanyone2_hair_20261112/native_soft_prompt/alpha'/f'{f:06}.png'),0)/255
  mask=np.uint8(sam>=128);closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8));y,x=np.mgrid[:mask.shape[0],:mask.shape[1]];closed[(x<1200)&(y>1220)]=mask[(x<1200)&(y>1220)]
  solid=ndi.binary_fill_holes(closed);dist=ndi.distance_transform_edt(solid)
  # Use unfilled geometry near the confirmed elbow opening, retaining real background gaps.
  dist[(x<1200)&(y>1220)]=ndi.distance_transform_edt(closed)[(x<1200)&(y>1220)]
  outer=ndi.distance_transform_edt(~solid);band=(dist<16)&(outer<=12)
  a=np.where(dist>=16,1.,0.);a[band]=ma[band]
  # Ensure Mat2 never modifies deeply interior pixels.
  assert np.all(a[dist>=16]==1)
  x0,y0,w,h=row['native_crop'];x0-=2665;y0-=650;comp=im[y0:y0+h,x0:x0+w].astype(float)*a[y0:y0+h,x0:x0+w,None]
  if (w,h)!=(1920,1080):comp=cv2.resize(comp,(1920,1080),interpolation=cv2.INTER_LANCZOS4)
  cv2.imwrite(str(out/'gt'/f'{row["index"]:03}.png'),np.uint8(np.clip(comp+.5,0,255)));tail.append(dict(index=row['index'],interior_alpha_min=1))
 (out/'policy_receipt.json').write_text(json.dumps(dict(policy='external boundary only; no Mat2 reduction in interiors, thin structures, guarded props/hair or possible actor overlaps',neural=receipts,tail=tail),indent=2))
 sys.path.insert(0,str(root/'tools'));from flying_delivery import assemble;from flying_encode import encode
 assemble(out);encode(out);old=out/'Marshawn908_1267_flying_strict_BC_slowmo_24fps.mp4';new=out/'Marshawn_exterior_only_matte.mp4';old.replace(new)
 receipt=json.loads((out/'delivery_receipt.json').read_text());receipt['video']=str(new);(out/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2));print('DONE',new,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
