#!/usr/bin/env python3
"""Independent-frame card photometry PRECHECK; never applies corrections."""
import argparse,json,hashlib,sys
from pathlib import Path
import cv2,numpy as np,tifffile
from PIL import Image,ImageDraw
from prepare_marshawn_canonical_pq import vendor
RAW=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local')
# Native lens raster bounds, manually reviewed in independent-frame previews.
BOXES={'KC':[4000,800,8192,2350],'KA':[5000,0,8192,1300],'KB':[3900,0,8192,1500],'LB':[4200,0,8192,1700],'JA':[3000,0,7600,1300],'LA':[4400,0,8192,1600]}
# Manually identified STATIC printed-card rectangles in KC stride8 preview.
CARD_BOXES=[[687,118,709,145],[753,117,773,144],[777,123,797,149],[839,118,858,144],[866,125,890,153],[937,121,960,149],[970,127,995,157],[687,163,709,191],[745,164,765,190],[774,172,797,197],[834,170,857,197],[865,174,891,203],[935,174,961,201],[970,181,995,209],[557,119,574,145],[573,109,589,136],[618,110,637,139],[637,125,654,151],[556,155,572,181],[571,170,589,193],[617,157,636,184],[638,160,655,187],[555,203,574,229],[573,203,590,227],[617,212,637,239],[637,204,655,230]]
YAP1=np.array([.2722287168,.6740817658,.0536895174])
def sha_bytes(a):return hashlib.sha256(a).hexdigest()
def Hmap(H,xy):
 p=np.c_[xy,np.ones(len(xy))]@H.T;return(p[:,:2]/p[:,2:]).astype('float32')
def load(cam,local,pq):
 box=BOXES[cam];x0,y0,x1,y1=box;p=RAW/cam/f'{cam}.{local:06d}.tif';m=tifffile.memmap(p,mode='r');codes=np.array(m[y0:y1,x0:x1]);grey=np.rint(codes.astype('float32').mean(2)/257).astype('uint8');linear=np.maximum(pq.log3g10_decode(codes.astype('float64')/65535)@pq.rgb_to_rgb_matrix('rwg','ap1').T,0).astype('float32')
 return dict(box=box,grey=grey,rgb=linear,codes=codes.astype('float32'),source=dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,ROI=box,ROI_uint16_sha256=sha_bytes(codes.tobytes())))
def local_xy(d,xy):return np.asarray(xy,'float32')-np.array(d['box'][:2],'float32')
def patch(d,xy):
 # Compare exactly homologous pixel footprints via local H, not projected SfM poses.
 xy=local_xy(d,xy);v=cv2.remap(d['rgb'],xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1,3);code=cv2.remap(d['codes'],xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1,3)
 y=v@YAP1;mean=float(y.mean());cv=float(y.std()/max(mean,1e-12));spread=float((np.quantile(y,.95)-np.quantile(y,.05))/max(mean,1e-12));good=mean>.005 and np.isfinite(v).all()and v.min()>0 and code.min()>100 and code.max()<65000 and cv<.075 and spread<.23
 return v.mean(0),dict(valid=bool(good),cv=cv,spread=spread)
def photometry(ref,targets,Hs,card,box,frame,frozen_points=None):
 x0,y0,x1,y1=box;grid=np.array([(x,y)for y in range(y0-28,y1+29,12)for x in range(x0-28,x1+29,12)],'float32');delta=np.array([(x,y)for y in[-4,-2,0,2,4]for x in[-4,-2,0,2,4]],'float32');rows=[]
 g=ref['grey'];crop=g[max(0,y0-ref['box'][1]):y1-ref['box'][1],max(0,x0-ref['box'][0]):x1-ref['box'][0]];white=float(np.quantile(crop,.72))
 for j,xy in enumerate(grid):
  inside=x0+8<xy[0]<x1-8 and y0+8<xy[1]<y1-8;outside=xy[0]<x0-8 or xy[0]>x1+8 or xy[1]<y0-8 or xy[1]>y1+8
  if not(inside or outside):continue
  loc=local_xy(ref,xy);ix,iy=np.rint(loc).astype(int)
  if not 5<=ix<g.shape[1]-5 or not 5<=iy<g.shape[0]-5:continue
  pointid=card*10000+j
  if frozen_points is not None:
   if pointid not in frozen_points:continue
   material=frozen_points[pointid]
  else:
   if inside and g[iy,ix]<white:continue
   material='paper_white'if inside else'outside_cardstock'
  foot=xy+delta;rr,qr=patch(ref,foot)
  if not qr['valid']:continue
  pointid=card*10000+j;radius=float(np.linalg.norm((xy-[4096,2160])/[4096,2160]));rows.append(dict(frame=frame,camera='KC',card=card,point=pointid,material=material,uv=xy.tolist(),rgb=rr.tolist(),quality=qr,radius=radius))
  for cam,H in Hs.items():
   mapped=Hmap(H,foot);d=targets[cam];local=local_xy(d,mapped)
   if(local.min(0)<5).any()or(local.max(0)>[d['grey'].shape[1]-6,d['grey'].shape[0]-6]).any():continue
   rgb,q=patch(d,mapped)
   if not q['valid']:continue
   uv=Hmap(H,xy[None])[0];rows.append(dict(frame=frame,camera=cam,card=card,point=pointid,material=material,uv=uv.tolist(),rgb=rgb.tolist(),quality=q,radius=float(np.linalg.norm((uv-[4096,2160])/[4096,2160]))))
 return rows

def main(a):
 cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=False);pq=vendor();cams=list(BOXES);data={cam:load(cam,448,pq)for cam in cams};ref=data['KC'];sift=cv2.SIFT_create(nfeatures=15000,contrastThreshold=.006,edgeThreshold=18);features={}
 for cam,d in data.items():
  kp,de=sift.detectAndCompute(d['grey'],None);features[cam]=(np.array([k.pt for k in kp],'float32')+d['box'][:2],de)
 matcher=cv2.BFMatcher();matches={}
 def pair(source,target):
  key=(source,target)
  if key not in matches:
   p,de=features[source];q,dt=features[target];forward=matcher.knnMatch(de,dt,k=2);reverse=matcher.match(dt,de);back={m.queryIdx:m.trainIdx for m in reverse};good=[m for z in forward if len(z)==2 for m,n in[z]if m.distance<.72*n.distance and back.get(m.trainIdx)==m.queryIdx];matches[key]=(p[[m.queryIdx for m in good]],q[[m.trainIdx for m in good]])
  return matches[key]
 bindings=[];measures=[];HsByCard={};tiles=[]
 for card,b0 in enumerate(CARD_BOXES):
  box=np.array(b0)*8;x0,y0,x1,y1=box;known={'KC':np.eye(3)}
  for hop in range(3):
   changed=False
   for cam in cams[1:]:
    if cam in known:continue
    for via in ['KC','KB','LB','LA','JA']:
     if via not in known or via==cam:continue
     p,q=pair(via,cam);p=Hmap(np.linalg.inv(known[via]),p)
     choose=(p[:,0]>x0)&(p[:,0]<x1)&(p[:,1]>y0)&(p[:,1]<y1);pp=p[choose];qq=q[choose]
     if len(pp)<6:continue
     H,mask=cv2.findHomography(pp,qq,cv2.USAC_MAGSAC,1.5,maxIters=10000,confidence=.999)
     if H is None or mask.sum()<6:continue
     pp=pp[mask[:,0]>0];qq=qq[mask[:,0]>0];coverage=np.ptp(pp,axis=0)/[x1-x0,y1-y0]
     if coverage.min()<.3:continue
     xx,yy=np.meshgrid(np.linspace(x0,x1-1,128),np.linspace(y0,y1-1,128));grid=np.c_[xx.ravel(),yy.ravel()].astype('float32');mapped=local_xy(data[cam],Hmap(H,grid));target=cv2.remap(data[cam]['grey'],mapped[:,0].reshape(128,128),mapped[:,1].reshape(128,128),cv2.INTER_LINEAR);rr=local_xy(ref,grid);source=cv2.remap(ref['grey'],rr[:,0].reshape(128,128),rr[:,1].reshape(128,128),cv2.INTER_LINEAR);visible_mask=(mapped[:,0]>=2)&(mapped[:,1]>=2)&(mapped[:,0]<data[cam]['grey'].shape[1]-2)&(mapped[:,1]<data[cam]['grey'].shape[0]-2);visible_fraction=float(visible_mask.mean());ncc=float(np.corrcoef(source.ravel()[visible_mask],target.ravel()[visible_mask])[0,1])if visible_fraction>=.30 and source.ravel()[visible_mask].std()>5 and target.ravel()[visible_mask].std()>5 else -1.;rms=float(np.sqrt(np.mean(np.sum((Hmap(H,pp)-qq)**2,axis=1))))
     if ncc<.90:continue
     known[cam]=H;changed=True;bindings.append(dict(card=card,camera=cam,via=via,hop=hop,H=H.tolist(),inliers=len(pp),reference_uv=pp.tolist(),target_uv=qq.tolist(),reprojection_RMS_px=rms,normalized_pattern_correlation=ncc,pattern_visible_fraction=visible_fraction,reference_box=box.tolist()))
     tile=Image.new('RGB',(256,151),'#111111');dd=ImageDraw.Draw(tile);tile.paste(Image.fromarray(source).convert('RGB'),(0,23));tile.paste(Image.fromarray(target).convert('RGB'),(128,23));dd.text((3,5),f'card{card} KC|{cam} via{via} {ncc:.3f}',fill='white');tiles.append(tile);break
   if not changed:break
  Hs={k:v for k,v in known.items()if k!='KC'}
  if Hs:HsByCard[card]=(box,Hs);measures+=photometry(ref,{c:data[c]for c in Hs},Hs,card,box,1700)
  print('card',card,'matched',list(Hs),flush=True)
 for cam in cams:data[cam]['source']['global_frame']=1700
 sources=[d['source']for d in data.values()]
 # Freeze1700 local mappings and samplecoordinates before opening validationimages.
 (a.output/'frozen1700_matches.json').write_text(json.dumps(bindings,indent=2)+'\n')
 frozen_points={r['point']:r['material']for r in measures if r['camera']=='KC'}
 del data;data={cam:load(cam,488,pq)for cam in cams};ref=data['KC']
 validation_pattern=[]
 for card,(box,Hs)in HsByCard.items():
  x0,y0,x1,y1=box;xx,yy=np.meshgrid(np.linspace(x0,x1-1,128),np.linspace(y0,y1-1,128));grid=np.c_[xx.ravel(),yy.ravel()].astype('float32');rr=local_xy(ref,grid);source=cv2.remap(ref['grey'],rr[:,0].reshape(128,128),rr[:,1].reshape(128,128),cv2.INTER_LINEAR);valid={}
  for cam,H in Hs.items():
   mapped=local_xy(data[cam],Hmap(H,grid));target=cv2.remap(data[cam]['grey'],mapped[:,0].reshape(128,128),mapped[:,1].reshape(128,128),cv2.INTER_LINEAR);visible_mask=(mapped[:,0]>=2)&(mapped[:,1]>=2)&(mapped[:,0]<data[cam]['grey'].shape[1]-2)&(mapped[:,1]<data[cam]['grey'].shape[0]-2);visible_fraction=float(visible_mask.mean());ncc=float(np.corrcoef(source.ravel()[visible_mask],target.ravel()[visible_mask])[0,1])if visible_fraction>=.30 and source.ravel()[visible_mask].std()>5 and target.ravel()[visible_mask].std()>5 else -1.;validation_pattern.append(dict(card=card,camera=cam,NCC=ncc,pattern_visible_fraction=visible_fraction,passed=ncc>=.9))
   if ncc>=.9:valid[cam]=H
  if valid:measures+=photometry(ref,{c:data[c]for c in valid},valid,card,box,1740,frozen_points)
 for cam in cams:data[cam]['source']['global_frame']=1740
 sources +=[d['source']for d in data.values()];(a.output/'measurements.json').write_text(json.dumps(measures,indent=2)+'\n')
 # Equal weighting perphysicalcard, first split byreferencecardidentity; no target1660data.
 by={(r['frame'],r['point']):r for r in measures if r['camera']=='KC'};pairs=[]
 for r in measures:
  if r['camera']=='KC':continue
  refrow=by.get((r['frame'],r['point']))
  if refrow is None:continue
  ev=np.log2((np.array(r['rgb'])@YAP1)/(np.array(refrow['rgb'])@YAP1));pairs.append(dict(**r,log_luminance_ratio_EV=float(ev)))
 fitted={};summary=[]
 for cam in cams[1:]:
  fit=[r for r in pairs if r['camera']==cam and r['frame']==1700 and r['material']=='paper_white'and r['card']%2==0];cardEV={c:float(np.median([r['log_luminance_ratio_EV']for r in fit if r['card']==c]))for c in sorted({r['card']for r in fit})}
  correction=-float(np.median(list(cardEV.values())))if cardEV else None;fitted[cam]=correction;row=dict(camera=cam,correction_EV_relative_KC=correction,fit_physical_cards=len(cardEV),fit_patches=len(fit),fit_card_offsets_EV=cardEV,validation={})
  for frame in[1700,1740]:
   for material in['paper_white','outside_cardstock']:
    v=[r for r in pairs if r['camera']==cam and r['frame']==frame and r['material']==material and(frame!=1700 or r['card']%2==1)]
    before=np.array([r['log_luminance_ratio_EV']for r in v]);after=before+correction if correction is not None else np.array([])
    row['validation'][f'{frame}_{material}']=dict(patches=len(v),cards=len({r['card']for r in v}),median_abs_EV_before=float(np.median(abs(before)))if len(before)else None,median_abs_EV_after=float(np.median(abs(after)))if len(after)else None,p90_abs_EV_before=float(np.quantile(abs(before),.9))if len(before)else None,p90_abs_EV_after=float(np.quantile(abs(after),.9))if len(after)else None,signed_median_after=float(np.median(after))if len(after)else None,radius_range=[min(r['radius']for r in v),max(r['radius']for r in v)]if v else None)
  summary.append(row)
 sheet=Image.new('RGB',(4*256,((len(tiles)+3)//4)*151),'#111111')
 for i,t in enumerate(tiles):sheet.paste(t,(i%4*256,i//4*151))
 sheet.save(a.output/'matched_card_patterns_contact.jpg',quality=95)
 report=dict(kind='INDEPENDENT1700_CARD_EV_DIAGNOSTIC_NOT_APPLIED',uses_target1660_eval_RGB=False,corrections_applied=False,NN_updates=0,reference_camera='KC',method='Pattern-verified percardlocalH, nativeROIlinearAP1D60; medianpercardlogYratio/equalcardmedian; evenreferencecardsfit,oddcardsholdout; frozen1700mappingsvalidate1740',source_ROIs=BOXES,manual_reference_cards_stride8=CARD_BOXES,sources=sources,accepted_bindings=len(bindings),validation_pattern_checks=validation_pattern,frozen1700_reference_point_count=len(frozen_points),rows=summary,graph_connected_to_KC=all(r['fit_physical_cards']>=2 for r in summary),limitations=['RelativeEV conflates staticviewdependentpaperBRDF/lighting withsensorresponse; notISOgroundtruth.','No WB/vignette correction fitted.','Outsidecardstock assumes localplanarity close toprintedpaper; diagnosticmaterialholdout, no geometricGT.','Subsetmaylack sufficientdisjointcards/radiusspan; mustreportinconclusive ratherthanapply.'],script_sha256=sha_bytes(Path(__file__).read_bytes()))
 (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
