"""Descriptor/USAC-selected independent1700 static tracks, supplied rig evaluated AFTER matching."""
from pathlib import Path
import json,itertools,hashlib
import numpy as np,cv2
from PIL import Image,ImageDraw
from audit_marshawn_geometry import projection,fundamental,sampson
R=Path('/fsx/tmp/marshawn_001660_hd');O=R/'static_calibration1700'
# Rectangles manually reviewed in independent1700 1024x540 source previews; coordinates multiply4.
ROI={'DC':[[0,0,280,190],[280,0,540,85]],'DB':[[40,0,320,170],[340,0,590,45]],'CC':[[0,150,180,350],[180,160,430,230]],'EB':[[0,0,310,190],[310,0,570,60]],'DD':[[0,205,400,290],[0,290,145,390]],'HC':[[360,90,540,235],[570,85,1000,160]],'HD':[[290,175,485,310],[580,170,890,235]],'IB':[[450,0,730,85],[770,0,1000,80]],'ID':[[400,160,515,290],[610,165,1000,240]],'HB':[[350,0,450,20],[820,0,995,20],[0,200,100,320]],'KA':[[695,0,930,45],[0,0,260,85],[105,175,180,230]],'KB':[[550,0,865,90],[940,0,1024,80],[0,0,280,170],[20,265,100,325]],'LB':[[670,0,970,140],[0,0,330,140],[110,220,190,275]],'LA':[[680,0,970,60],[0,0,340,85],[120,190,200,245]],'JA':[[470,0,730,40],[890,0,1010,75],[0,0,190,100]]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
 return dict(n=len(x),percentiles_rawpx=np.percentile(np.asarray(x)*2,[10,50,90,95,99]).tolist())if len(x)else dict(n=0)
def triangulate(group,features,rows):
 A=[]
 for c,k in group:
  P,*_=projection(rows[c]);u,v=features[c]['uv'][k];A.extend([u*P[2]-P[0],v*P[2]-P[1]])
 _,_,vh=np.linalg.svd(A);X=vh[-1,:3]/vh[-1,3];err=[];depth=[]
 for c,k in group:
  P,*_=projection(rows[c]);q=P@np.r_[X,1];err.append(float(np.linalg.norm(q[:2]/q[2]-features[c]['uv'][k])));depth.append(q[2])
 return X,err,depth

def main():
 cv2.setNumThreads(2);src=json.loads((O/'sources.json').read_text());rows={q['camera']:q['row']for q in src['records']};features={};pairs={};pair_report=[]
 for name in rows:
  im=cv2.imread(str(O/(name+'_4k.png')),cv2.IMREAD_GRAYSCALE);mask=np.zeros(im.shape,np.uint8)
  for x0,y0,x1,y1 in ROI[name]:mask[y0*4+4:y1*4-4,x0*4+4:x1*4-4]=255
  sift=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.008,edgeThreshold=12);kp,desc=sift.detectAndCompute(im,mask);uv=np.array([k.pt for k in kp])+ .5;features[name]=dict(uv=uv,descriptors=desc);np.savez_compressed(O/(name+'_static_features.npz'),detected_uv=uv,descriptors=desc);preview=Image.open(O/(name+'_preview.jpg')).convert('RGB');d=ImageDraw.Draw(preview)
  for rect in ROI[name]:d.rectangle(rect,outline='yellow',width=2)
  preview.save(O/(name+'_static_ROI.jpg'),quality=94);print('features',name,len(uv),flush=True)
 # Unconstrained F estimate depends only on observed coordinates/descriptors. GIVEN rig not a selection filter.
 bf=cv2.BFMatcher()
 for target,trains in src['groups'].items():
  for a,b in itertools.combinations([target]+trains,2):
   if (a,b)in pairs or (b,a)in pairs:continue
   fa,fb=features[a],features[b];aa=bf.knnMatch(fa['descriptors'],fb['descriptors'],k=2);bb=bf.knnMatch(fb['descriptors'],fa['descriptors'],k=2);back={m.queryIdx:m.trainIdx for m,n in bb if m.distance<.75*n.distance};m=[q for q,n in aa if q.distance<.75*n.distance and back.get(q.trainIdx)==q.queryIdx];ia=np.array([q.queryIdx for q in m],int);ib=np.array([q.trainIdx for q in m],int);ua,ub=fa['uv'][ia],fb['uv'][ib]
   if len(m)>=12:F,inlier=cv2.findFundamentalMat(ua,ub,cv2.USAC_MAGSAC,1.5,.999,10000);ok=inlier.ravel().astype(bool)if F is not None and F.shape==(3,3)else np.zeros(len(m),bool)
   else:F=None;ok=np.zeros(len(m),bool)
   rig=fundamental(rows[a],rows[b]);given=sampson(rig,ua,ub)if len(m)else np.empty(0);ind=sampson(F,ua,ub)if F is not None and F.shape==(3,3)else np.full(len(m),np.nan);pairs[(a,b)]=dict(a_indices=ia[ok],b_indices=ib[ok]);np.savez_compressed(O/f'pair_{a}_{b}.npz',descriptor_a_indices=ia,descriptor_b_indices=ib,detected_uv_a=ua,detected_uv_b=ub,unconstrained_F_inliers=ok,unconstrained_F=F,given_rig_F=rig,given_rig_sampson_4k=given,independent_F_sampson_4k=ind)
   rec=dict(cameras=[a,b],descriptor_mutual_ratio_count=len(m),unconstrained_F_inliers=int(ok.sum()),given_rig_on_ALL_descriptor_matches=stats(given),given_rig_on_independentF_inliers=stats(given[ok]),independentF_fit_residual=stats(ind[ok]));pair_report.append(rec);(O/'pair_progress.json').write_text(json.dumps(pair_report,indent=2));print('pair',a,b,len(m),int(ok.sum()),rec['given_rig_on_independentF_inliers'],flush=True)
 reports=[]
 def edge(a,b):
  if (a,b)in pairs:p=pairs[(a,b)];return zip(p['a_indices'],p['b_indices'])
  p=pairs[(b,a)];return zip(p['b_indices'],p['a_indices'])
 for target,trains in src['groups'].items():
  # Only TRAIN-TRAIN F-selected edges form 3D tracks; target features cannot join TRAIN tracks.
  parent={}
  def find(x):
   parent.setdefault(x,x)
   while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
   return x
  for a,b in itertools.combinations(trains,2):
   for i,j in edge(a,b):
    aa,bb=find((a,int(i))),find((b,int(j)))
    if aa!=bb:parent[bb]=aa
  groups={}
  for x in list(parent):groups.setdefault(find(x),[]).append(x)
  groups={k:g for k,g in groups.items()if len(g)>=3 and len({v[0]for v in g})==len(g)};targetlinks={}
  for a in trains:
   for ti,ai in edge(target,a):
    node=(a,int(ai))
    if node in parent:
     comp=find(node)
     if comp in groups:targetlinks.setdefault(int(ti),{}).setdefault(comp,set()).add(a)
  results=[];control=[]
  for ti,links in targetlinks.items():
   if len(links)!=1:continue
   comp,linkcams=next(iter(links.items()));g=groups[comp];X,trainerr,depth=triangulate(g,features,rows);P,*_=projection(rows[target]);h=P@np.r_[X,1];err=float(np.linalg.norm(h[:2]/h[2]-features[target]['uv'][ti]));rec=dict(target=target,target_keypoint=int(ti),target_detected_uv=features[target]['uv'][ti].tolist(),target_error_rawpixels=err*2,train_cameras=[q[0]for q in g],target_link_train_cameras=sorted(linkcams),train_reprojection_errors_rawpixels=(np.array(trainerr)*2).tolist(),xyz_train_only=X.tolist(),positive_depth=bool(min(depth)>0 and h[2]>0),train_observations=[dict(camera=c,keypoint=int(k),detected_uv=features[c]['uv'][k].tolist())for c,k in g]);results.append(rec)
   if len(g)==4:
    for left in range(4):
     leftcam,key=g[left];XX,ee,dd=triangulate(g[:left]+g[left+1:],features,rows);P,*_=projection(rows[leftcam]);hh=P@np.r_[XX,1];e=float(np.linalg.norm(hh[:2]/hh[2]-features[leftcam]['uv'][key]));control.append(dict(camera=leftcam,error_rawpixels=2*e,other_train_max_reprojection_rawpixels=float(max(ee)*2),positive_depth=bool(min(dd)>0 and hh[2]>0)))
  primary=[q for q in results if q['positive_depth']and max(q['train_reprojection_errors_rawpixels'])<3 and len(q['target_link_train_cameras'])>=2];allvalid=[q for q in results if q['positive_depth']];controls=[q for q in control if q['positive_depth']and q['other_train_max_reprojection_rawpixels']<3];report=dict(target=target,train_only_components_ge3=len(groups),all_target_associations=len(results),target_unfiltered_errors_rawpx=[q['target_error_rawpixels']for q in allvalid],primary_condition='>=3TRAINviews; allTRAINreproj<3rawpx; positive depth; independenttargetdescriptor/Flinks>=2TRAINcams; NOtargeterrorfilter',primary_errors_rawpixels=[q['target_error_rawpixels']for q in primary],primary_count=len(primary),primary_percentiles_rawpixels=np.percentile([q['target_error_rawpixels']for q in primary],[10,50,90,95,99]).tolist()if primary else None,train_leave1of4_control=controls,all_observations=results);reports.append(report);(O/'track_progress.json').write_text(json.dumps(reports,indent=2))
 result=dict(frame=1700,localframe=448,source_sha256=sha(O/'sources.json'),script_sha256=sha(__file__),static_ROIs_preview1024=ROI,unconstrainedF_method='Mutualratio.75 +USAC_MAGSAC1.5pixel4Kthreshold; GIVENrigNEVERusedformatchselection',all_errors_in_raw8K_pixels=True,true_detected_uv_preserved=True,source_coordinates_in_4k_colmap_centers=True,pairs=pair_report,targets=reports,no_1660_eval_RGB_used=True,no_calibration_or_NN_changes=True,manual_matches_review_pending=True);(O/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',[(q['target'],q['primary_count'],q['primary_percentiles_rawpixels'])for q in reports],flush=True)
if __name__=='__main__':main()
