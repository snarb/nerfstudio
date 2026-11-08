from pathlib import Path
import sys,json,time
import cv2,numpy as np
from PIL import Image
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(G/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_actor_coordinates import raw_mask_coverage_to_HD
from marshawn_actor_confidence import native_confidence
from marshawn_union_resolution_v5 import array_sha
cv2.setNumThreads(2)
O=G/'operations/1710_JE12226_current_positive';O.mkdir(exist_ok=True);Q=G/'operations/current_native_parts_v105/frame001710/qualification.json';q=json.loads(Q.read_text());r=q['records'][105];p=Path(r['NN_pair']['role_unknown_current_NN_archive']);assert sha(p)==r['NN_pair']['role_unknown_current_NN_archive_sha256'];assert r['camera']=='JE'and r['NN_pair']['originally_proposed_actor_union_pixels']==12226
with np.load(p)as z:R=np.zeros((4320,8192),bool);x0,y0,x1,y1=z['crop_xyxy'];R[y0:y1,x0:x1]=z['role_unknown'];assert R.sum()==12226
native=G/'frames/001710/native_masks/JE.npz'
with np.load(native)as z:old={k:z[k].copy()for k in z.files}
assert not(R&(old['black']|old['grey']|old['qualified_actor_rgb'])).any();after={k:x.copy()for k,x in old.items()};after['black']|=R;after['qualified_actor_rgb']|=R;after['actor_ownership_uncertain']|=R;after['source_ineligible']&=~R
np.savez_compressed(O/'exact_CURRENT_NN_positive.npz',qualified_new_union=R)
SOURCE=Path('/fsx/tmp/marshawn_001660_hd/color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png');tf=json.loads((SOURCE/'transforms.json').read_text());row=next(r for r in tf['frames']if r['physical_camera']=='JE');scope=raw_mask_coverage_to_HD(R.astype('float32'),row);valid=np.asarray(Image.open(SOURCE/row['mask_path']).convert('L'))>0
results={};arrays={}
for label,values in [('before',old),('after',after)]:
 nominal=values['black']|values['grey'];y,x=np.where(nominal);l=max(0,int(x.min())-129);t=max(0,int(y.min())-129);rr=min(8192,int(x.max())+130);b=min(4320,int(y.max())+130);veto=raw_mask_coverage_to_HD(values['source_ineligible'].astype('float32'),row)>1e-6;results[label]={}
 for width in [32,64]:
  crop,_=native_confidence(nominal[t:b,l:rr],values['source_ineligible'][t:b,l:rr],width);full=np.zeros((4320,8192),np.float32);full[t:b,l:rr]=crop;W=raw_mask_coverage_to_HD(full,row);W[~valid|veto]=0;S=(scope>1e-6)&valid;results[label]['W'+str(width)]={'scope_HD_pixels':int(S.sum()),'positive_pixels':int((W[S]>0).sum()),'zero_pixels':int((W[S]<=0).sum()),'sum':float(W[S].sum()),'maximum':float(W[S].max()),'coverage_weighted_mean':float((W*scope).sum()/max(scope.sum(),1e-12))};arrays[label+'_W'+str(width)]=W
np.savez_compressed(O/'projected_frozen_weights_before_after.npz',scope_coverage=scope,**arrays)
write(O/'simulation.json',dict(kind='FINITE_CURRENT_NATIVE_NN_PHOTO_POSITIVE_SIMULATION',frame=1710,camera='JE',native_before_sha256=sha(native),qualification_sha256=sha(Q),original_role_UNKNOWN_archive=str(p),original_role_UNKNOWN_archive_sha256=sha(p),exact_scope_sha256=array_sha(R),exact_scope_pixels=12226,proposal_archive=str(O/'exact_CURRENT_NN_positive.npz'),proposal_archive_sha256=sha(O/'exact_CURRENT_NN_positive.npz'),weights=results,native_geometry_UNKNOWN_unchanged=True,native_gray_and_all_old_actor_pixels_preserved=True,source_veto_before_pixels=int((old['source_ineligible']&R).sum()),physical_UNKNOWN_before_pixels=int((old['unknown']&R).sum()),not_applied=True,current_NN_only=True,created=time.time()));print(json.dumps(results,indent=2))
