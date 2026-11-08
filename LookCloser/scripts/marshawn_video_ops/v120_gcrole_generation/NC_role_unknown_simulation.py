from pathlib import Path
import sys,json
import cv2,numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations/semantic_actor_role_v120';sys.path[:0]=[str(O),str(V/'operations'),str(V/'code/LookCloser/scripts')]
from launch_prior_ROLE_goldens_v123 import scope
from marshawn_actor_confidence import native_confidence
from marshawn_actor_coordinates import raw_mask_coverage_to_HD
from prepare_marshawn_tuning import sha,write
R,proof=scope(1700,'NC',34);p=V/'frames/001700/native_masks/NC.npz'
with np.load(p)as z:old={k:z[k].copy()for k in z.files}
with np.load(proof['current_preSEM_native'])as z:pre={k:z[k].copy()for k in z.files}
source=json.loads((V/'sources/1700/NC/receipt.json').read_text());row=source['camera_row'];coverage=raw_mask_coverage_to_HD(R.astype('float32'),row);hd=coverage>1e-6
with np.load(V/'frames/001700/data/semantics/NC.npz')as z:original={k:z[k].copy()for k in z.files}
nominal=(old['black']|old['grey'])&~R;yy,xx=np.where(nominal);x0=max(0,int(xx.min())-129);x1=min(8192,int(xx.max())+130);y0=max(0,int(yy.min())-129);y1=min(4320,int(yy.max())+130);records=[]
for name,veto in[('restore_preSEM_veto',np.where(R,pre['source_ineligible'],old['source_ineligible'])),('retain_visible_candidate_veto_0',old['source_ineligible'])]:
 weights={};vetoHD=raw_mask_coverage_to_HD(veto.astype('float32'),row)>1e-6
 for key,width in[('W32',32),('W64',64)]:
  c,_=native_confidence(nominal[y0:y1,x0:x1],veto[y0:y1,x0:x1],width);full=np.zeros((4320,8192),np.float32);full[y0:y1,x0:x1]=c;w=raw_mask_coverage_to_HD(full,row);w[~original['valid']|vetoHD]=0;weights[key]=w
 stats={}
 for k,w in weights.items():
  stats[k]=dict(scope_positive=int((w[hd]>0).sum()),scope_sum=float(w[hd].sum()),scope_max=float(w[hd].max()),scope_min=float(w[hd].min()),scope_full_weight_ones=int((w[hd]==1).sum()),all_image_changed_pixels=int((w!=original[k]).sum()),changed_pixels_outside_HD_R=int(((w!=original[k])&~hd).sum()),unchanged_frozen_confidence_formula=True)
 np.savez_compressed(O/(name+'_SIMULATION_ONLY.npz'),**weights,role_unknown_scope=hd);records.append(dict(variant=name,stats=stats,veto_native_R=int(veto[R].sum()),original_unknown_native_R=int(old['unknown'][R].sum()),role_assertion_removed_no_FG_added=True))
result=dict(kind='PRIVATE_NC1700_ROLE_UNKNOWN_WEIGHTS_NO_INPUT_MUTATION',frame=1700,camera='NC',scope_native_pixels=int(R.sum()),scope_HD_pixels=int(hd.sum()),current_native_sha256=sha(p),preSEM_native_sha256=sha(proof['current_preSEM_native']),records=records,all_original_Native_DATA_R_Role_arrays_UNCHANGED=True);write(O/'NC1700_role_UNKNOWN_confidence_simulation.json',result);print(json.dumps(result,indent=2))
