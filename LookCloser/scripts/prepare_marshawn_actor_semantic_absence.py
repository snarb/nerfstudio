"""Conservative opt-in TRAIN semantic-absence ray proposal; never hard carving.

No alpha/depth labels. Protect all historical subject witness CELL corners as
well as selected nominal/hair support. The negative observation remains a
semantic/model hypothesis, requiring independent source review before training.
"""
from pathlib import Path
import argparse,hashlib,itertools,json
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt
from marshawn_actor_coordinates import raw_array_indices
from prepare_marshawn_actor_geometry import slab
R=Path('/fsx/tmp/marshawn_001660_hd')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def project(X,row):
 pose=np.asarray(row['transform_matrix']);v=(X-pose[:3,3])@pose[:3,:3];return np.c_[row['fl_x']*v[:,0]/-v[:,2]+row['cx'],row['fl_y']*(-v[:,1])/-v[:,2]+row['cy']],-v[:,2]
def protected_cells(row,xyz,box):
 # Whole witness cell envelope plus 2HD pixels calibration/raster margin.
 corner=np.asarray(list(itertools.product([-1,1],repeat=3)))*(box[1]-box[0])/256
 uv,z=project((xyz[:,None,:]+corner[None]).reshape(-1,3),row);uv=uv.reshape(-1,8,2);z=z.reshape(-1,8);mask=np.zeros((row['h'],row['w']),bool)
 for poly,depth in zip(uv,z):
  if np.any(depth<=0):continue
  mn=np.floor(poly.min(0)-2).astype(int);mx=np.ceil(poly.max(0)+2).astype(int);x0,y0=np.maximum(mn,0);x1,y1=np.minimum(mx,[row['w']-1,row['h']-1]);
  if x1>=x0 and y1>=y0:mask[y0:y1+1,x0:x1+1]=True
 return mask

def main(a):
 cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=False);m=json.loads(a.mask_manifest.read_text());tf=json.loads((Path(m['data_dir'])/'transforms.json').read_text());frames={r['physical_camera']:r for r in tf['frames']};b=json.loads(a.bounds.read_text());box=np.asarray(b['boxes'][a.variant]);body=np.load(R/'geometry/subject_witnesses.npz')['xyz'];historical_count=len(body)
 if a.actor_geometry:
  selection=np.load(a.actor_geometry/'point_selection.npz');cloud=np.load(R/'geometry/train_witnesses.npz')['xyz'];body=np.unique(np.vstack([body,cloud[selection['votes']>=3]]),axis=0)
 allparts=[];stats=[]
 for r in m['rows']:
  if r['split']!='TRAIN':continue
  cam=r['camera'];row=frames[cam];raw=np.load(r['raw_archive']);nominal=raw['nominal'];x0,y0,x1,y1=raw['crop_xyxy'];full=np.zeros((4320,8192),bool);full[y0:y1,x0:x1]=nominal;distance=distance_transform_edt(~full);hd=np.load(r['archive']);veto=hd['source_ineligible'];protected=protected_cells(row,body,box)
  yy,xx=np.mgrid[0:row['h']:a.stride,0:row['w']:a.stride];x=xx.ravel();y=yy.ravel();uv=np.c_[x+.5,y+.5];rawuv=raw_array_indices(uv,row);rd=cv2.remap(distance.astype(np.float32),rawuv[:,0].astype(np.float32).reshape(-1,1),rawuv[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel();valid=(rawuv[:,0]>=4)&(rawuv[:,0]<8188)&(rawuv[:,1]>=4)&(rawuv[:,1]<4316)&~veto[y,x];clear=(rd-8>256)&valid
  pose=np.asarray(row['transform_matrix']);D=np.c_[(uv[:,0]-row['cx'])/row['fl_x'],-(uv[:,1]-row['cy'])/row['fl_y'],-np.ones(len(uv))]@pose[:3,:3].T;D/=np.linalg.norm(D,axis=1,keepdims=True);O=np.broadcast_to(pose[:3,3],D.shape);lo,hi,bad=slab(O,D,box);start=np.maximum(lo,.052);candidate=clear&~bad&(hi>start);conflict=candidate&protected[y,x];selected=candidate&~protected[y,x]
  part={'origins':O[selected].copy(),'directions':D[selected],'t_start':start[selected],'t_end':hi[selected],'camera_idx':np.full(selected.sum(),len(stats),np.int64),'uv':uv[selected],'raw_uv':rawuv[selected],'conservative_raw_clearance_lower_bound':rd[selected]-8};allparts.append(part);stats.append({'camera':cam,'grid_proposed_pixels':len(uv),'actual_actor_box_crossing_clear_outside256':int(candidate.sum()),'whole_body_cell_projection_conflicts_REMOVED':int(conflict.sum()),'eligible_segments':int(selected.sum()),'after_protection_known_cell_conflicts':0,'source_actor_mask_raw_archive_sha256':r['raw_archive_sha256'],'mask_archive_sha256':r['archive_sha256']});print(json.dumps(stats[-1]),flush=True)
 p=a.output/'semantic_absence.npz';data={k:np.concatenate([part[k]for part in allparts])for k in allparts[0]};data['aabb']=box;data['camera_names']=np.asarray([r['camera']for r in stats]);np.savez_compressed(p,**data);receipt={'kind':'TRAIN_SEMANTIC_ABSENCE_ACTOR_SLAB_SOFT_TAU_CANDIDATES','uses_masks':True,'uses_eval_rgb':False,'approved_for_training':False,'mask_manifest':str(a.mask_manifest),'mask_manifest_sha256':sha(a.mask_manifest),'transforms_sha256':m['transforms_sha256'],'bounds_source':str(a.bounds),'bounds_source_sha256':sha(a.bounds),'variant':a.variant,'aabb':box.tolist(),'archive':str(p),'archive_sha256':sha(p),'minimum_raw_actor_clearance':256,'additional_native_kernel_calibration_margin':8,'stride_HD_pixels':a.stride,'protected_historical_subject_rows':historical_count,'protected_union_points':len(body),'actor_geometry':str(a.actor_geometry)if a.actor_geometry else None,'actor_geometry_manifest_sha256':sha(a.actor_geometry/'manifest.json')if a.actor_geometry else None,'protection':'projected ALL8corners of witness cell(boxextent/128) plus2HDpixels, not only point centre; sourceveto overrides; AFTER removal0 known projectedcell conflicts','normalizer_or_geometry_confidence_NOT_depth_truth':True,'no_witness_depth_or_alpha_GT':True,'soft_TRUEtau_only_no_hard_gridcarve':True,'same_geometry_all_loss_widths':True,'source_scope':'TRAIN63 only; nominal+acceptednativehair; verifiedothers/explicitocclusionUNKNOWN/invalid ineligible','global_count':len(data['t_start']),'camera_rows':stats,'limitations':'Unseen hair/limbs beyond conservativedilation remain segmentation uncertainty; these softabsence proposals need actual source contact review, not whole-volume free-space truth'};(a.output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['mask-manifest','bounds','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--actor-geometry',type=Path);p.add_argument('--variant',default='FOOT_WIDE');p.add_argument('--stride',type=int,default=8);main(p.parse_args())
