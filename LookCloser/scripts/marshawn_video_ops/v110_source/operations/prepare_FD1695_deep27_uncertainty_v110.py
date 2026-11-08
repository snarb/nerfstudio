from pathlib import Path
import sys,json,time,numpy as np,cv2
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from crew_guard_plan_v53 import candidates
from marshawn_union_resolution_v5 import array_sha
from marshawn_actor_coordinates import raw_mask_coverage_to_HD
from marshawn_actor_confidence import native_confidence

def prepare():
 frame=1695;cam='FD';f=V/'frames/001695';native=f/'native_masks/FD.npz';cp,m=candidates(V,frame,cam);source=V/'sources/1695/FD/receipt.json';s=json.loads(source.read_text());coords=V/'operations/1695_FD2_readonly/deep27.json';xy=json.loads(coords.read_text())['native_xy'];approval=V/'operations/root_1695_FD2_deep27_unresolved_proposal_approval.json';assert approval.exists();P=np.zeros((4320,8192),bool)
 for x,y in xy:P[y,x]=True
 assert P.sum()==27
 with np.load(cp)as z:RAW=z['masks'][[v for v in m['selected'].values()if v is not None]].any(0);M=z['masks'][2]
 D=cv2.distanceTransform(RAW.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);assert not(P&~RAW).any()and not(P&~M).any()and(D[P]>32).all()and D[P].max()<34.366
 meta=json.loads(native.with_suffix('.json').read_text());up=json.loads((f/meta['union_resolution']['receipt']).read_text());pre=Path(up['pre_overlap_archive'])
 with np.load(pre)as z:refined=z['pre_black']|z['pre_grey'];assert not(P&refined).any()
 with np.load(native)as z:
  A=z['black']|z['grey'];H=z['qualified_actor_rgb'];veto=z['source_ineligible'];U=z['unknown'];assert not(P&(A|H)).any()and veto[P].all()and U[P].all()
 coverage=raw_mask_coverage_to_HD(P.astype('float32'),s['camera_row'])>1e-6;vHD=raw_mask_coverage_to_HD(veto.astype('float32'),s['camera_row'])>1e-6;weights={}
 for width in [32,64]:
  raw,_=native_confidence(A,veto,width);HD=raw_mask_coverage_to_HD(raw,s['camera_row']);HD[vHD]=0;weights['W'+str(width)]=dict(covered_HD_pixels=int(coverage.sum()),positive_HD_pixels=int((HD[coverage]>0).sum()),actual_scope_sum=float(HD[coverage].sum()))
 files={str(p.relative_to(f/'data')):sha(p)for p in(f/'data').rglob('*')if p.is_file()};before=sha(native);out=f/'guard_proposal_uncertainty_inputs'/cam;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'mask.npz',unresolved_RAW_proposal=P)
 proof=dict(kind='ROOT_SCOPED_RAW_PROPOSAL_UNCERTAINTY_ONLY',frame=frame,camera=cam,instance=2,scope_pixels=27,source_receipt_sha256=sha(source),candidate_archive_sha256=sha(cp),native_archive_sha256=before,mask_archive_sha256=sha(out/'mask.npz'),mask_array_sha256=array_sha(P),refined_PRE_archive_sha256=sha(pre),ROOT_approval_path=str(approval),ROOT_approval_sha256=sha(approval),exact_coordinates_sha256=sha(coords),physical_geometry_UNKNOWN_and_veto_scope_all1=True,current_A_H_refinedPRE_scope_all0=True,original_frozen_confidence_projected_scope=weights,no_RGB_weight_mask_or_grid_array_changed=True,no_actor_NONACTOR_CREW_or_FREE_certificate=True,generic32_rule_unchanged=True,all_existing_DATA_files_sha256=files,created=time.time());assert sha(native)==before
 for p,h in files.items():assert sha(f/'data'/p)==h
 write(out/'proof.json',proof);return proof
if __name__=='__main__':prepare()
