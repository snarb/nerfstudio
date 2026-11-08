from pathlib import Path
import sys,json,time,numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from crew_guard_plan_v53 import candidates
from marshawn_actor_coordinates import raw_mask_coverage_to_HD
from marshawn_actor_confidence import native_confidence

def prepare():
 frame=1690;cam='NB';f=V/'frames/001690';native=f/'native_masks/NB.npz';cp,m=candidates(V,frame,cam);approval=V/'operations/root_1690_NB5_unresolved_PRE_guard_approval.json';assert approval.exists();source=V/'sources/1690/NB/receipt.json';s=json.loads(source.read_text());meta=json.loads(native.with_suffix('.json').read_text());union=json.loads((f/meta['union_resolution']['receipt']).read_text());prepath=Path(union['pre_overlap_archive'])
 with np.load(cp)as z:P=z['masks'][5]&z['masks'][[x for x in m['selected'].values()if x is not None]].any(0)
 assert P.sum()==3511
 with np.load(prepath)as z:pre=z['pre_black']|z['pre_grey']
 assert not(P&pre).any()
 with np.load(native)as z:
  A=z['black']|z['grey'];H=z['qualified_actor_rgb'];veto=z['source_ineligible'];unknown=z['unknown'];assert not(P&(A|H)).any()and veto[P].all()and unknown[P].all()
 scope=raw_mask_coverage_to_HD(P.astype('float32'),s['camera_row'])>1e-6;vetoHD=raw_mask_coverage_to_HD(veto.astype('float32'),s['camera_row'])>1e-6;assert vetoHD[scope].all();weights={}
 for width in [32,64]:
  raw,_=native_confidence(A,veto,width);weight=raw_mask_coverage_to_HD(raw,s['camera_row']);weight[vetoHD]=0;assert(weight[scope]==0).all();weights['W'+str(width)]=dict(scope_pixels=int(scope.sum()),positive_pixels=0,sum=0.0)
 oldfiles={str(p.relative_to(f/'data')):sha(p)for p in(f/'data').rglob('*')if p.is_file()};native_sha=sha(native);out=f/'qualified_unresolved_PRE_inputs/NB_person5';out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'mask.npz',unresolved_RAW_selected_PRE=P)
 proof=dict(kind='ROOT_SCOPED_UNRESOLVED_RAW_SELECTED_PRE_GUARD_EXCEPTION',frame=frame,camera=cam,instance=5,scope_pixels=3511,source_receipt_sha256=sha(source),candidate_archive_sha256=sha(cp),native_archive_sha256=native_sha,mask_archive_sha256=sha(out/'mask.npz'),mask_array_sha256=array_sha(P),refined_PRE_archive=str(prepath),refined_PRE_archive_sha256=sha(prepath),ROOT_approval_path=str(approval),ROOT_approval_sha256=sha(approval),projected_scope_W32_W64_zero_before_guard=True,projected_frozen_confidence_proof=weights,geometry_UNKNOWN_scope_all1=True,current_actor_and_H_scope_all0=True,source_ineligible_scope_all1=True,no_DATA_or_native_arrays_changed=True,all_existing_DATA_files_sha256=oldfiles,classification='UNRESOLVED_SOURCE_VISIBILITY_NOT_NONACTOR_NOT_CREW_NOT_FOREGROUND_NOT_FREE',raw_selected_PRE_preserved_for_audit=True,generic_current_PRE_hard_fail_unchanged=True,created=time.time());assert native_sha==sha(native)
 for path,digest in oldfiles.items():assert sha(f/'data'/path)==digest
 write(out/'proof.json',proof);return proof
if __name__=='__main__':prepare()
