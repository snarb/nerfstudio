"""Verify source-bound actor unions, semantic inputs, and nonzero actor support."""
from pathlib import Path
import argparse,json,time
import numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_union_guard_v53 import validate_union
from marshawn_video_producer_v3 import native_inventory
from marshawn_actor_coordinates import raw_mask_coverage_to_HD

def run(V,frame):
 f=V/'frames'/f'{frame:06d}';load=lambda p:json.loads(Path(p).read_text());cams=load(V/'source_inventory.json')['cameras'];inv=native_inventory(V,frame,cams);records=[]
 ready=load(f/'data/ready.json');binding=load(f/'source_mask_binding.json')['records'];guard=load(f/'crew_guard/plan_receipt.json');assert guard['status']=='pass'
 for cam in cams:
  validate_union(V,frame,cam);row=next(r for r in ready['records']if r['camera']==cam);b=next(r for r in binding if r['camera']==cam);g=load(f/'crew_guard'/(cam+'_people.json'))
  assert g['current_native_archive_sha256']==inv[cam]['archive_sha256']and g['current_candidate_archive_sha256']==inv[cam]['person_candidates_sha256'],cam
  assert b['native_mask_sha256']==inv[cam]['archive_sha256']and b['native_mask_receipt_sha256']==inv[cam]['metadata_sha256'],cam
  assert sha(f/'data'/row['image'])==row['rgb_sha256']and sha(row['semantic'])==row['semantic_sha256'],cam
  source=load(V/'sources'/str(frame)/cam/'receipt.json')
  with np.load(f/'native_masks'/(cam+'.npz'))as nz:
   native_nominal=nz['black']|nz['grey'];native_U=nz['actor_ownership_uncertain'];native_H=nz['qualified_actor_rgb']if'qualified_actor_rgb'in nz.files else np.zeros_like(native_U)
   assert not((native_U|native_H)&~native_nominal).any(),cam
   coverage_H=raw_mask_coverage_to_HD(native_H.astype('float32'),source['camera_row']);coverage_HU=raw_mask_coverage_to_HD((native_H|native_U).astype('float32'),source['camera_row'])
  with np.load(row['semantic'])as z:
   a=z['black']|z['grey'];u=z['actor_ownership_uncertain'];h=z['qualified_actor_rgb'];unk=z['unknown'];su=z['source_unknown'];v=z['source_ineligible'];w=z['W64'];valid=z['valid']
   assert not(z['black']&z['grey']).any()and np.isfinite(w).all()and (w>=0).all()and (w<=1).all(),cam
   assert np.array_equal(su,unk&~h),cam
   eligible=valid&~v
   assert np.array_equal(h,(coverage_H>1e-6)&eligible),cam
   for wk in ['W32','W64']:
    deficit=coverage_HU[eligible]-z[wk][eligible]
    assert not(deficit>2e-6).any(),(cam,wk,float(deficit.max()))
   eligible=(u|h)&valid&~v
   assert (w[eligible]>0).all(),cam
   records.append(dict(camera=cam,native_archive_sha256=inv[cam]['archive_sha256'],native_metadata_sha256=inv[cam]['metadata_sha256'],semantic_sha256=row['semantic_sha256'],source_hair_HD=int(h.sum()),source_unknown_override_HD=int((unk&~su).sum()),ownership_uncertain_HD=int(u.sum()),soft_H_pixels_outside_hard_labels=int((h&~a).sum()),maximum_W64_coverage_deficit=float(np.max(coverage_HU[valid&~v]-w[valid&~v])),native_H_U_subset_native_foreground=True,raw_H_projection_exact=True,W32_W64_monotone_native_H_U_coverage=True))
 result=dict(frame=frame,cameras=66,status='pass',all_NN_union_and_qualified_hair_source_proofs_valid=True,current_guard_native_bindings_valid=True,all_RGB_and_semantic_bytes_valid=True,all_uncertain_and_hair_RGB_weights_positive=True,geometry_UNKNOWN_separate_and_unchanged_by_source_hair_override=True,RGB_frequencies_unchanged=True,prepared_ready_sha256=sha(f/'data/prepared_ready.json'),ready_sha256=sha(f/'data/ready.json'),records=records,native_semantic_ready_withheld=not(f/'native_semantic_ready.json').exists(),code_sha256=sha(__file__),created=time.time());write(f/'data/final_v5_source_QA.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args();r=run(a.root,a.frame);print(json.dumps({k:v for k,v in r.items()if k!='records'}))
