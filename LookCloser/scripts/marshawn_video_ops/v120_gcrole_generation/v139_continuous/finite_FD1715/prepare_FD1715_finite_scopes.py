"""Read-only scope preparation for ROOT-reviewed CURRENT FD1715 components."""
from pathlib import Path
import sys,json,time,cv2,numpy as np
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(G/'operations'),str(G/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_clean_source_context_v134 import configure
configure()
from crew_guard_plan_v53 import candidates
from marshawn_current_parts_v105 import crew_core
cv2.setNumThreads(1);F=G/'frames/001715';O=G/'operations/1715_FD_components_current_positive';O.mkdir(exist_ok=True)
p=F/'native_masks/FD.npz';nh=sha(p);cp,cm=candidates(G,1715,'FD');ch=sha(cp)
with np.load(p)as z:old={k:z[k].copy()for k in z.files}
with np.load(cp)as z:people=z['masks'];M=people[8];rawPRE=people[[x for x in cm['selected'].values()if x is not None]].any(0)
_,lab=cv2.connectedComponents(M.astype('uint8'),8);A=old['black']|old['grey']|old['qualified_actor_rgb'];positive=np.isin(lab,[1,2,4,7,9,10,11]);R=positive&~A;P=np.isin(lab,[5,6]);assert P.sum()==283 and (P&rawPRE).sum()==283
meta=json.loads(p.with_suffix('.json').read_text());ur=json.loads((F/meta['union_resolution']['receipt']).read_text());pre=Path(ur['pre_overlap_archive']);assert sha(pre)==ur['pre_overlap_archive_sha256']
with np.load(pre)as z:refined=z['pre_black']|z['pre_grey']
assert not(P&(A|refined)).any()and old['source_ineligible'][P].all()and old['unknown'][P].all()
crew,crewproof=crew_core(1715,'FD',people)
assert not(R&crew).any(),'Positive proposal overlaps independently current-qualified CREW'
role=F/'role_unknown_visibility_inputs_v135/FD/CURRENT_NN_role_UNKNOWN.npz'
if role.exists():
 with np.load(role)as z:U=z['role_unknown'];soft=z['soft_eligible']
else:U=np.zeros_like(A);soft=np.zeros_like(A)
for path,arrays in [(O/'exact_CURRENT_NN_positive.npz',dict(qualified_new_union=R)),(O/'exact_CURRENT_RAW_PRE_photo283.npz',dict(unresolved_RAW_proposal=P))]:
 with path.open('wb')as f:np.savez_compressed(f,**arrays)
rows=[]
for k in range(1,12):
 c=lab==k;rows.append(dict(component=k,pixels=int(c.sum()),new_union_pixels=int((R&c).sum()),current_actor_H_pixels=int((A&c).sum()),positive=bool(k in [1,2,4,7,9,10,11]),independent_CURRENT_CREW_overlap_new=int((R&c&crew).sum()),role_UNKNOWN_overlap_new=int((R&c&U).sum()),mask_sha256=array_sha(c)))
assert sha(p)==nh and sha(cp)==ch
proof=dict(frame=1715,camera='FD',instance=8,positive_components=[1,2,4,7,9,10,11],unchanged_components=[3,5,6,8],candidate_archive=str(cp),candidate_archive_sha256=ch,native_before_sha256=nh,source_receipt_sha256=sha(G/'sources/1715/FD/receipt.json'),source_native_image_sha256=json.loads((G/'sources/1715/FD/receipt.json').read_text())['raw_native_image_sha256'],refined_PRE_archive=str(pre),refined_PRE_archive_sha256=sha(pre),proposal_archive=str(O/'exact_CURRENT_NN_positive.npz'),proposal_archive_sha256=sha(O/'exact_CURRENT_NN_positive.npz'),exact_scope_array_sha256=array_sha(R),new_actor_union_pixels=int(R.sum()),independent_CURRENT_CREW_overlap_new=int((R&crew).sum()),CURRENT_CREW_proofs=crewproof,photo_RAW_PRE283_archive=str(O/'exact_CURRENT_RAW_PRE_photo283.npz'),photo_RAW_PRE283_archive_sha256=sha(O/'exact_CURRENT_RAW_PRE_photo283.npz'),photo_RAW_PRE283_array_sha256=array_sha(P),RAW_PRE283_pixels=int(P.sum()),RAW_PRE283_refined_PRE_overlap0=True,RAW_PRE283_actor_H_overlap0=True,RAW_PRE283_veto_UNKNOWN_all1=True,role_UNKNOWN_original_pixels=int(U.sum()),role_UNKNOWN_intersection_positive=int((U&R).sum()),role_UNKNOWN_remaining_pixels=int((U&~R).sum()),component_counts=rows,diagnostic_proof_sha256=sha(G/'operations/1715_FD8_readonly/proof.json'),source_photos={str(x):sha(x)for x in sorted((G/'operations/1715_FD8_readonly').glob('*.jpg'))},no_source_arrays_changed=True,created=time.time())
write(O/'scope_proof.json',proof);print(json.dumps(dict(path=str(O/'scope_proof.json'),sha256=sha(O/'scope_proof.json'),new_actor_union_pixels=int(R.sum()),role_UNKNOWN_remaining_pixels=int((U&~R).sum()),crew_overlap=int((R&crew).sum())),indent=2),flush=True)
