from pathlib import Path
import json,sys,time
import cv2,numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from crew_guard_plan_v53 import candidates
import marshawn_current_parts_v105 as core
core.V=G

def assess(frame,cam,index):
 qpath=V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'qualification.json';row=json.loads(qpath.read_text())['records'][index];assert row['camera']==cam;q=row['NN_pair'];x0,y0,x1,y1=row['ROI'];ms=[]
 for e in[q['a'],q['b']]:
  assert sha(e['archive'])==e['archive_sha256']
  with np.load(e['archive'])as z:ms.append(z['masks'][e['index']].copy())
 _,lab=cv2.connectedComponents((ms[0]&ms[1]).astype('uint8'),8);part=lab==q['NN_intersection_component'];cp,meta=candidates(G,frame,cam)
 with np.load(cp)as z:masks=z['masks']
 br=next(x for x in json.loads((V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'bindings.json').read_text())['records']if x['camera']==cam and x['ROI']==row['ROI']);allowed=np.zeros_like(part)
 for pr in br['components']:
  _,lab=cv2.connectedComponents(masks[pr['instance']].astype('uint8'),8);allowed|=lab[y0:y1,x0:x1]==pr['native_component']
 part&=cv2.distanceTransform((~allowed).astype('uint8'),cv2.DIST_L2,5)<=64
 pre=V/'operations/ordered_source_stage_v107'/f'frame{frame:06d}'/'applications'/cam/('old_'+cam+'.npz');pre=pre if pre.exists()else V/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz')
 with np.load(pre)as z:A=(z['black']|z['grey'])[y0:y1,x0:x1]
 R=part&~A;assert int(R.sum())==q['new_actor_union_pixels']
 crew,proof=core.crew_core(frame,cam,masks);known=sorted({r['instance']for r in proof});rows=[]
 for i in known:
  m=masks[i,y0:y1,x0:x1];rows.append(dict(instance=i,new_residual_overlap=int((m&R).sum()),new_residual_fraction=int((m&R).sum())/max(int(R.sum()),1),semantic_component_overlap=int((m&part).sum()),semantic_component_fraction=int((m&part).sum())/max(int(part.sum()),1)))
 union=masks[known,y0:y1,x0:x1].any(0)if known else np.zeros_like(part)
 return dict(frame=frame,camera=cam,index=index,family=q['family'],residual_pixels=int(R.sum()),R_array_sha256=array_sha(R),full_head_qualified_CURRENT_CREW_RAW_union_residual_overlap=int((union&R).sum()),full_HEAD_CREW_RAW_union_residual_fraction=int((union&R).sum())/max(int(R.sum()),1),conservative_head_connected_crew_core_residual_overlap=int((crew[y0:y1,x0:x1]&R).sum()),known_full_CREW_instances=rows,current_CREW_head_proof=proof,current_native_used_for_CREW=str(G/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz')),current_native_sha256=sha(G/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz')),current_candidates_sha256=sha(cp),preSEM_native_sha256=sha(pre),warning_only_NOT_actor_or_nonactor_certificate=True)
if __name__=='__main__':
 cv2.setNumThreads(1);cases=[(1695,'GC',30),(1695,'MD',21),(1695,'MD',25),(1700,'GC',28),(1700,'ED',151),(1700,'NC',34),(1705,'GC',51),(1705,'FB',29),(1705,'FB',34),(1705,'GD',117),(1705,'DE',2)];rows=[]
 for c in cases:
  r=assess(*c);rows.append(r);print(c,r['full_HEAD_CREW_RAW_union_residual_fraction'],flush=True)
 write(V/'operations/semantic_actor_role_v120/full_CURRENT_CREW_RAW_goldens.json',dict(kind='PRIVATE_CURRENT_FULL_HEAD_QUALIFIED_CREW_RAW_WARNING_METRICS',records=rows,no_NN_or_source_modified=True,private_not_policy_authority=True,created=time.time()))
