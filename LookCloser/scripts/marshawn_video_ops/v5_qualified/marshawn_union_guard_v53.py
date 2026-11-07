"""Fail closed if a current NN actor union loses foreground at finalization."""
from pathlib import Path
import json,numpy as np
from prepare_marshawn_tuning import sha
from marshawn_union_resolution_v5 import array_sha

def validate_union(root,frame,camera):
 f=Path(root)/'frames'/f'{frame:06d}';native=f/'native_masks'/f'{camera}.npz';record=json.loads(native.with_suffix('.json').read_text());r=record.get('union_resolution')
 if not r or r.get('version')!=5:raise ValueError('Missing source-bound union conservation receipt '+camera)
 p=Path(r['receipt']);p=p if p.is_absolute()else f/p
 if sha(p)!=r['receipt_sha256']:raise ValueError('Union proof changed '+camera)
 proof=json.loads(p.read_text());assert proof['frame']==frame and proof['camera']==camera
 if proof['source_receipt_sha256']!=sha(Path(root)/'sources'/str(frame)/camera/'receipt.json'):raise ValueError('Foreign union source frame '+camera)
 if proof['new_native_sha256']!=sha(native)or record['archive_sha256']!=sha(native):raise ValueError('Union native archive changed '+camera)
 with np.load(native)as z:
  b=z['black'];g=z['grey'];v=z['source_ineligible'];u=z['actor_ownership_uncertain'];actual=(b|g)&~v
  if(b&g).any():raise ValueError('Actor labels are not disjoint '+camera)
  if(u&~actual).any():raise ValueError('Uncertain ownership lost positive actor support '+camera)
  if array_sha(u)!=proof['ownership_uncertain_sha256']:raise ValueError('Ownership uncertainty changed '+camera)
  if proof.get('qualified_current_actor_additions'):
   qualified=np.zeros_like(actual)
   for addition in proof['qualified_current_actor_additions']:
    ap=f/addition['receipt']
    if sha(ap)!=addition['receipt_sha256']:raise ValueError('Qualified current actor source proof changed')
    hp=json.loads(ap.read_text());assert hp['frame']==frame and hp['camera']==camera and hp['source_receipt_sha256']==proof['source_receipt_sha256']
    native_hair=f/'qualified_hair_inputs'/camera/'qualified_hair_native.npz'
    if sha(native_hair)!=hp['native_hair_archive_sha256']:raise ValueError('Qualified native hair changed')
    with np.load(native_hair)as hz:qualified|=hz['qualified_actor_rgb']
   if 'qualified_actor_rgb'not in z.files or not np.array_equal(z['qualified_actor_rgb'],qualified):raise ValueError('Qualified actor RGB scope changed')
   if array_sha(qualified)!=proof['qualified_actor_rgb_sha256']:raise ValueError('Qualified actor mask hash differs')
   prepath=Path(proof['pre_overlap_archive'])
   if sha(prepath)!=proof['pre_overlap_archive_sha256']:raise ValueError('Original current NN PRE archive changed')
   with np.load(prepath)as pz:pb=pz['pre_black']if'pre_black'in pz.files else pz['black'];pg=pz['pre_grey']if'pre_grey'in pz.files else pz['grey'];expected=(pb|pg|qualified)&~v
   if not np.array_equal(actual,expected):raise ValueError('Actor union differs from current NN base plus qualified hair')
  if array_sha(actual)!=proof['eligible_pre_union_sha256']or proof['eligible_pre_union_sha256']!=proof['eligible_post_union_sha256']:raise ValueError('Current actor foreground was erased '+camera)
 return proof
