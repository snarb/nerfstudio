"""Current native head uncertainty feeds admission only, never RGB confidence."""
from pathlib import Path
import json,shutil,time
import cv2,numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_actor_coordinates import raw_mask_coverage_to_HD

def build(root,frame,observations,policy_path):
 root=Path(root);f=root/'frames'/f'{frame:06d}';data=f/'data';out=data/'admission_unknown_hair';policy=json.loads(Path(policy_path).read_text());radius=int(policy['geometry_UNKNOWN_radius_native']);records=[]
 if(f/'native_semantic_ready.json').exists():raise ValueError('Never rewrite a published head geometry source layer')
 cameras=json.loads((root/'source_inventory.json').read_text())['cameras'];assert set(observations)==set(cameras)and len(cameras)==66
 out.mkdir(exist_ok=True)
 for camera in cameras:
  source_path=root/'sources'/str(frame)/camera/'receipt.json';source=json.loads(source_path.read_text());native=f/'native_masks'/(camera+'.npz');full=np.zeros((4320,8192),bool);proofs=[];cd=out/camera;cd.mkdir(exist_ok=True)
  for index,tile in enumerate(observations[camera]):
   proposal=Path(tile['proposal']);qpath=Path(tile['qualification']);q=json.loads(qpath.read_text())
   assert q['frame']==frame and q['camera']==camera and q['source_receipt_sha256']==sha(source_path)
   if sha(proposal)!=q['tile_proposal_sha256']:raise ValueError('Current native head proposal changed')
   with np.load(proposal)as z:h=z['geometry_hair_proposal'];box=z['crop_xyxy'].tolist()
   x0,y0,x1,y1=box;l=max(0,x0-radius);t=max(0,y0-radius);r=min(8192,x1+radius);b=min(4320,y1+radius);pad=np.zeros((b-t,r-l),bool);pad[y0-t:y1-t,x0-l:x1-l]=h;dist=cv2.distanceTransform((~pad).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);full[t:b,l:r]|=dist<=radius
   td=cd/f'tile{index:02d}';td.mkdir(exist_ok=True);shutil.copy2(proposal,td/'current_native_proposal.npz');shutil.copy2(qpath,td/'qualification_original.json');spec=Path(q['source_spec']);C=Path(q['candidates']);shutil.copy2(spec,td/'source_spec.json');shutil.copy2(C/'receipt.json',td/'NN_receipt.json')
   for selected in q.get('selected',[]):
    archive=Path(selected['archive']);assert sha(archive)==selected['archive_sha256'];shutil.copy2(archive,td/archive.name)
   if q.get('manual_current_selection'):
    archive=Path(q['manual_current_selection']['archive']);assert sha(archive)==q['manual_current_selection']['archive_sha256'];shutil.copy2(archive,td/archive.name)
   proofs.append(dict(directory=str(td.relative_to(out)),crop_xyxy=box,status=q['status'],source_spec_sha256=sha(spec),NN_receipt_sha256=sha(C/'receipt.json'),current_source_proposal_sha256=sha(proposal),native_overlap_edges=q.get('required_native_overlap_edges',[])))
  with np.load(native)as z:qualified=z['qualified_actor_rgb'].copy()if'qualified_actor_rgb'in z.files else np.zeros_like(full)
  # Qualified source hair already has positive evidence. Its uncertain outer
  # region, semantic disagreement, weak/occluded proposals remain UNKNOWN.
  full&=~qualified;HD=raw_mask_coverage_to_HD(full.astype('float32'),source['camera_row'])>1e-6;archive=out/(camera+'.npz');np.savez_compressed(archive,unknown_hair=HD);np.savez_compressed(cd/'unknown_hair_native.npz',unknown_hair_native=full)
  records.append(dict(camera=camera,archive=archive.name,archive_sha256=sha(archive),source_rgb_sha256=source['image_sha256'],source_native_image_sha256=source['raw_native_image_sha256'],source_receipt_sha256=sha(source_path),native_actor_archive_sha256=sha(native),radius_native=radius,applied=True,fresh_current_native_NN=True,current_native_tile_count=len(proofs),unknown_HD_pixels=int(HD.sum()),proofs=proofs,no_RGB_W64_original_UNKNOWN_or_source_UNKNOWN_changes=True))
 files={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='manifest.json'};manifest=dict(kind='CURRENT_NATIVE_HEAD_HAIR_ADMISSION_UNKNOWN_V1',frame=frame,cameras=66,records=records,files_sha256=files,policy_sha256=sha(policy_path),geometry_only=True,no_RGB_W64_original_UNKNOWN_or_source_UNKNOWN_changes=True,previous_head_used_only_as_ROI_proposal=True,current_frame_NN_masks_only=True,created=time.time());write(out/'manifest.json',manifest);return out/'manifest.json'

def bind(root,frame):
 f=Path(root)/'frames'/f'{frame:06d}';data=f/'data';p=data/'admission_unknown_hair/manifest.json';m=json.loads(p.read_text());ready=json.loads((data/'ready.json').read_text());assert m['frame']==frame and len(m['records'])==66
 for row in m['records']:
  current=next(r for r in ready['records']if r['camera']==row['camera']);assert current['rgb_sha256']==row['source_rgb_sha256']and sha(f/'native_masks'/(row['camera']+'.npz'))==row['native_actor_archive_sha256']
 for name,digest in m['files_sha256'].items():assert sha(p.parent/name)==digest
 prepared=json.loads((data/'prepared_ready.json').read_text());prepared['admission_unknown_hair_manifest_sha256']=sha(p);write(data/'prepared_ready.json',prepared);return sha(p)
