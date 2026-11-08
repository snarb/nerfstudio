"""Source-verified V5 native installation; never publishes DATA or field jobs."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_union_guard_v5 import validate_union

def atomcopy(source,destination):
 destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix(destination.suffix+'.installtmp');shutil.copy2(source,temp)
 if sha(temp)!=sha(source):raise ValueError('Input copy changed')
 temp.replace(destination)
def install(root,frame,marker,release):
 authorization=json.loads(release.read_text())
 if not authorization.get('authorized')or(frame not in authorization.get('frames',[])and not authorization.get('future_frames_authorized')):raise ValueError('V5 input installation is not released')
 entry=json.loads(marker.read_text());ready=Path(entry['all66_review_ready'])
 if sha(ready)!=entry['all66_review_ready_sha256']:raise ValueError('Isolated source receipt changed')
 summary=json.loads(ready.read_text());iso=Path(entry['isolated_campaign'])/'frames'/f'{frame:06d}';live=root/'frames'/f'{frame:06d}';cameras=json.loads((root/'source_inventory.json').read_text())['cameras'];assert set(summary['cameras'])==set(cameras)and len(cameras)==66
 if(live/'dispatch_ready.json').exists():raise ValueError('Never edit already dispatched inputs')
 backup=live/'native_masks_before_union_v5';backup.mkdir(exist_ok=True);records=[]
 # Preflight ALL sources before changing the first live file.
 for camera in cameras:
  item=summary['cameras'][camera];source=iso/'native_masks'/f'{camera}.npz';metadata=source.with_suffix('.json');m=json.loads(metadata.read_text());assert item['canonical_post_clear_replay_exact']and item['union_conserved']
  if sha(source)!=item['resolved_archive_sha256']or m['archive_sha256']!=sha(source):raise ValueError('Resolved native changed '+camera)
  if item['source_receipt_sha256']!=sha(root/'sources'/str(frame)/camera/'receipt.json')or m['source_receipt_sha256']!=item['source_receipt_sha256']:raise ValueError('Foreign source frame '+camera)
  current=live/'native_masks'/source.name
  if current.exists():
   if sha(current)not in[item['original_native_sha256'],item['resolved_archive_sha256']]:raise ValueError('Unreviewed live native change '+camera)
   with np.load(current)as a,np.load(source)as b:
    for k in ['source_ineligible','unknown']:assert np.array_equal(a[k],b[k]),(camera,k)
    for k in ['black','grey']:assert not(a[k]&~b[k]).any(),(camera,k)
  records.append(dict(camera=camera,old=item['original_native_sha256'],new=item['resolved_archive_sha256'],source_receipt_sha256=item['source_receipt_sha256'],restored_pixels=item['restored_pixels']))
 for row in records:
  camera=row['camera'];source=iso/'native_masks'/f'{camera}.npz';metadata=source.with_suffix('.json');current=live/'native_masks'/source.name
  if current.exists():
   if not(backup/current.name).exists():
    atomcopy(current,backup/current.name);atomcopy(current.with_suffix('.json'),backup/current.with_suffix('.json').name)
   if sha(backup/current.name)!=row['old']:raise ValueError('Original canonical backup changed '+camera)
  destination=live/'union_resolution_inputs'/camera
  if destination.exists()and not(live/'union_resolution_inputs_before_all66_v5'/camera).exists():shutil.copytree(destination,live/'union_resolution_inputs_before_all66_v5'/camera)
  shutil.copytree(iso/'union_resolution_inputs'/camera,destination,dirs_exist_ok=True)
  for folder in ['pre_clear','person_candidates']:
   for item in(iso/folder).glob(camera+'*'):
    if item.is_file():atomcopy(item,live/folder/item.name)
    elif item.is_dir():shutil.copytree(item,live/folder/item.name,dirs_exist_ok=True)
  atomcopy(source,current);m=json.loads(metadata.read_text());m['live_install']=dict(input_entry_sha256=sha(marker),source_metadata_sha256=sha(metadata),installation_release_sha256=sha(release),old_canonical_native_sha256=row['old'],source_visual_review_pending=True);write(current.with_suffix('.json'),m)
  validate_union(root,frame,camera)
 receipt=dict(frame=frame,cameras=66,entry_sha256=sha(marker),installation_release_sha256=sha(release),all_union_proofs_validated=True,all_old_actor_pixels_preserved=True,all_veto_and_UNKNOWN_arrays_unchanged=True,no_RGB_or_frequency_changes=True,records=records,total_restored_native_actor_pixels=sum(r['restored_pixels']for r in records),source_visual_review_pending=True,prepared_publication_allowed=False,created=time.time());write(live/'union_v5_installed.json',receipt);return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--marker',type=Path,required=True);p.add_argument('--release',type=Path,required=True);a=p.parse_args();print(json.dumps(install(a.root,a.frame,a.marker,a.release)))
