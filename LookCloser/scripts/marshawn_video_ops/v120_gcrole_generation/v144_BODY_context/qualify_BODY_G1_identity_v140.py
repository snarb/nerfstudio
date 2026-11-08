"""Private CPU-only pack identity canary; no NN or live task writes."""
from pathlib import Path
import sys,json,os,time
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'operations/sam3_assist/code'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from native_pack_cpu8_v2.marshawn_native_pack_cpu8 import pack_records
import marshawn_sam3_assist_v5 as assist

def main():
 O=G/'operations/BODY_G1_identity_v140';O.mkdir(exist_ok=True);release=json.loads((V/'operations/sam3_assist/v5/native_future_release_v114.json').read_text());P=O/'parallel';D=O/'direct';cameras=['DE','JE'];records,execution=pack_records(G,1715,cameras,P,release,workers=2,campaign_identity=V,logical_root=G/'operations/sam3_assist/v5/frame001715');rows=[]
 for camera in cameras:
  for host,folder in [('silver','tasks'),('olive','tasks_local')]:
   task=D/folder/camera;remote=Path(release['remote_root'])/'operations/sam3_assist/v5/frame001715/tasks'/camera if host=='silver'else G/'operations/sam3_assist/v5/frame001715'/folder/camera
   assist.pack(G,1715,camera,task,str(remote),str(Path(release['remote_root'])/'code'if host=='silver'else V/'code'),str(Path(release['remote_root'])/'tools/sam3_overlay')if host=='silver'else release['local_overlay'],V/'operations/marshawn_union_resolution_v5.py')
   parallel=P/folder/camera;files=[]
   for p in task.iterdir():
    q=parallel/p.name
    if p.is_symlink():assert q.is_symlink()and p.readlink()==q.readlink()
    elif p.is_file():assert sha(p)==sha(q),(camera,host,p.name)
    else:continue
    files.append(p.name)
   s=json.loads((parallel/'task.json').read_text());policy=json.loads((parallel/'seed_policy_v5.json').read_text());actual=G/'frames/001710/native_masks'/(camera+'.npz');assert s['reference']['frame']==1710 and s['reference']['archive_sha256']==sha(actual)and policy['reference_archive_sha256']==sha(actual)
   rows.append(dict(camera=camera,host=host,task=str(parallel/'task.json'),task_sha256=sha(parallel/'task.json'),prior_native=str(actual),prior_native_sha256=sha(actual),actual_clean_G1_reference=True,exact_original_direct_pack_file_and_symlink_parity=True,compared_files=files,seed_policy=str(parallel/'seed_policy_v5.json'),seed_policy_sha256=sha(parallel/'seed_policy_v5.json')))
 write(O/'PASS.json',dict(status='pass',frame=1715,campaign_data_root=str(G),immutable_implementation_identity=str(V),records=rows,execution=execution,no_live_tasks_or_GPU_NN_or_source_metadata_modified=True,created=time.time()));print(O/'PASS.json',flush=True)
if __name__=='__main__':main()
