from pathlib import Path
import os,json,shutil,fcntl,time,hashlib
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(q,indent=2)+'\n');os.replace(t,p)
def link(p,target):
 if p.exists()or p.is_symlink():assert p.is_symlink()and p.resolve()==target.resolve();return
 p.symlink_to(target,target_is_directory=target.is_dir())
def stage():
 G.mkdir(exist_ok=True);(G/'operations').mkdir(exist_ok=True);(G/'frames').mkdir(exist_ok=True)
 lock=open(G/'operations/generation_initialization.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
 for name in ['code','sources']:link(G/name,V/name)
 for name in ['source_inventory.json','common_box.json','prep_release.json']:shutil.copy2(V/name,G/name)
 for frame in range(1660,1695,5):link(G/'frames'/f'{frame:06d}',V/'frames'/f'{frame:06d}')
 for frame in[1695,1700,1705]:
  f=G/'frames'/f'{frame:06d}';origin=V/'frames'/f'{frame:06d}';receipt=f/'generation_origin.json'
  if receipt.exists():continue
  f.mkdir(exist_ok=True)
  for p in origin.iterdir():
   if p.name in ['data','native_review','native_semantic_ready.json','source_mask_binding.json','ordered_current_semantic_component_stage.json']:continue
   if p.is_dir():shutil.copytree(p,f/p.name,symlinks=True,dirs_exist_ok=True)
   else:shutil.copy2(p,f/p.name)
  data=f/'data';data.mkdir(exist_ok=True)
  for name in ['images','lookcloser_frequencies','frequency_specs']:
   if(origin/'data'/name).exists():link(data/name,origin/'data'/name)
  archived=f/'origin_g0_controls';archived.mkdir(exist_ok=True)
  for p in[origin/'native_semantic_ready.json',origin/'source_mask_binding.json',origin/'ordered_current_semantic_component_stage.json']+list((origin/'data').glob('*.json')):
   if p.exists():shutil.copy2(p,archived/(('data_'if p.parent.name=='data'else'')+p.name))
  paths=list((origin/'native_masks').glob('*'))+[origin/'native_semantic_ready.json',origin/'data/ready.json',origin/'data/prepared_ready.json']
  write(receipt,dict(kind='READ_ONLY_ORIGINAL_SOURCE_GENERATION_ADOPTION_V1',generation='gcrole_g1',frame=frame,origin_root=str(V),new_root=str(G),original_published_source_sha256=sha(origin/'native_semantic_ready.json'),original_files_sha256={str(p.relative_to(origin)):sha(p)for p in paths},RAW_and_original_NN_read_only=True,all_new_DATA_and_mutable_source_proofs_generation_owned=True,RGB_frequency_shared_only_same_frame_exact_pixels=True,published_controls_not_adopted_as_new_ready=True,created=time.time()))
 write(G/'operations/generation_staging_ready.json',dict(kind='SOURCE_GCROLE_G1_STAGING_ONLY',status='staged_not_corrected_or_published',frames=[1695,1700,1705],origin=str(V),no_old_published_source_modified=True,created=time.time()))
 print('STAGED',G,flush=True)
if __name__=='__main__':stage()
