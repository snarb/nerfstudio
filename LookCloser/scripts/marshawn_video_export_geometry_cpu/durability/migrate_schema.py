"""One-time metadata migration; keep images and old qualification receipts."""
from pathlib import Path
import fcntl,json,os,shutil,time
import cpu_exec

def migrate(params_root,archive_root,expected_adapter_sha):
 root=Path(params_root);archive=Path(archive_root);spec=root/'control_olive/spec.json'
 if not spec.exists():return dict(required=False,reason='Controls not staged; bridge will add schema key')
 current=cpu_exec.read(spec)
 if current.get('source_visibility_adapter_sha256')==expected_adapter_sha:return dict(required=False,reason='Schema already current')
 if 'source_visibility_adapter_sha256'in current:raise ValueError('Unexpected existing source-visibility adapter')
 if cpu_exec.sha(root/'one.py')!=expected_adapter_sha:raise ValueError('Frozen adapter changed')
 # No live whole-video postprocessor is permitted during control migration.
 probe=(root/'postprocess.lock').open('a+')
 fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
 archive.mkdir(parents=True,exist_ok=True);oldsha=cpu_exec.sha(spec);oldspec=archive/'old_control_spec.json'
 if oldspec.exists():
  if cpu_exec.sha(oldspec)!=oldsha:raise ValueError('Different previous schema migration')
 else:shutil.copy2(spec,oldspec)
 output=Path(current['output']);receipts=list((output/'frames').glob('*/complete.json'))
 for receipt in receipts:
  record=cpu_exec.read(receipt)
  if record['binding']['spec_sha256']!=oldsha:continue
  # Only the already-qualified early first frame exists before all72 finish.
  if receipt.parent.name!='0000':raise ValueError('Refuse a schema change after ordinary export has begun')
  target=archive/'frame0000';target.mkdir(exist_ok=True)
  for name,value in record['images'].items():
   image=receipt.parent/name
   if cpu_exec.sha(image)!=value:raise ValueError('Old qualification PNG changed')
   saved=target/name
   if not saved.exists():os.link(image,saved)
   elif cpu_exec.sha(saved)!=value:raise ValueError('Previous archived PNG differs')
  saved=target/'complete_old_spec.json'
  if not saved.exists():shutil.copy2(receipt,saved)
  elif cpu_exec.sha(saved)!=cpu_exec.sha(receipt):raise ValueError('Previous qualification receipt differs')
  # Images retain their public paths. Only the old completion marker moves.
  receipt.rename(target/'original_complete_marker.json')
 updated=dict(current,source_visibility_adapter_sha256=expected_adapter_sha);cpu_exec.atomic(spec,updated);result=dict(kind='RENDER_CONTROL_METADATA_ONLY_SCHEMA_MIGRATION_V1',required=True,old_spec_sha256=oldsha,new_spec_sha256=cpu_exec.sha(spec),archive=str(archive),PNGs_preserved=True,old_qualification_receipt_preserved=True,models_DATA_shaders_unchanged=True,time=time.time());cpu_exec.atomic(archive/'complete.json',result);return result
