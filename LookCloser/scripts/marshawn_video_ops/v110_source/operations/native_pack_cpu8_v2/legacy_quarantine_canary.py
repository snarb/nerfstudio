from pathlib import Path
import sys,json,shutil,hashlib
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations/native_pack_cpu8_v2';sys.path.insert(0,str(V/'operations'))
from native_pack_cpu8_v2.marshawn_native_pack_cpu8 import pack_records
if __name__=='__main__':
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); prior=json.loads((O/'canonical_crash_regression/stopped_after_base_manifest.json').read_text());source=Path(prior['path']).parent; root=O/'legacy_quarantine';task=root/'tasks/AA';shutil.copytree(source,task,symlinks=True);old=sha(task/'task.json');release=json.loads((V/'operations/sam3_assist/v5/native_future_release_v7.json').read_text());rows,execution=pack_records(V/'operations/native_pack_cpu8_v1/snapshot',1695,['AA'],root,release,workers=1,campaign_identity=V,logical_root=V/'operations/sam3_assist/v5/frame001695');archives=list((root/'.packing_interrupted/AA').iterdir());assert len(archives)==1 and sha(archives[0]/'task.json')==old
 equal={folder:sha(root/folder/'AA/task.json')==sha(V/'operations/sam3_assist/v5/frame001695'/folder/'AA/task.json')for folder in ['tasks','tasks_local']};assert all(equal.values());(O/'legacy_quarantine_canary.json').write_text(json.dumps(dict(passed=True,unsealed_legacy_bundle_preserved=True,both_host_task_bytes_equal=equal,helper_sha256=sha(O/'marshawn_native_pack_cpu8.py')),indent=2)+'\n')
