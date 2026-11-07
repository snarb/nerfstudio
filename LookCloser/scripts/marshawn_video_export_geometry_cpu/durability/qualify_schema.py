from pathlib import Path
import json,shutil,types,time,fcntl
import cpu_exec
from bridge_v3 import Bridge
from migrate_schema import migrate
O=Path(__file__).parent;Q=O/'qualification_schema_v3';Q.mkdir(exist_ok=True);P=O.parent/'video_geometry_cpu_params_v2';root=Q/'TEST_PARAMS';(root/'control_olive').mkdir(parents=True,exist_ok=True);shutil.copy2(P/'one.py',root/'one.py');adapter=cpu_exec.sha(root/'one.py');output=Q/'TEST_OUTPUT';f=output/'frames/0000';f.mkdir(parents=True,exist_ok=True)
spec=dict(output=str(output),qualification_kind='METADATA_ONLY_NOT_A_RENDER');cpu_exec.atomic(root/'control_olive/spec.json',spec);old_spec=cpu_exec.sha(root/'control_olive/spec.json');images={}
for name in ['lookcloser.png','hybrid.png']:(f/name).write_bytes(b'TEST_ONLY_NOT_AN_IMAGE');images[name]=cpu_exec.sha(f/name)
cpu_exec.atomic(f/'complete.json',dict(kind='TEST_METADATA_ONLY_NOT_A_RENDER',binding=dict(spec_sha256=old_spec),images=images));old_receipt=cpu_exec.sha(f/'complete.json');archive=Q/'preserved';result=migrate(root,archive,adapter);assert result['required'];assert cpu_exec.sha(archive/'old_control_spec.json')==old_spec;assert cpu_exec.sha(archive/'frame0000/complete_old_spec.json')==old_receipt;assert not(f/'complete.json').exists()
for name,value in images.items():assert cpu_exec.sha(f/name)==value and cpu_exec.sha(archive/'frame0000'/name)==value
assert cpu_exec.read(root/'control_olive/spec.json')['source_visibility_adapter_sha256']==adapter;assert migrate(root,archive,adapter)['required']is False
r=types.SimpleNamespace(__file__=str(root/'route.py'),read=cpu_exec.read,write=cpu_exec.atomic,identity=cpu_exec.identity,shutil=shutil);b=Bridge(r,Q/'unusedledger',O/'cpu_exec.py',shader_sha='unused');mtime=(root/'control_olive/spec.json').stat().st_mtime_ns;b.write_state(root/'control_olive/spec.json',spec);assert(root/'control_olive/spec.json').stat().st_mtime_ns==mtime
proof=dict(kind='GT_SPEC_METADATA_ADAPTER_AND_RECEIPT_PRESERVATION_QUALIFICATION_V1',passed=True,adapter_sha256=adapter,old_spec_and_receipt_byte_preserved=True,PNG_bytes_and_public_paths_preserved=True,migration_idempotent=True,identical_controls_not_rewritten=True,no_model_or_GPU_touched=True,source_sha256=cpu_exec.sha(__file__),bridge_sha256=cpu_exec.sha(O/'bridge_v3.py'),migration_sha256=cpu_exec.sha(O/'migrate_schema.py'),finished=time.time());cpu_exec.atomic(Q/'complete.json',proof);print(json.dumps(proof))
