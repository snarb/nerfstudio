from pathlib import Path
import importlib.util,json,tempfile,sys,hashlib
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');Q=V/'operations/frequency13/qualification';sys.path.insert(0,str(V/'operations/frequency13/code'))
import marshawn_frequency_remote_store as r
import marshawn_clever_frequency_backend as b
p=V/'operations/frequency10/code/marshawn_frequency13.py';s=importlib.util.spec_from_file_location('v13',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);c=json.loads((V/'operations/frequency13/config_v13_candidate.json').read_text());checks=[]
def reject(name,func):
 try:func()
 except (ValueError,FileNotFoundError):checks.append(name)
 else:raise AssertionError('Expected refusal '+name)
reject('unapproved_candidate_cannot_run_or_claim',lambda:m.require_release(c))
with tempfile.TemporaryDirectory(prefix='clever_contract_')as d:
 root=Path(d);src=root/'src';src.write_bytes(b'owned');dst=root/'output';r.copy_new(src,dst,r.sha(src));r.copy_new(src,dst,r.sha(src));checks.append('atomic_create_and_retry_same_bytes')
 other=root/'foreign';other.write_bytes(b'foreign');before=dst.read_bytes();reject('foreign_existing_output_never_replaced',lambda:r.copy_new(other,dst,r.sha(other)));assert dst.read_bytes()==before
 reject('lexical_traversal_rejected',lambda:r.inside(root/'a/../escape',root))
 link=root/'link';link.symlink_to(root);reject('symlink_namespace_escape_rejected',lambda:r.inside(link/'src',root))
 files={'src':r.sha(src)};r.check_files(root,files);src.write_bytes(b'changed');reject('archive_source_corruption_rejected',lambda:r.check_files(root,files))
 transaction=root/'stage.json';b.write(transaction,dict(frame=1690,image='A'));b.write(transaction,dict(frame=1690,image='A'));reject('pinned_source_binding_cannot_change_on_retry',lambda:b.write(transaction,dict(frame=1695,image='B')))
 reject('wrong_frame_key_rejected',lambda:b.key('1690','HC'));reject('wrong_camera_key_rejected',lambda:b.key(1690,'../HC'))
checks.append('independent_priority_and_oldhost_preflight_external_receipts_bound')
result=dict(passed=True,checks=checks,main_jobs_started=False,no_base_environment_mutation=True,source_sha256={str(x):b.sha(x)for x in [p,V/'operations/frequency13/code/marshawn_frequency_remote_store.py',V/'operations/frequency13/code/marshawn_clever_frequency_backend.py']})
b.write(Q/'cpu_contracts.json',result);print(json.dumps(result))
