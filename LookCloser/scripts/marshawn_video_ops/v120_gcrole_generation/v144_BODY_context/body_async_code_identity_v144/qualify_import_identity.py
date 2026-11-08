from pathlib import Path
import sys,json,types,time
O=Path(__file__).parent;G=O.parent.parent;sys.path.insert(0,str(O))
import marshawn_native_async_collect as corrected
config=corrected.read(G/'operations/sam3_assist/v5/frame001715/all66_config.json');assert config['campaign']==str(G)
base=corrected.base_module(config);expected=(G/'operations/sam3_assist/code/marshawn_sam3_union_batch_v62_runtime.py').resolve();assert Path(base.__file__).resolve()==expected;base.guard(config)
name='marshawn_sam3_union_batch_v62_runtime';saved=sys.modules[name];fake=types.ModuleType(name);fake.__file__='/tmp/unqualified_foreign_gather.py';sys.modules[name]=fake
try:
 try:corrected.base_module(config)
 except ValueError:negative=True
 else:raise AssertionError('Foreign import accepted')
finally:sys.modules[name]=saved
corrected.write(O/'qualification.json',dict(status='pass',actual_G1_configuration_sha256=corrected.sha(G/'operations/sam3_assist/v5/frame001715/all66_config.json'),config_campaign=str(G),actual_base_module_physical=str(expected),actual_base_module_sha256=corrected.sha(expected),original_all_task_and_code_guard_pass=True,foreign_module_import_still_rejected=negative,config_and_132_tasks_unchanged=True,created=time.time()));print(O/'qualification.json')
