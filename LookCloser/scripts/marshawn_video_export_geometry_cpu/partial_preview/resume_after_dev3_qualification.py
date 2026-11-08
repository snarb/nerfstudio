from pathlib import Path
import importlib.util,json,sys,time
C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');O=C/'operations/video_partial_first6_v1';A=Path(__file__).parent;src=O/'partial.py';s=importlib.util.spec_from_file_location('qualified_partial',src);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
base=m.setup;adapter=A/'dev3_entry.py';source_sha=m.sha(adapter);original_request_sha=m.sha(O/'inputs.json');worker='6603aa06dc8ef2df2cd89cada087729eaa40bd81be5b5c981e2e3e2369539eba';newname='partial6_geometry_0004_UUID_a02'
def setup():
 if m.sha(adapter)!=source_sha or m.sha(O/'inputs.json')!=original_request_sha:raise ValueError('Partial adapter/request changed')
 inputs=base();job=inputs['jobs'][4];old=dict(job);job['name']=newname;cmd=job['command'];job['command']=[cmd[0],str(adapter),'--worker',cmd[1],'--expected-sha256',worker,'--',*cmd[2:]]
 m.immutable(A/'partial_retry_request_a02.json',dict(original_request_sha256=original_request_sha,adapter_sha256=source_sha,old_job=old,new_job=job,no_shader_or_model_change=True))
 state=m.read(O/'state.json')
 if newname not in state['jobs']:
  archive=A/'partial_state_before_UUID_retry_a02.json';m.immutable(archive,state);state['jobs'][newname]=dict(status='pending');m.write(O/'state.json',state)
 return inputs
m.setup=setup
try:m.main()
finally:
 m.write(Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native10_requests/partial_first6_released_R0.json'),dict(status='complete',host_gpus={'dev3':[0]},new_reservation_retired=True,running_workers_remain_under_actual_global_leases=True,time=time.time()))
