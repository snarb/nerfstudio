from pathlib import Path
import sys,json,time,numpy as np
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations/sam3_assist/code'),str(V/'code/LookCloser/scripts')]
import marshawn_sam3_assist as io
import marshawn_sam3_union_capture_v5 as capture
R=G/'operations/sam3_assist/v5/frame001715';rows=[];module=io.canonical(G/'code')
for cam in io.read(G/'source_inventory.json')['cameras']:
 native=G/'frames/001710/native_masks'/(cam+'.npz');expected=io.sha(native);hosts=[]
 with np.load(native)as z:
  U=z['actor_ownership_uncertain'];reference={a:module.small(z[a]&~U)for a in ['black','grey']}
 for folder,host in [('tasks','silver'),('tasks_local','olive')]:
  p=R/folder/cam/'task.json';j=io.read(p);policy=p.with_name('seed_policy_v5.json');q=io.read(policy);assert j['reference']['frame']==1710 and j['reference']['archive_sha256']==expected and q['reference_archive_sha256']==expected
  with np.load(p.with_name('reference.npz'))as z:
   for a in ['black','grey']:assert np.array_equal(z[a],reference[a])and io.array_sha(z[a])==j['reference']['arrays_sha256'][a]and capture.array_sha(z[a])==q['output_arrays_sha256'][a]
  hosts.append(dict(host=host,task=str(p),task_sha256=io.sha(p),seed_policy_sha256=io.sha(policy),reference_arrays_sha256=j['reference']['arrays_sha256'],actual_numeric_reference_equals_clean_G1_native_small_actor_minus_U=True))
 assert hosts[0]['reference_arrays_sha256']==hosts[1]['reference_arrays_sha256'];rows.append(dict(camera=cam,actual_prior_native=str(native),actual_prior_native_sha256=expected,hosts=hosts));print(cam,flush=True)
O=G/'operations/body_g1_v140';p=O/'actual_1715_all132_clean_G1_references.json';io.write(p,dict(status='pass',frame=1715,prior=1710,records=rows,all66_clean_G1_reference_archives=True,all132_actual_numeric_reference_arrays_match_clean_G1=True,hash_functions_distinguished='task includes dtype/shape; seed policy hashes raw bytes',no_NN_config_or_execution_yet=not(R/'all66_config.json').exists(),created=time.time()));print('PASS',p,flush=True)
