"""Actual frozen CLI tail rendering: no frame2020 model or neural geometry."""
from pathlib import Path
import sys,json,shutil,subprocess,os,time
import numpy as np
from PIL import Image
O=Path(__file__).parent;sys.path.insert(0,str(O));import cpu_exec
from bridge_v3 import Bridge
C=O.parents[1];P=C/'operations/video_geometry_cpu_params_v2';control=O/'qualification_metadata_v3/control';control.mkdir(parents=True,exist_ok=True);previous=P/'qualification_first0/control'
for n in ['path.json','cpu_spec.json','source_transforms.json']:
 destination=control/('spec.json'if n=='cpu_spec.json'else n)
 if n!='cpu_spec.json':
  if destination.exists():assert cpu_exec.sha(previous/n)==cpu_exec.sha(destination)
  else:shutil.copy2(previous/n,destination)
for n in ['worker.py','worker_params.py','one.py','cap_loader.py','render_gate_loader.py','focal_scale.py','hybrid_params.json','root_release.json']:
 destination=control/n
 if destination.exists():assert cpu_exec.sha(P/n)==cpu_exec.sha(destination)
 else:shutil.copy2(P/n,destination)
spec=cpu_exec.read(previous/'cpu_spec.json');spec['output']=str(O/'qualification_metadata_v3/output');spec['source_visibility_adapter_sha256']=cpu_exec.sha(P/'one.py');cpu_exec.atomic(control/'spec.json',spec)
assert not(C/'frames/002020/data').exists()
request=cpu_exec.read(control/'path.json');spec=cpu_exec.read(control/'spec.json');results=[]
# One actual neural roundtrip with the added metadata key, no new GPU work.
neural=[sys.executable,str(control/'worker_params.py'),'cpu','--code',str(C/'code'),'--spec',str(control/'spec.json'),'--path',str(control/'path.json'),'--index','0']
with(O/'actual_neural_metadata_CLI.log').open('ab')as log:
 result=subprocess.run(neural,cwd=control,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0
reference=C/'operations/video_fleet10/private_profile_v1/matched_first3/frames/0000/candidate';neural_proof={}
for name,gold in [('lookcloser.png','lookcloser_4K.png'),('hybrid.png','hybrid_4K.png')]:
 a=np.asarray(Image.open(Path(spec['output'])/'frames/0000'/name));b=np.asarray(Image.open(reference/gold));maximum=int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max());assert maximum==0;neural_proof[name]=maximum
for index in [72,119]:
 assert request['records'][index]['kind']=='actual_gt'and request['records'][index]['frame_id']=='002020'
 assert not(Path(spec['gathered_geometry'])/'frames'/f'{index:04d}'/'geometry_complete.json').exists()
 command=[sys.executable,str(control/'one.py'),'--cpu','--code',str(C/'code'),'--spec',str(control/'spec.json'),'--path',str(control/'path.json'),'--index',str(index)]
 # These commands intentionally bypass the neural-only exec bridge.
 bridge=Bridge(type('R',(),dict(__file__=str(P/'route.py'),read=staticmethod(cpu_exec.read),write=staticmethod(cpu_exec.atomic),identity=staticmethod(cpu_exec.identity),shutil=shutil)),O/'GT_test_unused_ledger',O/'cpu_exec.py',shader_sha=cpu_exec.sha(P/'worker_params.py'))
 assert bridge.cpu_command(command)is False
 wrong=[sys.executable,str(control/'worker_params.py'),'cpu','--code',str(C/'code'),'--spec',str(control/'spec.json'),'--path',str(control/'path.json'),'--index',str(index)]
 try:bridge.inputs(wrong)
 except ValueError as exc:assert'neural0..71'in str(exc)
 else:raise AssertionError('GT was accepted by neural executor')
 with(O/'actual_GT_CLI.log').open('ab')as log:result=subprocess.run(command,cwd=control,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,result.returncode
 out=Path(spec['output'])/'frames'/f'{index:04d}';receipt=cpu_exec.read(out/'complete.json');a=np.asarray(Image.open(out/'lookcloser.png'));b=np.asarray(Image.open(out/'hybrid.png'));assert np.array_equal(a,b)
 results.append(dict(index=index,record=request['records'][index],receipt_sha256=cpu_exec.sha(out/'complete.json'),lookcloser_sha256=cpu_exec.sha(out/'lookcloser.png'),hybrid_sha256=cpu_exec.sha(out/'hybrid.png'),raster=list(Image.open(out/'lookcloser.png').size),geometry_or_2020_model_required=False))
start=np.asarray(Image.open(Path(spec['output'])/'frames/0072/lookcloser.png'));end=np.asarray(Image.open(Path(spec['output'])/'frames/0119/lookcloser.png'));a=start[:1937,420:];b=end[87:,:3420];assert a.shape==b.shape;difference=int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max());assert difference==0
proof=dict(kind='ACTUAL_FROZEN_ONE_CLI_GT72_119_NO_NEURAL_DATA_QUALIFICATION_V1',passed=True,results=results,neural_metadata_roundtrip_max_RGB_code_difference=neural_proof,source_visibility_adapter_sha256=spec['source_visibility_adapter_sha256'],actual_common_native_pixel_overlap_max_code_difference=difference,cwd=str(control),actual_frozen_CPU_entrypoint=True,source_frame_frozen='002020',source_no_image_resize=True,CUDA_not_used=True,neural_bridge_rejects_GT_explicitly=True,no_model_or_training_source_changed=True,source_code_sha256=cpu_exec.sha(__file__),bridge_sha256=cpu_exec.sha(O/'bridge_v3.py'),finished=time.time());cpu_exec.atomic(O/'actual_GT72_119_neural_metadata_qualification_v3.json',proof);print(json.dumps({k:v for k,v in proof.items()if k!='results'}))
