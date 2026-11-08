"""Read-only six-package audit against source-extracted eventual final controls."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,shlex,subprocess,sys,time,types
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');F=Path('/fsx/tmp/marshawn_fleet10_140k_20261107');O=Path(__file__).parent;P=C/'operations/video_geometry_cpu_params_v2';D=C/'operations/video_geometry_cpu_durable_v3'
sys.path.insert(0,str(D));from bridge_v3 import Bridge
from migrate_schema import migrate

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serialize(v):return(json.dumps(v,indent=2)+'\n').encode()
def immutable(p,v):
 p=Path(p);b=serialize(v)
 if p.exists():assert p.read_bytes()==b,'Refuse changing audit snapshot';return
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def call(h,args):return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h['address'],shlex.join(list(map(str,args)))]if h.get('address')else list(map(str,args)),check=True,capture_output=True,text=True,timeout=30)

plan=read(F/'video_plan.json');export=read(V/'video/export_pipeline.json');request=read(export['path']);request['source_transforms']='source_transforms.json';path_sha=hashlib.sha256(serialize(request)).hexdigest();assert path_sha=='35953efcc54ce0464bc45091c0a999936bd1a12c65c6a55b735c313d972d60be';immutable(O/'prospective_path.json',request)
assert sha(C/'frames/001660/data/transforms.json')==request['source_transforms_sha256']
router=module(P/'route.py','source_only_final_router');tree=ast.parse((P/'route.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');loop=next(n for n in main.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and [x.id for x in n.target.elts]==['name','h'])
assign=next(n for n in loop.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='spec'for t in n.targets));silver=next(n for n in loop.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and isinstance(n.test.left,ast.Name)and n.test.left.id=='name');selected=ast.fix_missing_locations(ast.Module(body=[assign,silver],type_ignores=[]));code=compile(selected,str(P/'route.py'),'exec')
base=read(export['spec']);controls={};capture={};dummy=types.SimpleNamespace(__file__=str(P/'route.py'),read=read,write=lambda p,v:capture.update({str(p):copy.deepcopy(v)}),identity=lambda p:None,shutil=types.SimpleNamespace(copy2=lambda *x:None));bridge=Bridge(dummy,O/'unused_ledger',D/'cpu_exec.py',shader_sha=sha(P/'worker_params.py'))
for name,h in plan['hosts'].items():
 root=Path(h['root']);local=P/('control_'+name);env=dict(router.__dict__,base=base,out=C/'video/geometry_cpu_delivery_params_v2',gather=C/'video/geometry_cpu_float_v1',root=root,name=name);exec(code,env);spec=env['spec'];bridge.write_state(local/'spec.json',spec)
 # The real Bridge protected-write path proves equality if controls already exist.
 final=copy.deepcopy(spec);final['source_visibility_adapter_sha256']=sha(P/'one.py')
 if(local/'spec.json').exists():assert(local/'spec.json').read_bytes()==serialize(final)
 else:assert capture[str(local/'spec.json')]==final
 immutable(O/'prospective_controls'/name/'spec.json',final);controls[name]=dict(spec=final,spec_sha256=hashlib.sha256(serialize(final)).hexdigest(),existing_equal=(local/'spec.json').exists())
# Extract ONLY the pure historical candidate-order function, avoiding CUDA imports.
legacy=Path(base['legacy_hybrid_root'])/'campaign/misha50_hybrid.py';lt=ast.parse(legacy.read_text());fn=next(n for n in lt.body if isinstance(n,ast.FunctionDef)and n.name=='candidates');ns=dict(np=np);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(legacy),'exec'),ns)
profile=read(P/'hybrid_params.json');frameproof=[]
for i,record in enumerate(request['records'][:6]):
 frame=record['frame_id'];data=C/'frames'/frame/'data';gpath=C/'video/geometry_cpu_float_v1/frames'/f'{i:04d}';receipt=read(gpath/'geometry_complete.json');binding=receipt['binding'];cp=receipt['checkpoint_binding'];assignment_path=Path(plan['assignments'])/(frame+'.json');a=read(assignment_path);h=plan['hosts'][a['host']];assert a['host_root']==h['root']
 for field,expected in dict(path_sha256=path_sha,record=record,model_frame_id=frame,data_ready_sha256=sha(data/'ready.json'),transforms_sha256=sha(data/'transforms.json'),worker_sha256=sha(P/'worker.py'),code_seal_sha256=sha(C/'code_seal.json')).items():assert binding[field]==expected,(i,field)
 assert receipt['kind']=='NATIVE_MODEL_BOUND_FLOAT_GEOMETRY_V1'and cp['active_hard140k_revision']and 0<cp['completed_iterations']<=140000 and cp['production_campaign']==a['host_root'];native=Path(a['host_root'])/'frames'/frame;completion_bytes=call(h,['cat',str(native/'frame_completion.json')]).stdout.encode();completion=json.loads(completion_bytes);assert hashlib.sha256(completion_bytes).hexdigest()==binding['frame_completion_sha256'];selector=completion['selectors']['actor_lpips'];assert selector['checkpoint_sha256']==cp['checkpoint_sha256']and selector['step']==cp['step']
 # Every tensor/quantile file, row and semantic/validity input is source-bound.
 for n,v in receipt['files_sha256'].items():assert sha(gpath/n)==v,(i,n)
 target=read(gpath/'target.json');assert target['binding']['row']==record['camera']and target['binding']['checkpoint_sha256']==cp['checkpoint_sha256']and target['sha256']==sha(gpath/'target.npz')
 tf=read(data/'transforms.json');rows={r['physical_camera']:dict(r,camera_id=k)for k,r in enumerate(tf['frames'])};allowed={r['physical_camera']for r in tf['frames']if r['file_path']in tf['train_filenames']};sources=types.SimpleNamespace(rows=rows,allowed=allowed);expected_names=ns['candidates'](sources,record['camera'],16);assert expected_names==receipt['source_names'];assert len(expected_names)==16
 for name in expected_names:
  source=receipt['source_rows'][name];assert source['row']==rows[name];assert sha(data/'semantics'/(name+'.npz'))==source['semantic_sha256']and sha(data/rows[name]['mask_path'])==source['valid_mask_sha256'];meta=read(gpath/'sources'/(name+'.json'));assert meta['binding']['row']==rows[name]and meta['binding']['checkpoint_sha256']==cp['checkpoint_sha256']and meta['sha256']==sha(gpath/'sources'/(name+'.npz'))
 for name,anchor in profile['pixel_scale']['reference_calibration'].items():assert all(abs(rows[name][k]-v)<1e-10 for k,v in anchor.items())
 assert receipt['pipeline_tensor_SHA_before']==receipt['pipeline_tensor_SHA_after']and receipt['native_HC_parity']['max_RGB_byte_difference']<=1
 frameproof.append(dict(index=i,frame=frame,host=a['host'],checkpoint_sha256=cp['checkpoint_sha256'],selected_step=cp['step'],completed_iterations=cp['completed_iterations'],geometry_receipt_sha256=sha(gpath/'geometry_complete.json'),assignment_sha256=sha(assignment_path),native_frame_completion_sha256=binding['frame_completion_sha256'],donor_count=16,rows_masks_files_and_focal_calibration_verified=True))
# Actual schema migration already happened: the old marker is intentionally archived.
archive=D/'preserved_pre_GT_schema_qualification';oldspec=read(archive/'old_control_spec.json');oldmarker=archive/'frame0000/original_complete_marker.json';old=read(oldmarker);current=controls['olive']['spec'];assert old['binding']['spec_sha256']==sha(archive/'old_control_spec.json');assert old['binding']['path_sha256']==path_sha;delta={k:dict(old=oldspec.get(k),current=current.get(k))for k in set(oldspec)|set(current)if oldspec.get(k)!=current.get(k)};assert set(delta)=={'source_visibility_adapter_sha256'}
canonical=C/'video/geometry_cpu_delivery_params_v2/frames/0000';assert not(canonical/'complete.json').exists()
for name,v in old['images'].items():assert sha(archive/'frame0000'/name)==v and sha(canonical/name)==v
before={str(p):sha(p)for p in [P/'control_olive/spec.json',oldmarker,*[canonical/n for n in old['images']]]};migration=migrate(P,O/'unused_migration_archive',sha(P/'one.py'));assert migration['required']is False and not(O/'unused_migration_archive').exists();assert before=={str(p):sha(p)for p in map(Path,before)}
result=dict(kind='READ_ONLY_EXACT_FINAL_CONTROL_AND_FIRST6_GEOMETRY_BINDING_AUDIT',passed=True,source_P_route_sha256=sha(P/'route.py'),source_Bridge_sha256=sha(D/'bridge_v3.py'),full_path_sha256=path_sha,prospective_spec_SHA256={h:r['spec_sha256']for h,r in controls.items()},existing_olive_control_matches_exact_eventual_final_spec=True,all6_canonical_geometry_compatible=frameproof,canonical_index0=dict(active_completion_marker_present=False,old_completion_archived=str(oldmarker),old_spec_sha256=old['binding']['spec_sha256'],eventual_spec_sha256=controls['olive']['spec_sha256'],only_schema_delta=delta,actual_migrate_schema_now_noop=migration,old_PNGs_and_archive_SHA_verified=True,expected_action='Re-render index0 with unchanged released CPUshader under new spec; never adopt/restamp old spec marker',old_archival_hardlinks_safe='Frozen save writes a temporary PNG then atomicreplace, preserving archived inode'),GPU_or_CPU_render_launched=False,canonical_files_mutated=False,rolling_coordinator_implemented=False,time=time.time());immutable(O/'report.json',result);print(json.dumps(dict(passed=True,report=str(O/'report.json'),sha256=sha(O/'report.json'),specs=result['prospective_spec_SHA256'],index0=result['canonical_index0'])))
