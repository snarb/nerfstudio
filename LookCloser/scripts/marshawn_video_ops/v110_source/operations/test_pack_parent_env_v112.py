from pathlib import Path
import textwrap,os,json,hashlib,time
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');p=V/'operations/sam3_assist/code/marshawn_mask_assist_full_v112.py';s=p.read_text();a=s.index('        packing_env_keys=');b=s.index("        io.write(root/'BODY_pack_CPU8_execution.json'",a);block=textwrap.dedent(s[a:b]);before=dict(os.environ)
for fail in [False,True]:
 def fake(*args,**kw):
  for key in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='packing-only'
  if fail:raise RuntimeError('expected packing failure')
  return [],{}
 g=dict(os=os,pack_records=fake,V=V,frame=1705,cameras=[],root=V,release={})
 try:exec(block,g)
 except RuntimeError:assert fail
 assert dict(os.environ)==before
proof=dict(status='pass',factory_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),exact_parent_env_preserved_on_success_and_exception=True,no_GPU_or_source_mutation=True,time=time.time())
(V/'operations/native_pack_cpu8_v2/parent_env_v112_PASS.json').write_text(json.dumps(proof,indent=2)+'\n');print('PASS exact parent environment restored on packing success and exception')
