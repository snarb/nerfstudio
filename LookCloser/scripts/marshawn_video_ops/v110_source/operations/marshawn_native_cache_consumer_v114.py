"""Optional create-only source PNG cache; original transport and SHA gate remain authoritative."""
from pathlib import Path
import json,sys,subprocess,shlex,time,uuid

def begin(V,frame,release,root,fleet):
 from prepare_marshawn_tuning import sha,write
 token=None
 try:
  O=V/'operations/native_png_prefetch_v1';cache=json.loads((O/'release.json').read_text())
  if sha(O/'release.json')!='74245e7657069eab1e1abef950eb1a37664653d50021da45db152623b1cd7e49':raise ValueError('Optional cache release changed')
  for p,d in cache['files_sha256'].items():
   if sha(p)!=d:raise ValueError('Optional cache helper changed '+p)
  sys.path.insert(0,str(O/'code'))
  from marshawn_native_png_prefetch import source_binding
  owner=fleet.identity();marker=O/'consumers'/f'{frame}.json';source=V/'sources'/str(frame)/'source_ready.json';cameras=json.loads(source.read_text())['cameras'];nonce=uuid.uuid4().hex
  token=dict(marker=str(marker),owner=owner,nonce=nonce)
  value=dict(kind='NATIVE_PNG_CACHE_CONSUMER_V1',frame=frame,source_ready_sha256=sha(source),status='active',owner=owner,nonce=nonce,normal_stage_target=str(Path(release['remote_root'])/'operations/sam3_assist/v5'/f'frame{frame:06d}'/'tasks'),created=time.time());write(marker,value)
  request=root/'optional_native_cache_request.json';remote=Path(release['remote_root'])/'operations/native_png_prefetch_v1';remote_request=remote/'requests'/f'consumer_{frame}_{nonce}.json'
  write(request,dict(root=str(remote),helper_sha256=cache['files_sha256'][str(O/'code/marshawn_native_png_cache_store.py')],bindings=[source_binding(V,frame,c,verify_png=False)for c in cameras]))
  subprocess.run(['rsync','-rt',str(request),release['remote_host']+':'+str(remote_request)],check=True,capture_output=True,text=True,timeout=15)
  cmd=['timeout','40',release['remote_python'],cache['consumer_API']['remote_helper'],'link-frame','--request',str(remote_request)]
  answer=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',release['remote_host'],shlex.join(cmd)],check=True,capture_output=True,text=True,timeout=45)
  result=json.loads(answer.stdout);write(root/'optional_native_cache_result.json',dict(result=result,request_sha256=sha(request),cache_release_sha256=sha(O/'release.json'),original_rsync_and_source_SHA_checks_still_required=True))
 except Exception as error:
  try:write(root/'optional_native_cache_result.json',dict(optional_cache_error=str(error),fall_through_original_transport=True,cache_not_authority_for_source_ready=True))
  except Exception:pass
 return token

def finish(token,status):
 if token is None:return
 try:
  from prepare_marshawn_tuning import write
  p=Path(token['marker']);v=json.loads(p.read_text())
  if v.get('nonce')!=token['nonce']or v.get('owner')!=token['owner']:return
  v.update(status=status,finished=time.time());write(p,v)
 except Exception:pass
