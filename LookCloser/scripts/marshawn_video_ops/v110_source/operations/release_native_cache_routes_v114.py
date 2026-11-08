from pathlib import Path
import sys,json,fcntl,time,hashlib
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';C=O/'sam3_assist/code';R=O/'sam3_assist/v5';K=O/'native_png_prefetch_v1';sys.path[:0]=[str(O),str(C),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
approval=K/'ROOT_authorization.json';assert sha(approval)=='6dd8eefefae4b18fddb8dc258dc3feb22d70104dacd8e7bb4e04748a9fcbdee9'
r=json.loads((R/'native_future_release_v114.json').read_text());files=r['files_sha256']
for p in [approval,O/'test_native_cache_consumer_v114.py',K/'consumer_v114_CPU_PASS.json']:files[str(p)]=sha(p)
r.update(ROOT_cache_authorization_sha256=sha(approval),optional_cache_consumer_CPU_qualification_sha256=sha(K/'consumer_v114_CPU_PASS.json'))
write(R/'native_future_release_v114.json',r)
import marshawn_mask_assist_full_v114 as full
full.guard(V,1705)
cli=C/'marshawn_mask_assist_full_v114.py';routes=O/'native_pack_cpu8_v2/routes';routes.mkdir(exist_ok=True);records=[]
for frame in range(1705,2016,5):
 lock=open(routes/(str(frame)+'.lock'),'a+');fcntl.flock(lock,fcntl.LOCK_EX);choice=routes/(str(frame)+'.json');root=R/f'frame{frame:06d}'
 if choice.exists():records.append(dict(frame=frame,status='existing_choice_preserved',choice_sha256=sha(choice)));continue
 if root.exists()and any(root.iterdir()):records.append(dict(frame=frame,status='existing_BODY_inputs_preserved'));continue
 write(choice,dict(frame=frame,version=114,cli=str(cli),cli_sha256=sha(cli),preexisting_BODY_inputs_preserved=False,native_release_sha256=sha(R/'native_future_release_v114.json'),ROOT_cache_authorization_sha256=sha(approval),kind='QUALIFIED_SHARED_BODY_STAGE_ROUTE_RELEASE',no_NN_launch_before_existing_producer_previous_2D_proof=True,created=time.time()))
 records.append(dict(frame=frame,status='new_unclaimed_BODY_route',choice_sha256=sha(choice)))
write(K/'routes_v114_release.json',dict(kind='SHARED_SOURCE_PREFETCH_UNCLAIMED_BODY_CACHE_ROUTES_V114',records=records,native_release_sha256=sha(R/'native_future_release_v114.json'),ROOT_authorization_sha256=sha(approval),current1700_unchanged=True,existing_source_prefetch_CPU_owners_unchanged=True,created=time.time()));print('released',sum(x['status']=='new_unclaimed_BODY_route'for x in records),'unclaimed routes; existing work preserved')
