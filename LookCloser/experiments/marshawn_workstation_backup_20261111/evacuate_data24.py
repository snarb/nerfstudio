import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109')
BACKUP=Path('/cache/ubuntu/lookcloser/emergency_workstations_20261111')
sys.path.insert(0,str(ROOT/'tools'))
from common import atomic,read,sha
from native_transfer import ensure_native

def validate(native,frame):
    ready=read(native/'native_ready.json')
    records=ready['records']
    if ready['frame']!=frame or len(records)!=72 or len({r['camera'] for r in records})!=72:
        raise ValueError('Frame/camera coverage differs')
    if not ready['native_no_resize']:
        raise ValueError('Native resolution requirement differs')
    for row in records:
        if sha(native/row['image'])!=row['rgb_sha256']:
            raise ValueError('RGB checksum differs '+row['camera'])
        semantic=native/'semantics'/(row['camera']+'.npz')
        if sha(semantic)!=row['semantic_sha256']:
            raise ValueError('Semantic checksum differs '+row['camera'])
        if not (native/row['camera_row']['mask_path']).is_file():
            raise ValueError('Sampling mask absent')
    if not (native/'points3D.txt').is_file():
        seed=ROOT/'seed_points'/f'{frame:06d}/sparse/text/points3D.txt'
        shutil.copy2(seed,native/'points3D.txt')
    return dict(frame=frame,cameras=72,native_no_resize=True,native_ready_sha256=sha(native/'native_ready.json'),seed_sha256=sha(native/'points3D.txt'),path=str(native),time=time.time())

def transfer(frame):
    folder=ROOT/'frames'/f'{frame:06d}'
    folder.mkdir(parents=True,exist_ok=True)
    native=folder/'native'
    if not (native/'native_ready.json').exists():
        candidates=[
            ROOT/'external_models/clever'/f'{frame:06d}/native',
            ROOT/'repairs_votes3_20261110/frames'/f'{frame:06d}/native',
        ]
        for name,prefix in [('ws07','home/pavlo/marshawn_campaign_20261109'),('ws08','home/nvuser/lookcloser/campaigns/marshawn908_20261109')]:
            candidates.append(BACKUP/name/'filesystem'/prefix/'frames'/f'{frame:06d}/native')
        candidate=next((p for p in candidates if (p/'native_ready.json').exists()),None)
        if candidate:
            validate(candidate,frame)
            with (folder/'native_transfer.lock').open('a+') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                if not (native/'native_ready.json').exists():
                    temporary=folder/'native.evacuation'
                    if temporary.exists():shutil.rmtree(temporary)
                    shutil.copytree(candidate,temporary,symlinks=True)
                    if native.exists():shutil.rmtree(native)
                    temporary.replace(native)
        else:
            ensure_native(ROOT,frame)
    receipt=validate(native,frame)
    # Existing native data may originate from a local source or verified archive.
    # Persist the same admission receipt expected by the normal training entry.
    with (folder/'native_transfer.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        if not (native/'transfer_complete.json').exists():
            atomic(native/'transfer_complete.json',receipt)
        atomic(native/'source_cache_released.json',dict(frame=frame,native_ready_sha256=receipt['native_ready_sha256'],adopted_verified_local_copy=True))
    atomic(BACKUP/'data24'/f'{frame:06d}.json',receipt)
    print('DATA24_VERIFIED',frame,flush=True)
    return receipt

while not all((BACKUP/name/'checkpoint_verification.json').exists() for name in ['ws07','ws08']):
    time.sleep(30)
for name in ['ws07','ws08']:
    if read(BACKUP/name/'checkpoint_verification.json')['errors']:
        raise ValueError('Checkpoint backup verification failed')
(BACKUP/'data24').mkdir(exist_ok=True)
frames=list(range(908,1268,5))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    receipts=list(pool.map(transfer,frames))
atomic(BACKUP/'data24_complete.json',dict(time=time.time(),frames=frames,total=72,verified=len(receipts),records=receipts))
