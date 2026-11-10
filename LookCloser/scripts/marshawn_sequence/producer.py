"""CPU-only, locked per-frame source preparation on its NVIDIA data owner."""
import argparse
import fcntl
from pathlib import Path
import shutil
from common import read,atomic,sha


def produce(root,frame,linear=False,nonblocking=False):
    root=Path(root);settings=read(root/'settings.json');folder=root/'prepared'/f'{frame:06d}'
    folder.parent.mkdir(parents=True,exist_ok=True)
    lock=folder.with_suffix('.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
    except BlockingIOError:return
    if nonblocking and (folder/'producer_complete.json').exists():return
    import sys
    sys.path[:0]=[settings['source_code']]
    from prepare import prepare
    ready=prepare(frame,folder,root/'box.json',root/'split.json',workers=3,retain_linear=linear)
    source=Path(settings.get('seed_root',str(root/'seed_points')))/f'{frame:06d}'/'sparse/text/points3D.txt'
    if not source.exists():raise FileNotFoundError('Required original seed cloud '+str(source))
    shutil.copy2(source,folder/'points3D.txt')
    atomic(folder/'producer_complete.json',dict(frame=frame,native_ready_sha256=sha(folder/'native_ready.json'),seed_sha256=sha(folder/'points3D.txt'),source_owner=__import__('socket').gethostname(),retain_linear=linear,total_native_pixels=sum(r['camera_row']['w']*r['camera_row']['h'] for r in ready['records'])))


def release(root,frame,expected_ready_sha):
    root=Path(root);folder=root/'prepared'/f'{frame:06d}'
    if not 908<=frame<=1268 or not expected_ready_sha:raise ValueError('Scoped frame and acknowledged manifest required')
    folder.parent.mkdir(parents=True,exist_ok=True)
    with folder.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not folder.exists():return
        if folder.is_symlink():raise ValueError('Refusing to release linked staging data')
        receipt=read(folder/'producer_complete.json')
        if receipt['native_ready_sha256']!=expected_ready_sha:raise ValueError('Staging acknowledgement differs')
        history=root/'prepared_released';history.mkdir(exist_ok=True)
        atomic(history/(f'{frame:06d}.json'),dict(receipt,released_after_verified_transfer=True))
        shutil.rmtree(folder)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--linear',action='store_true');p.add_argument('--nonblocking',action='store_true');p.add_argument('--release',action='store_true');p.add_argument('--expected-ready-sha');a=p.parse_args()
    if a.release:release(a.root,a.frame,a.expected_ready_sha)
    else:produce(a.root,a.frame,a.linear,a.nonblocking)
