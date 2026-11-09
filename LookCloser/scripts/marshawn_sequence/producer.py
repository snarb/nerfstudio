"""CPU-only, locked per-frame source preparation on its NVIDIA data owner."""
import argparse
import fcntl
from pathlib import Path
import shutil
from common import read,atomic,sha


def produce(root,frame,linear=False):
    root=Path(root);settings=read(root/'settings.json');folder=root/'prepared'/f'{frame:06d}'
    folder.parent.mkdir(parents=True,exist_ok=True)
    lock=folder.with_suffix('.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    import sys
    sys.path[:0]=[settings['source_code']]
    from prepare import prepare
    ready=prepare(frame,folder,root/'box.json',root/'split.json',workers=3,retain_linear=linear)
    source=Path(settings.get('seed_root',str(root/'seed_points')))/f'{frame:06d}'/'sparse/text/points3D.txt'
    if not source.exists():raise FileNotFoundError('Required original seed cloud '+str(source))
    shutil.copy2(source,folder/'points3D.txt')
    atomic(folder/'producer_complete.json',dict(frame=frame,native_ready_sha256=sha(folder/'native_ready.json'),seed_sha256=sha(folder/'points3D.txt'),source_owner=__import__('socket').gethostname(),retain_linear=linear,total_native_pixels=sum(r['camera_row']['w']*r['camera_row']['h'] for r in ready['records'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--linear',action='store_true');a=p.parse_args();produce(a.root,a.frame,a.linear)
