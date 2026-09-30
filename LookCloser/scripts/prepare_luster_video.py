"""Copy calibrated Luster frames through dev3 and freeze one sequence AABB."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'LookCloser/scripts'
REFERENCE = Path('/home/brans/lookcloser_luster/000470')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def environment():
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO)+os.pathsep+str(SCRIPTS),
               CUDA_HOME='/home/brans/repos/nerfstudio/.cuda128-toolchain',
               TORCH_EXTENSIONS_DIR='/home/brans/.cache/torch_extensions_lookcloser',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='2')
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env['CUDA_HOME']+'/bin'+os.pathsep+env['PATH']
    return env


def ingest(root, frame, host):
    source = root / 'source'; source.mkdir(parents=True, exist_ok=True)
    remote = f'/fsx/tmp/luster/root_8s/working/fullres/{frame}'
    for part in ['images', 'sparse/text']:
        dest = source/'frame'/part; dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync','-a',f'{host}:{remote}/{part}/',str(dest)+'/'],check=True)
    names = [f'cam_{i}/{frame}.png' for i in range(1,166) if i != 164]
    listing = source/'mask_files.txt'; listing.write_text('\n'.join(names)+'\n')
    (source/'masks').mkdir(exist_ok=True)
    subprocess.run(['rsync','-a','--files-from',str(listing),f'{host}:/fsx/tmp/luster/masks_sam3final_8s/',str(source/'masks')+'/'],check=True)
    for origin, target in [(f'/fsx/tmp/luster/masks_8s/cam_164/{frame}.png','plate_difference_cam164.png'),
                           ('/fsx/tmp/luster/root_8s/bounds_8s.json','bounds_8s.json')]:
        subprocess.run(['rsync','-a',f'{host}:{origin}',str(source/target)],check=True)
    overrides={}
    for cid in range(1,166):
        if cid==164:continue
        rgb=source/'frame/images'/f'cam_{cid:03d}_{frame}.jpg';mask=source/'masks'/f'cam_{cid}'/f'{frame}.png'
        with Image.open(rgb) as im:expected=im.size
        with Image.open(mask) as im:actual=im.size
        if actual==expected:continue
        relative=f'mask_fallbacks/cam_{cid:03d}.png';target=source/relative;target.parent.mkdir(exist_ok=True)
        subprocess.run(['rsync','-a',f'{host}:/fsx/tmp/luster/masks_8s/cam_{cid}/{frame}.png',str(target)],check=True)
        with Image.open(target) as im:
            if im.size!=expected:raise ValueError(f'Both mask variants mismatch: {frame}/{cid}')
        overrides[str(cid)]=dict(path=relative,reason='SAM dimensions mismatch calibrated RGB',sam_size=actual,rgb_size=expected)
    write(source/'mask_overrides.json',overrides)
    code = '''import hashlib,json,sys
from pathlib import Path
frame=sys.argv[1];overrides=json.loads(sys.argv[2]);base=Path('/fsx/tmp/luster/root_8s/working/fullres')/frame
items={str(Path('frame')/p.relative_to(base)):p for p in (base/'images').glob('*.jpg')}
items.update({str(Path('frame')/p.relative_to(base)):p for p in (base/'sparse/text').iterdir() if p.is_file()})
items.update({f'masks/cam_{i}/{frame}.png':Path('/fsx/tmp/luster/masks_sam3final_8s')/f'cam_{i}'/f'{frame}.png' for i in range(1,166) if i!=164 and str(i) not in overrides})
items.update({v['path']:Path('/fsx/tmp/luster/masks_8s')/f'cam_{i}'/f'{frame}.png' for i,v in overrides.items()})
items['plate_difference_cam164.png']=Path('/fsx/tmp/luster/masks_8s/cam_164')/f'{frame}.png'
items['bounds_8s.json']=Path('/fsx/tmp/luster/root_8s/bounds_8s.json')
print(json.dumps({k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in items.items()}))
'''
    result=subprocess.check_output(['ssh','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(code)+' '+shlex.quote(frame)+' '+shlex.quote(json.dumps(overrides))],text=True)
    write(source/'remote_manifest.json',json.loads(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--start',type=int,default=470)
    p.add_argument('--end',type=int,default=529);p.add_argument('--host',default='ubuntu@dev3');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    frames=[f'{i:06d}' for i in range(args.start,args.end+1)]
    if (args.root/'preparation_complete.json').exists():
        if json.loads((args.root/'preparation_complete.json').read_text())['frames']!=frames:raise ValueError('Prepared frame range differs')
        print('Sequence preparation already complete');return
    write(args.root/'manifest.json',dict(frames=frames,fps=30,duration_seconds=len(frames)/30,
          source='/fsx/tmp/luster/root_8s/working/fullres',normalization_reference=str(REFERENCE/'data/bounds_audit.json'),
          status='preparing',pid=os.getpid()))
    def prepare_one(frame):
        root=args.root/'frames'/frame;logdir=root/'logs';logdir.mkdir(parents=True,exist_ok=True)
        if (root/'data/audit_preprocessing.json').exists():return frame
        if shutil.disk_usage(args.root).free < 9*2**30:raise RuntimeError('Less than 9 GiB free before preprocessing')
        write(root/'prepare_progress.json',dict(phase='preparing',frame=frame,pid=os.getpid(),time=time.time()))
        if frame=='000470':
            for folder in ['source','data']:
                if not (root/folder).exists():shutil.copytree(REFERENCE/folder,root/folder)
        else:
            ingest(root,frame,args.host)
            if not (root/'data/complete.json').exists():
                with (logdir/'prepare.log').open('a') as log:
                    subprocess.run([sys.executable,str(SCRIPTS/'prepare_luster_frame.py'),str(root),'--frame',frame,
                         '--normalization-reference',str(REFERENCE/'data/bounds_audit.json'),'--reuse-images'],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
            with (logdir/'audit_preprocessing.log').open('a') as log:
                subprocess.run([sys.executable,str(SCRIPTS/'audit_luster_data.py'),str(root)],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
        return frame
    failures={}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(prepare_one,frame):frame for frame in frames}
        for count,future in enumerate(as_completed(futures),1):
            frame=futures[future]
            try:future.result()
            except Exception as exc:failures[frame]=str(exc)
            status=dict(phase='preparing',frame=frame,prepared=count-len(failures),total=len(frames),failures=failures,pid=os.getpid(),time=time.time())
            write(args.root/'progress.json',status);print(json.dumps(status),flush=True)
    if failures:
        write(args.root/'preparation_failures.json',failures)
        raise RuntimeError(f'Preparation failed for {len(failures)} frames; see preparation_failures.json')
    # Identical field coordinates are mandatory before copying temporal weights.
    audits=[json.loads((args.root/'frames'/f/'data/bounds_audit.json').read_text()) for f in frames]
    if any(a['normalization']!=audits[0]['normalization'] or a['scale']!=audits[0]['scale'] for a in audits):raise ValueError('Sequence coordinates differ')
    bounds=np.array([a['bounds'] for a in audits]);union=np.stack([bounds[:,0].min(0),bounds[:,1].max(0)])
    for frame in frames:
        data=args.root/'frames'/frame/'data';meta=json.loads((data/'transforms.json').read_text())
        write(data/'frame_transforms_before_sequence_bounds.json',meta)
        meta['blur_aabb']=union.tolist();write(data/'transforms.json',meta)
        complete=json.loads((data/'complete.json').read_text())
        complete['frame_transforms_sha256']=complete['transforms_sha256']
        complete['transforms_sha256']=hashlib.sha256((data/'transforms.json').read_bytes()).hexdigest()
        complete['sequence_bounds']=str(args.root/'sequence_bounds.json');write(data/'complete.json',complete)
        # A later frequency/data audit must bind the new transforms hash.
        (data/'audit_ready.json').unlink(missing_ok=True)
    write(args.root/'sequence_bounds.json',dict(bounds=union.tolist(),frames=frames,
          frame_bounds={f:a['bounds'] for f,a in zip(frames,audits)},reference_bounds=audits[0]['bounds'],
          normalization=audits[0]['normalization'],scale=audits[0]['scale']))
    write(args.root/'preparation_complete.json',dict(frames=frames,bounds=union.tolist()))
    manifest=json.loads((args.root/'manifest.json').read_text());manifest['status']='prepared'
    write(args.root/'manifest.json',manifest)


if __name__=='__main__':main()
