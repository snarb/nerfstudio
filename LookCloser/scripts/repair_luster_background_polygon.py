"""Apply an explicitly reviewed background-only polygon, preserving prior inputs."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image
from prepare_luster_video import write
from archive_luster_checkpoint import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('spec',type=Path)
    args=p.parse_args();spec=json.loads(args.spec.read_text())
    if not spec.get('reviewed') or not spec.get('reason'):raise ValueError('Explicit visual review required')
    frame=spec['frame'];camera=int(spec['camera']);base=args.root/'frames'/frame;data=base/'data'
    if len(frame)!=6 or not frame.isdigit():raise ValueError('Invalid frame')
    if not spec['id'] or any(not (c.isalnum() or c=='_') for c in spec['id']):raise ValueError('Invalid revision id')
    source=base/'source';name=f'cam_{camera:03d}_{frame}.png'
    meta=json.loads((data/'transforms.json').read_text())
    if f'images/{name}' not in meta['train_filenames']:raise ValueError('This repair supports train views only; eval protocol must stay fixed')
    rgb_path=source/'frame/images'/f'cam_{camera:03d}_{frame}.jpg'
    mask_path=source/'masks'/f'cam_{camera}'/f'{frame}.png'
    if sha(rgb_path)!=spec['rgb_sha256'] or sha(mask_path)!=spec['mask_sha256']:
        raise ValueError('Reviewed source identity differs')
    revision=data/'revisions'/spec['id']
    if revision.exists():raise ValueError('Revision already exists')
    rgb=np.array(Image.open(rgb_path).convert('RGB'));mask=np.array(Image.open(mask_path).convert('L'))
    polygon=np.array(spec['polygon_full_resolution'],dtype=np.int32)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3:raise ValueError('Invalid polygon')
    if (polygon<0).any() or (polygon[:,0]>=mask.shape[1]).any() or (polygon[:,1]>=mask.shape[0]).any():
        raise ValueError('Polygon outside calibrated image')
    remove=np.zeros(mask.shape,np.uint8);cv2.fillPoly(remove,[polygon],1)
    repaired=mask.copy();repaired[remove>0]=0
    if not np.any(mask!=repaired):raise ValueError('Empty correction')
    revision.mkdir(parents=True)
    paths=[f'images/{name}',f'masks/{name}',
           *[str(x.relative_to(data)) for x in (data/'lookcloser_frequencies').glob(f'{Path(name).stem}.*')],
           'derived_manifest.json','audit_ready.json','audit_preprocessing.json','frequency_complete.json']
    before={}
    for relative in paths:
        path=data/relative
        if path.exists():
            target=revision/'before'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);before[relative]=sha(path)
    size=Image.open(data/'images'/name).size
    coverage=cv2.resize(repaired,size,interpolation=cv2.INTER_AREA)
    masked=cv2.resize(rgb.astype(np.float32)*(repaired[...,None]/255.),size,interpolation=cv2.INTER_AREA)
    Image.fromarray(np.rint(masked).clip(0,255).astype('uint8')).save(data/'images'/name,compress_level=1)
    Image.fromarray(coverage).save(data/'masks'/name,compress_level=1)
    # A removed silhouette subset cannot enlarge the existing conservative hull
    # or AABB. Keep both fixed to isolate the correction; rerun the data audit.
    for relative in paths:
        if relative.startswith('lookcloser_frequencies/') or relative in {'audit_ready.json','frequency_complete.json'}:
            (data/relative).unlink(missing_ok=True)
    record=dict(spec=spec,before=before,time=time.time(),removed_source_pixels=int((mask!=repaired).sum()),
                after={f'{folder}/{name}':sha(data/folder/name) for folder in ['images','masks']},
                geometry='Existing conservative hull, coordinates, AABB and ROIs unchanged; corrected mask is a subset',
                status='requires_frequency_refit_and_data_audit')
    write(revision/'receipt.json',record);write(data/'revision.json',dict(id=spec['id'],receipt=str(revision/'receipt.json')))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
