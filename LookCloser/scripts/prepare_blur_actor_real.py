"""Freeze a 62/3 real foreground diagnostic matching the historical actor extent.

RGB remains the observed full photo. Train masks limit supervision; full-frame
renders therefore have an untrained background and are not a full-scene result.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    root=Path('/home/brans/lookcloser_artifacts/blur_ablation_fresh')
    source=root/'real'
    historical=Path('/mnt/data/dec5_lookcloser_recovery_v2/real_all')
    metadata=deepcopy(json.loads((source/'transforms.json').read_text()))
    previous=json.loads((historical/'transforms.json').read_text())
    by_name={Path(r['file_path']).stem:r for r in previous['frames']}
    out=root/'actor_real';out.mkdir(exist_ok=True)
    (out/'masks').mkdir(exist_ok=True)
    for name,target in [('images',source/'images'),('lookcloser_frequencies',source/'lookcloser_frequencies')]:
        if not (out/name).exists():(out/name).symlink_to(target)
    receipt=[]
    for row in metadata['frames']:
        stem=Path(row['file_path']).stem
        mask=out/'masks'/(stem+'.png')
        if row['file_path'] in metadata['train_filenames']:
            original=by_name[stem]
            if row['transform_matrix']!=original['transform_matrix']:raise ValueError('Historical camera gauge differs')
            target=historical/original['mask_path']
            if not mask.exists():mask.symlink_to(target)
            valid=np.array(Image.open(mask))>0
            receipt.append(dict(image=row['file_path'],mask=str(target),valid_fraction=float(valid.mean())))
        elif not mask.exists():
            # Eval masks are never used for scores or training. A common all-ones
            # mask only satisfies the stock parser's all-frames mask contract.
            Image.fromarray(np.full((row['h'],row['w']),255,dtype='uint8')).save(mask)
        row['mask_path']='masks/'+mask.name
    metadata['blur_aabb']=previous['distillation']['actor_bounds']
    metadata['diagnostic_scope']='Real foreground only; background unsupervised. No teacher RGB or depth.'
    (out/'transforms.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'mask_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(out,len(metadata['train_filenames']),len(metadata['val_filenames']))


if __name__=='__main__':main()
