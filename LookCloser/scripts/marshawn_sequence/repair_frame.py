"""Independent corrected frame plus automatic export audit and six-view renders."""
import argparse
import subprocess
import sys
from pathlib import Path
from common import read, atomic, sha
from production import run
from audit_exports import audit


def repair(root, frame):
    root=Path(root)
    run(root,frame)
    report=audit(root)
    atomic(root/'export_audit.json',report)
    row=next(x for x in report['results'] if x['frame']==frame)
    if row['status']!='verified':raise ValueError('Repaired model export did not verify: '+str(row))
    folder=root/'frames'/f'{frame:06d}'
    settings=read(root/'settings.json');tools=Path(settings.get('production_tools',settings['tools']))
    base=[sys.executable,str(tools/'render_fullhd.py'),'--root',str(root),
          '--spec',str(folder/'train/spec.json'),'--checkpoint',str(folder/'model/inference.ckpt'),
          '--sizes',str(root/'original_camera_sizes.json')]
    review=folder/'final_review'
    for name,extra in [('fullhd',[]),('novel',['--novel-fraction','.25'])]:
        if not (review/name/'render_receipt.json').exists():
            subprocess.run(base+['--output',str(review/name)]+extra,check=True)
        receipt=read(review/name/'render_receipt.json')
        if receipt['checkpoint_sha256']!=row['model_sha256'] or receipt['output_canvas']!=[1920,1080]:
            raise ValueError('Repaired final render provenance differs')
    completion=read(folder/'frame_complete.json')
    atomic(folder/'repair_ready.json',dict(frame=frame,recipe_sha256=completion['recipe_sha256'],
        checkpoint_sha256=row['model_sha256'],export_verified=True,render_ready=True,
        visual_reviewed=False,frame_completion_sha256=sha(folder/'frame_complete.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frame',required=True,type=int)
    a=p.parse_args();repair(a.root,a.frame)
