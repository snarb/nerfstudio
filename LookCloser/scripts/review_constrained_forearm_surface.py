"""Reuse an exact canary and score a frozen three-time constrained transfer."""
from pathlib import Path
from copy import deepcopy
import argparse
from joint_temporal_texture import read,sha,atomic_json

OUT=Path('/mnt/data/dec5_constrained_forearm_surface_guard16')
OLD=Path('/mnt/data/dec5_constrained_forearm_surface')
REVIEW=Path('/mnt/data/dec5_constrained_forearm_guard16_review')


def reuse_pilot():
    frame='001037';previous=read(OLD/frame/'request.json');current=read(OUT/frame/'request.json')
    previous=deepcopy(previous);previous['guard_max_rounds']=current['guard_max_rounds']
    previous['scripts']['study_coherent_forearm_replacement.py']=current['scripts']['study_coherent_forearm_replacement.py']
    if previous!=current:raise ValueError('Guard work limit is not the only changed input')
    if sha(OLD/frame/'guarded.ply')!=sha(OUT/frame/'guarded.ply'):raise ValueError('Cannot reuse RGB of changed mesh')
    result=read(OUT/frame/'geometry_result.json')
    if not result['observed_guard_passed']:raise ValueError('New candidate is not verified')
    hashes={}
    for view in ['moving','H004_A005_1210M6']:
        root=OLD/'rgb'/frame/view/'guarded';folder=root/'frames'/frame;c=read(folder/'complete.json')
        if c['request_sha256']!=sha(root/'request.json'):raise ValueError('Changed previous rendering')
        for n,h in c['hashes'].items():
            if sha(folder/n)!=h:raise ValueError('Changed prior output')
            hashes[str(folder/n)]=h
        hashes[str(root/'request.json')]=sha(root/'request.json')
    link=OUT/'rgb'/frame;link.parent.mkdir(exist_ok=True)
    if link.exists():
        if not link.is_symlink() or link.resolve()!=(OLD/'rgb'/frame).resolve():raise ValueError('Unexpected reuse destination')
    else:link.symlink_to(OLD/'rgb'/frame,target_is_directory=True)
    atomic_json(OUT/frame/'rgb_reuse.json',dict(source=str(OLD/'rgb'/frame),mesh_bytes_exact=True,
        old_result_sha256=sha(OLD/frame/'geometry_result.json'),new_result_sha256=sha(OUT/frame/'geometry_result.json'),
        old_request_sha256=sha(OLD/frame/'request.json'),new_request_sha256=sha(OUT/frame/'request.json'),
        only_guard_work_limit_changed=True,reused_hashes=hashes))
    print('Reused exact 001037 RGB',flush=True)


def score():
    import review_confidence_boundary_completion as scorer
    scorer.ROOTS={'previous':Path('/mnt/data/dec5_forearm_admission_quadric_bounded'),'constrained':OUT}
    scorer.run(REVIEW,['001029','001033','001037'],variants=['previous','constrained'])


def rgb_extent():
    import numpy as np
    from PIL import Image
    metrics=read(REVIEW/'metrics.json');records=[]
    for frame in ['001029','001033','001037']:
        for view in ['moving','H004_A005_1210M6']:
            paths=[Path('/mnt/data/dec5_forearm_admission_quadric_bounded')/'rgb'/frame/view/'guarded/frames'/frame/'frame.png',
                OUT/'rgb'/frame/view/'guarded/frames'/frame/'frame.png']
            for p in paths:
                if sha(p)!=metrics['prediction_hashes'][str(p)]:raise ValueError('Changed scored image')
            a,b=[np.array(Image.open(p)) for p in paths];changed=np.any(a!=b,2);y,x=np.nonzero(changed)
            records.append(dict(frame=frame,view=view,changed_rgb_pixels=int(changed.sum()),
                changed_rgb_pixels_top_1400_rows=int(changed[:1400].sum()),
                changed_bbox=None if not len(x) else [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]))
    atomic_json(REVIEW/'rgb_change_extent.json',dict(records=records,scope='pixel identity diagnostic, not full-frame quality metrics',script_sha256=sha(__file__)))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['reuse-pilot','score','rgb-extent']);a=p.parse_args()
    if a.action=='reuse-pilot':reuse_pilot()
    elif a.action=='score':score()
    else:rgb_extent()
