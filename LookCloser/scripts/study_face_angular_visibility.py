"""Matched face-only angular controls with or without skin-consensus admission.

Train RGB only, same mesh/camera/color, no averaging. The old renderer remains
unchanged outside skin-supported source replacements. This is a diagnostic,
not a validated full-video mode.
"""
import argparse
import hashlib
import inspect
from pathlib import Path
import shutil
import numpy as np
import study_face_interior_visibility as base
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_face_angular_visibility')
PARENT=Path('/mnt/data/dec5_face_interior_visibility90_001123')
MODES=['raster','consensus']


def angular_proposals(old_valid,face_support,weights,chosen,mode):
    if mode not in MODES:raise ValueError(mode)
    j=np.arange(old_valid.shape[1]);safe=np.clip(chosen,0,old_valid.shape[0]-1)
    anchor=(chosen>=0)&(chosen<old_valid.shape[0])&old_valid[safe,j]
    votes=(old_valid&face_support).sum(0);old=weights[safe,j]
    candidate=face_support&(weights>old[None])&anchor[None]&(votes[None]>=3)
    if mode=='raster':candidate&=old_valid
    return candidate,votes,old


def transformed_render():
    source=inspect.getsource(base.render)
    old="weights=quality((direction*normal[f]).sum(-1),length,'incidence2')*angle[:,None]"
    assert source.count(old)==1
    return source.replace(old,"weights=np.broadcast_to(angle[:,None],length.shape)")


def configure(mode,frame='001123'):
    out=ROOT/frame/mode
    source=transformed_render()
    base.ROOT=out;base.FRAME=frame
    base.proposals=lambda *args:angular_proposals(*args,mode=mode)
    base.__dict__['__file__']=__file__
    exec(compile(source,__file__+':angular_render','exec'),base.__dict__)
    return out,hashlib.sha256(source.encode()).hexdigest()


def main(stage,mode):
    original=Path(base.__file__).resolve();out,generated=configure(mode)
    if stage=='prepare':
        assert not out.exists();out.mkdir(parents=True)
        q=read(PARENT/'request.json')
        for p,h in q['input_hashes'].items():assert sha(p)==h
        assert sha(PARENT/'face_masks.npz')==q['face_masks_sha256']
        shutil.copyfile(PARENT/'face_masks.npz',out/'face_masks.npz')
        q['input_hashes'].update({str(original):sha(original),str(Path(__file__).resolve()):sha(__file__),
            str(PARENT/'request.json'):sha(PARENT/'request.json')})
        q.update(script_sha256=sha(__file__),generated_render_sha256=generated,
            source_quality='angular_only_inside_train_skin_consensus',
            admission_mode=mode,old_selected_source_requires_face_semantics=False,
            allow_already_raster_valid_replacements=True,full_video_accepted=False)
        save(out/'request.json',q)
    else:
        assert read(out/'request.json')['generated_render_sha256']==generated
        base.render()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','render']);p.add_argument('--mode',choices=MODES,required=True);a=p.parse_args()
    base.torch.set_num_threads(2)
    with base.torch.inference_mode():main(a.stage,a.mode)
