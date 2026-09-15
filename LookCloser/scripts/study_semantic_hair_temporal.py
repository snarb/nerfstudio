"""Twelve consecutive actor times / moving cameras for the frozen hair policy.

Runtime configuration is explicit and hash-bound here; the two-frame producer
scripts and their already-published experimental artifacts stay unchanged.
"""
from pathlib import Path
import argparse
import os
import subprocess
from build_train_hair_semantics import read,sha,write as atomic_json

ROOT=Path('/mnt/data/dec5_semantic_hair_temporal_v2')
FRAMES=[f'{i:06d}' for i in range(1073,1097,2)]


def configuration():
    config=dict(frames=FRAMES,semantics=str(ROOT/'semantics'),render=str(ROOT/'render'),
        script_hashes={name:sha(Path(__file__).with_name(name)) for name in [Path(__file__).name,
            'build_train_hair_semantics.py','study_semantic_hair_sources.py']},
        recipe_changed_from_two_frame=False,production_promoted=False,temporal_rgb_blending=False)
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'configuration.json').exists():assert read(ROOT/'configuration.json')==config
    else:atomic_json(ROOT/'configuration.json',config)


def semantics(action):
    import build_train_hair_semantics as stage
    original=stage.ROOT;stage.ROOT=ROOT/'semantics';stage.FRAMES=FRAMES
    models=stage.ROOT/'models';models.mkdir(parents=True,exist_ok=True)
    for name,_ in stage.MODELS.values():
        source=original/'models'/name;target=models/name
        if target.exists():assert target.is_symlink() and target.resolve()==source.resolve()
        else:target.symlink_to(source)
    getattr(stage,action)()


def render(action):
    import study_semantic_hair_sources as worker
    worker.ROOT=ROOT/'render';worker.SEMANTICS=ROOT/'semantics';worker.FRAMES=FRAMES
    getattr(worker,action)()


def package():
    import numpy as np
    from PIL import Image,ImageDraw
    from study_semantic_hair_sources import BASE
    dest=ROOT/'review';dest.mkdir(exist_ok=True)
    # Small overviews and unscaled crown/side-hair pixel crops complement the
    # full-resolution per-frame crown/face/body panels retained by review().
    for first in range(0,len(FRAMES),4):
        overview=Image.new('RGB',(4*540,505),(25,25,25));draw=ImageDraw.Draw(overview)
        crown=Image.new('RGB',(4*640,505),(25,25,25));dc=ImageDraw.Draw(crown)
        for col,frame in enumerate(FRAMES[first:first+4]):
            for side,root in enumerate([BASE,ROOT/'render']):
                im=Image.open(root/'frames'/frame/'frame.png').convert('RGB')
                overview.paste(im.resize((270,480),Image.Resampling.LANCZOS),(col*540+side*270,25))
                # Fixed screen rectangle only for human review, never the source policy.
                crown.paste(im.crop((160,40,480,520)),(col*640+side*320,25))
            draw.text((col*540+4,5),frame+' baseline | semantic',fill='white')
            dc.text((col*640+4,5),frame+' baseline | semantic 1:1',fill='white')
        overview.save(dest/f'overview_{first:02d}.png');crown.save(dest/f'crown_{first:02d}.png')
    (dest/'sequence').mkdir(exist_ok=True)
    for index,frame in enumerate(FRAMES):
        path=dest/'sequence'/f'{index:04d}.png';source=ROOT/'render'/'frames'/frame/'frame.png'
        if path.exists():assert path.is_symlink() and path.resolve()==source.resolve()
        else:path.symlink_to(source)
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    subprocess.run(['ffmpeg','-y','-v','error','-framerate','24','-i',str(dest/'sequence'/'%04d.png'),
        '-c:v','libx264','-crf','16','-preset','slow','-pix_fmt','yuv420p',str(dest/'diagnostic_12f.mp4')],env=env,check=True)
    q=read(BASE/'request.json');selected=[r for r in q['inventory'] if r['frame_id'] in FRAMES]
    positions=np.array([r['camera']['transform_matrix'] for r in selected])[:,:3,3]
    assert len(selected)==12 and np.unique(positions,axis=0).shape[0]==12
    atomic_json(dest/'package.json',dict(frames=FRAMES,actor_times_unique=True,camera_positions_unique=12,
        diagnostic_duration_seconds=.5,fps=24,script_sha256=sha(__file__),
        movie_sha256=sha(dest/'diagnostic_12f.mp4'),visual_status='pending',production_promoted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['stage','infer','render','review','package']);a=parser.parse_args()
    configuration()
    if a.action in ['stage','infer']:semantics(a.action)
    elif a.action in ['render','review']:render(a.action)
    else:package()
