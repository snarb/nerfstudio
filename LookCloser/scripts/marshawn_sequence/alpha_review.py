"""Auxiliary opacity leakage on fixed, conservatively labelled rendered rays."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from common import read,atomic,sha
from worker_background import trusted_background


def review(data,render,stride=8):
    data,render=Path(data),Path(render);path=render/'background_leakage.json'
    if path.exists():return read(path)
    views={}
    for name in read(data/'ready.json')['validation']:
        source=render/(name+'_alpha.png')
        with np.load(data/'semantics'/(name+'.npz')) as s,np.load(data/'parts'/(name+'.npz')) as p:
            trusted=trusted_background(s,p['unknown_people']|p['crew_exclusion'])
        queried=np.zeros(trusted.shape,bool)
        for x0,y0,x1,y1 in read(render/(name+'_render_domain.json'))['rendered_rois']:queried[y0:y1,x0:x1]=True
        alpha=np.asarray(Image.open(source),dtype=np.float32)[::stride,::stride]/65535
        values=alpha[(trusted&queried)[::stride,::stride]]
        if values.size:
            views[name]=dict(mean_opacity=float(values.mean()),p95_opacity=float(np.quantile(values,.95)),sampled_background_rays=int(values.size),alpha_sha256=sha(source))
    result=dict(mean_opacity_equal_views=float(np.mean([v['mean_opacity'] for v in views.values()])) if views else None,by_view=views,stride=stride,domain='auxiliary sampled trusted background within identical native actor ROIs; unqueried exterior excluded; not a full-scene geometry certificate')
    atomic(path,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--render',required=True);a=p.parse_args()
    print(__import__('json').dumps(review(a.data,a.render)),flush=True)
