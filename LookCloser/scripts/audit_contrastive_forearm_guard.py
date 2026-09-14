"""Fresh prefix/ray audit for the explicit depth-discriminating confidence rule."""
from pathlib import Path
from functools import partial
import argparse
from joint_temporal_texture import read,sha,atomic_json
import audit_color_qualified_forearm as base
from contrastive_forearm_depth_guard import make_guard


def run(root,frame):
    request=read(root/frame/'request.json');margin=request['observed_guard']['comparison_margin']
    for group in ['rgb_guard_script_hashes','comparison_guard_script_hashes']:
        for name,h in request[group].items():
            if sha(Path(__file__).with_name(name))!=h:raise ValueError('Changed comparison guard')
    original=base.make_guard
    try:
        base.make_guard=partial(make_guard,margin=margin);base.run(root,frame)
        path=root/'fresh_audit'/(frame+'.json');result=read(path)
        result.update(comparison_wrapper_sha256=sha(__file__),comparison_margin=margin,rgb_only_guard_not_equivalent=True)
        atomic_json(path,result)
    finally:base.make_guard=original


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True)
    p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_contrastive_guard'))
    a=p.parse_args();run(a.root,a.frame)
