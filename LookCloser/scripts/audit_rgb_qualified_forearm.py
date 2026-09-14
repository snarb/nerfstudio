"""Fresh geometry-prefix and 124-ray audit of the explicitly RGB-qualified guard."""
from pathlib import Path
from functools import partial
import argparse
from joint_temporal_texture import read,sha,atomic_json
import audit_color_qualified_forearm as base
from rgb_qualified_forearm_depth_guard import make_guard


def run(root,frame):
    request=read(root/frame/'request.json');limit=request['observed_guard']['rgb_mean_abs_limit']
    for name,h in request['rgb_guard_script_hashes'].items():
        if sha(Path(__file__).with_name(name))!=h:raise ValueError('Changed RGB guard')
    original=base.make_guard
    try:
        base.make_guard=partial(make_guard,rgb_limit=limit)
        base.run(root,frame)
        path=root/'fresh_audit'/(frame+'.json');result=read(path)
        result.update(rgb_wrapper_script_sha256=sha(__file__),rgb_mean_abs_limit=limit,
                      chroma_only_guard_not_equivalent=True)
        atomic_json(path,result)
    finally:
        base.make_guard=original


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',required=True)
    a=p.parse_args();run(a.root,a.frame)
