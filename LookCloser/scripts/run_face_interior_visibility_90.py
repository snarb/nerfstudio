"""Retain the empty .95 preflight; test a train-only reviewed .90 mask gate.

The .95 threshold admitted zero pixels in all62 train cameras (central-face
maxima240..242/255). This exact adapter changes only that semantic threshold,
not mesh/ray/quality/vote gates, and binds both generated and original code.
"""
import argparse
import hashlib
import inspect
from pathlib import Path
import study_face_interior_visibility as base
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_face_interior_visibility90_001123')


def main(stage):
    source=inspect.getsource(base.prepare)
    for old,new in [('conf>=243','conf>=230'),('confidence_u8_min=243','confidence_u8_min=230')]:
        assert source.count(old)==1
        source=source.replace(old,new)
    proof=dict(wrapper_sha256=sha(__file__),base_script_sha256=sha(base.__file__),
        generated_prepare_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_empty_request_sha256=sha(base.ROOT/'request.json'),
        threshold_basis='Zero .95 support across all62 train views; .90 is a global semantic threshold, not a target-specific mask edit.')
    base.ROOT=ROOT;base.__dict__['__file__']=__file__
    if stage=='prepare':
        exec(compile(source,__file__+':prepare90','exec'),base.__dict__);base.prepare()
        save(ROOT/'adapter.json',proof)
    else:
        assert read(ROOT/'adapter.json')==proof
        base.render()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','render']);a=p.parse_args()
    base.torch.set_num_threads(2)
    with base.torch.inference_mode():main(a.stage)
