"""Same-raw Poisson control with measured-refined semantic masks only.

Preserve proposal mesh, direct-depth support, seed certificates and final
62-camera free-space gates. Refined masks affect geometry admission only.
"""
import argparse
import hashlib
import importlib
import inspect
from pathlib import Path
from joint_temporal_texture import read,sha,atomic_json
from transfer_close_boundary_completion import ROOT as BASE,SOURCE,FRAMES,optional_override
from refine_measured_head_masks import ROOT as MASKS

ROOT=Path('/mnt/data/dec5_measured_head_mask_completion')


def inject_refined_masks(code):
    marker="    evidence=root/'admission';evidence.mkdir(exist_ok=False)"
    if code.count(marker)!=1:
        raise ValueError('Unexpected guard preparation')
    return code.replace(marker,"    masks=load_refined_masks(FRAME,names,masks)\n"+marker)


def load_refined_masks(frame,names,original):
    import numpy as np
    root=MASKS/frame;r=read(root/'result.json');q=read(root/'request.json')
    assert r['request_sha256']==sha(root/'request.json')
    assert read(root/'cameras.json')==names
    for p,h in r['hashes'].items():
        assert sha(root/p)==h
    value=np.load(root/'masks.npz')['masks']
    assert value.shape==original.shape and value[original.astype(bool)].all()
    assert r['original_masks_preserved'] and not q['candidate_geometry_used'] and not q['heldout_used']
    for p,h in q['scripts'].items():
        assert sha(p)==h,p
    return value


def run(frame):
    root=ROOT/frame;root.mkdir(parents=True,exist_ok=False)
    mr=read(MASKS/frame/'result.json');mq=read(MASKS/frame/'request.json')
    source=read(SOURCE/frame/'request.json')
    assert mq['original_masks_sha256']==source['source_mask_sha256']
    raw_result=read(BASE/frame/'result.json')
    assert raw_result['request_sha256']==sha(BASE/frame/'request.json')
    names=['request.json','result.json',*raw_result['hashes'].keys()]
    for name in names:
        path=BASE/frame/name
        if name in raw_result['hashes']:
            assert sha(path)==raw_result['hashes'][name]
        (root/name).symlink_to(path.resolve(),target_is_directory=False)
    guard=importlib.import_module('guard_poisson_jaw_completion')
    fit=importlib.import_module('study_interpolated_poisson_jaw')
    checker=importlib.import_module('audit_interpolated_poisson_jaw')
    for module in [guard,fit,checker]:
        module.OUT=root;module.SOURCE=SOURCE/frame;module.FRAME=frame
    code=inject_refined_masks(optional_override(inspect.getsource(guard.prepare)))
    marker="    for arm,keep in [('strict',strict),('anchored',anchored)]:"
    assert code.count(marker)==1;code=code.split(marker)[0]
    scripts=[Path(__file__),Path(guard.__file__),Path(fit.__file__),Path(checker.__file__),
        Path(__file__).with_name('transfer_close_boundary_completion.py')]
    request=dict(frame=frame,matched_raw_result_sha256=sha(BASE/frame/'result.json'),
        baseline_controller_complete_sha256=sha(BASE/frame/'controller_complete.json'),
        masks_request_sha256=sha(MASKS/frame/'request.json'),masks_result_sha256=sha(MASKS/frame/'result.json'),
        admission_execution_sha256=hashlib.sha256(code.encode()).hexdigest(),
        scripts={str(p.resolve()):sha(p) for p in scripts},
        exact_same_raw_mesh=True,depth_and_certificate_thresholds_unchanged=True,
        texture_masks_changed=False,heldout_used=False,production_updated=False)
    atomic_json(root/'controller_request.json',request)
    guard.load_refined_masks=load_refined_masks
    exec(compile(code,__file__+':refined_semantics','exec'),guard.__dict__)
    guard.prepare(root);fit.prepare();checker.run()
    atomic_json(root/'controller_complete.json',dict(request_sha256=sha(root/'controller_request.json'),
        result_sha256=sha(root/'interpolated'/frame/'result.json'),
        audit_sha256=sha(root/'interpolated'/frame/'audit.json'),production_updated=False,visual_status='pending'))
    print(frame,'same-raw refined mask completion and native audit complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=FRAMES,required=True)
    a=p.parse_args();run(a.frame)
