"""Opt-in ablation of a minimum-gap proposal heuristic; measured gates unchanged.

Small hole-crossing triangles can have a centroid almost on an existing face.
Rejecting them as redundant before depth validation leaves real narrow cracks.
This controller tests removal of that heuristic with the full original safety
gates, on two actual times. It does not change production defaults.
"""
import argparse
import hashlib
import importlib
import inspect
from pathlib import Path
from joint_temporal_texture import read,sha,atomic_json

ROOT=Path('/mnt/data/dec5_close_boundary_completion')
SOURCE=Path('/mnt/data/dec5_jaw_measured_mask_control')


def transform(source):
    replacements={
        'minimum_centroid_distance=.00002':'minimum_centroid_distance=0.0',
        'center_distance>=.00002':'center_distance>=0.0',
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Unexpected proposal implementation')
        source=source.replace(old,new)
    return source


def configure(frame):
    raw=importlib.import_module('study_poisson_jaw_completion')
    raw.OUT=ROOT/frame;raw.SOURCE=SOURCE/frame;raw.FRAME=frame
    guard=importlib.import_module('guard_poisson_jaw_completion')
    fit=importlib.import_module('study_interpolated_poisson_jaw')
    for module in [guard,fit]:
        module.OUT=raw.OUT;module.SOURCE=raw.SOURCE;module.FRAME=frame
    return raw,guard,fit


def run(frame,action):
    raw,guard,fit=configure(frame);output=ROOT/frame
    if action=='audit':
        audit=importlib.import_module('audit_interpolated_poisson_jaw')
        audit.OUT=output;audit.SOURCE=SOURCE/frame;audit.FRAME=frame;audit.run();return
    code=transform(inspect.getsource(raw.prepare))
    exec(compile(code,__file__+':proposal_adapter','exec'),raw.__dict__)
    raw.prepare(output)
    atomic_json(output/'controller_request.json',dict(frame=frame,
        source_sha256=sha(__file__),proposal_execution_sha256=hashlib.sha256(code.encode()).hexdigest(),
        change='Remove minimum centroid-gap heuristic only; no maximum-distance or evidence gate relaxation',
        original_proposal_script_sha256=sha(raw.__file__),
        same_measured_admission_and_native_free_space_gates=True,
        geometry_uses_target_camera=False,heldout_used=False,production_changed=False))
    # Compute identical semantic/depth/anchor evidence, skipping the two unused
    # strict/nearest-arm meshes. The interpolated arm still runs all 124 rays.
    preparation=inspect.getsource(guard.prepare)
    marker="    for arm,keep in [('strict',strict),('anchored',anchored)]:"
    if preparation.count(marker)!=1:raise ValueError('Unexpected admission implementation')
    preparation=preparation.split(marker)[0]
    exec(compile(preparation,__file__+':admission_only','exec'),guard.__dict__)
    guard.prepare(output);fit.prepare()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','audit']);p.add_argument('--frame',choices=['001193','001195'],required=True)
    args=p.parse_args();run(args.frame,args.action)
