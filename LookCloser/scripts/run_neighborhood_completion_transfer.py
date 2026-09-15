"""Opt-in frozen-method transfer on times with audited native depth/mask inputs.

Prototype modules are configured in this isolated worker before their dependent
imports. No old request, producer source or production default is rewritten.
"""
from pathlib import Path
from copy import deepcopy
import argparse,importlib
from joint_temporal_texture import read,sha,atomic_json,cameras

SOURCE=Path('/mnt/data/dec5_jaw_measured_mask_control')
ROOT=Path('/mnt/data/dec5_neighborhood_completion_transfer')


def configure(frame):
    source=SOURCE/frame
    req=read(source/'request.json')
    if req['frame']!=frame:raise ValueError('Input time mismatch')
    raw=importlib.import_module('study_poisson_jaw_completion')
    raw.OUT=ROOT/frame;raw.SOURCE=source;raw.FRAME=frame
    guard=importlib.import_module('guard_poisson_jaw_completion')
    fit=importlib.import_module('study_interpolated_poisson_jaw')
    for module in [guard,fit]:module.OUT=raw.OUT;module.SOURCE=source;module.FRAME=frame
    return raw,guard,fit


def run(frame,action):
    raw,guard,fit=configure(frame);output=ROOT/frame
    if action=='prepare':
        raw.prepare(output);guard.prepare(output);fit.prepare();return
    if action=='audit':
        module=importlib.import_module('audit_interpolated_poisson_jaw')
        module.OUT=output;module.SOURCE=SOURCE/frame;module.FRAME=frame;module.run();return
    import render_smooth_temporal_mesh_video as renderer
    from study_native_texture_footprint import install
    implementation=install();renderer.torch.set_num_threads(2)
    parent=renderer.verify_request(Path('/mnt/data/dec5_phase30_early_texture_dynamic_150'))
    mesh=output/'interpolated'/frame/'mesh.ply';result=read(mesh.parent/'result.json')
    if sha(mesh)!=result['hashes']['mesh.ply'] or not result['observed_guard_passed']:raise ValueError('Unverified candidate')
    rows,_,_=cameras(frame)
    for view in ['moving','F004_E005_1210FP']:
        for variant in ['baseline','repaired']:
            request=deepcopy(parent);request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame]
            row=request['inventory'][0]
            if view!='moving':row['camera']=next(r for r in rows if r['physical_camera']==view)
            if variant=='repaired':row.update(mesh=str(mesh),mesh_sha256=sha(mesh))
            request.update(partial_diagnostic_only=True,full_video_candidate=False,frozen_neighborhood_transfer=True,
                geometry_result_sha256=sha(mesh.parent/'result.json'),native_footprint_implementation_sha256=implementation,
                same_footprint_for_both_geometry_variants=True)
            for name in ['run_neighborhood_completion_transfer.py','study_native_texture_footprint.py','native_texture_footprint.py']:
                request['script_hashes'][name]=sha(Path(__file__).with_name(name))
            dest=output/'rgb'/view/variant;dest.mkdir(parents=True,exist_ok=False);(dest/'frames').mkdir()
            atomic_json(dest/'request.json',request);renderer.render(dest,[frame])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render','audit'])
    p.add_argument('--frame',choices=['001083','001123','001193','001195'],required=True);a=p.parse_args();run(a.frame,a.action)
