"""Transfer narrow-crack Poisson completion to two earlier measured times.

Reuse frozen proposal/evidence/certificate/native-ray gates. Unlike the late
diagnostic frames, these controls use original source masks without an override.
The optional-override adapter changes no vote or depth threshold.
"""
import argparse
import hashlib
import importlib
import inspect
from pathlib import Path
from joint_temporal_texture import read, sha, atomic_json
from study_close_boundary_completion import transform

ROOT = Path('/mnt/data/dec5_close_boundary_transfer')
SOURCE = Path('/mnt/data/dec5_jaw_repair_transfer')
MOVIE = Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
FRAMES = ['001083', '001123']


def optional_override(code):
    start = "    override=Path(base['mask_override']['root'])/FRAME"
    end = "    evidence=root/'admission';evidence.mkdir(exist_ok=False)"
    if code.count(start) != 1 or code.count(end) != 1:
        raise ValueError('Unexpected frozen semantic preparation')
    a,b = code.index(start),code.index(end)
    block = code[a:b]
    return code[:a]+"    if 'mask_override' in base:\n"+''.join('    '+line for line in block.splitlines(keepends=True))+code[b:]


def configure(frame):
    if frame not in FRAMES:
        raise ValueError('Unsupported transfer frame')
    modules = [importlib.import_module(n) for n in ['study_poisson_jaw_completion',
        'guard_poisson_jaw_completion','study_interpolated_poisson_jaw','audit_interpolated_poisson_jaw']]
    for module in modules:
        module.OUT = ROOT/frame;module.SOURCE = SOURCE/frame;module.FRAME = frame
    return modules


def run(frame, audit=False):
    raw,guard,fit,checker = configure(frame)
    if audit:
        checker.run();return
    ROOT.mkdir(exist_ok=True)
    source = read(SOURCE/frame/'request.json')
    movie = next(r for r in read(MOVIE/'request.json')['inventory'] if r['frame_id'] == frame)
    assert source['source_mesh_sha256'] == movie['mesh_sha256'] == sha(movie['mesh'])
    assert 'mask_override' not in source
    assert source['source_mask_sha256'] == movie['source_masks']['masks_sha256']
    proposal_code = transform(inspect.getsource(raw.prepare))
    admission_code = optional_override(inspect.getsource(guard.prepare))
    marker = "    for arm,keep in [('strict',strict),('anchored',anchored)]:"
    if admission_code.count(marker) != 1:
        raise ValueError('Unexpected guard entrypoint')
    admission_code = admission_code.split(marker)[0]
    scripts = [Path(__file__), Path(raw.__file__),Path(guard.__file__),Path(fit.__file__),
        Path(checker.__file__),Path(__file__).with_name('study_close_boundary_completion.py')]
    request = dict(frame=frame, source_request_sha256=sha(SOURCE/frame/'request.json'),
        movie_request_sha256=sha(MOVIE/'request.json'), source_mesh_sha256=movie['mesh_sha256'],
        scripts={str(p.resolve()):sha(p) for p in scripts},
        proposal_execution_sha256=hashlib.sha256(proposal_code.encode()).hexdigest(),
        admission_execution_sha256=hashlib.sha256(admission_code.encode()).hexdigest(),
        min_centroid_gap=0., thresholds_unchanged_from_close_boundary=True,
        source_masks='original, no measured override at these two times',
        uses_target_for_geometry=False, uses_eval_rgb=False, production_updated=False,
        purpose='Generalization control, not an accepted full temporal reconstruction')
    reqpath=ROOT/(frame+'_controller_request.json')
    if reqpath.exists():
        raise ValueError('Existing transfer attempt; inspect it rather than restart or overwrite')
    atomic_json(reqpath,request)
    exec(compile(proposal_code,__file__+':proposal','exec'),raw.__dict__)
    exec(compile(admission_code,__file__+':admission','exec'),guard.__dict__)
    raw.prepare(ROOT/frame);guard.prepare(ROOT/frame);fit.prepare()
    checker.run()
    atomic_json(ROOT/frame/'controller_complete.json',dict(request_sha256=sha(reqpath),
        result_sha256=sha(ROOT/frame/'interpolated'/frame/'result.json'),
        audit_sha256=sha(ROOT/frame/'interpolated'/frame/'audit.json'),
        independently_replayed_depth_and_certificates=True, visual_status='pending', production_updated=False))
    print(frame,'geometry and independent audit complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=FRAMES,required=True)
    p.add_argument('--audit',action='store_true');a=p.parse_args();run(a.frame,a.audit)
