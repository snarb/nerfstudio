"""Transfer the frozen observed-neighborhood jaw completion to time 000995.

No per-frame threshold adjustment or mask override. Existing late-time meshes
are reused only for subsequent wider-camera validation, not treated as a full
temporal campaign. Every added surface remains a disclosed geometric prior.
"""
from pathlib import Path
import hashlib
import importlib
import inspect
from joint_temporal_texture import read, sha, atomic_json
from study_confidence_depth_prior import load_real
from study_close_boundary_completion import transform
from transfer_close_boundary_completion import optional_override

ROOT = Path('/mnt/data/dec5_midsequence_jaw_completion')
FRAME = '000995'
VIDEO = Path('/mnt/data/dec5_large_motion_choices_v3/left_high_arc')
DEPTH_ROOT = Path('/mnt/data/dec5_full_block_transfer')


def run():
    root = ROOT/FRAME; source = root/'source'; output = root/'geometry'
    root.mkdir(parents=True, exist_ok=False); source.mkdir()
    entry = next(r for r in read(VIDEO/'request.json')['inventory'] if r['frame_id'] == FRAME)
    previous = next(r for r in read('/mnt/data/dec5_phase30_early_texture_dynamic_150/request.json')['inventory'] if r['frame_id'] == FRAME)
    assert entry['mesh_sha256'] == previous['mesh_sha256'] == sha(entry['mesh'])
    assert entry['source_masks'] == previous['source_masks']
    masks = Path(entry['source_masks']['root'])
    assert sha(masks/'masks.npz') == entry['source_masks']['masks_sha256']
    rows, _, receipt = load_real(DEPTH_ROOT, FRAME)
    assert len({r['physical_camera'] for r in rows}) == 62
    atomic_json(source/'request.json', dict(frame=FRAME, source_mesh=entry['mesh'],
        source_mesh_sha256=entry['mesh_sha256'], depth_root=str(DEPTH_ROOT), depth_receipt=receipt,
        source_mask_sha256=sha(masks/'masks.npz'), source_mask_names_sha256=sha(masks/'cameras.json'),
        source_video_request_sha256=sha(VIDEO/'request.json'), mask_override_used=False))
    modules = [importlib.import_module(n) for n in ['study_poisson_jaw_completion',
        'guard_poisson_jaw_completion', 'study_interpolated_poisson_jaw', 'audit_interpolated_poisson_jaw']]
    raw, guard, fit, audit = modules
    for module in modules: module.OUT=output; module.SOURCE=source; module.FRAME=FRAME
    proposal = transform(inspect.getsource(raw.prepare))
    admission = optional_override(inspect.getsource(guard.prepare))
    marker = "    for arm,keep in [('strict',strict),('anchored',anchored)]:"
    assert admission.count(marker) == 1
    admission = admission.split(marker)[0]
    scripts = [Path(__file__), *[Path(m.__file__) for m in modules],
        Path(__file__).with_name('study_close_boundary_completion.py'),
        Path(__file__).with_name('transfer_close_boundary_completion.py')]
    atomic_json(root/'request.json', dict(frame=FRAME, source_request_sha256=sha(source/'request.json'),
        scripts={str(p): sha(p) for p in scripts},
        proposal_execution_sha256=hashlib.sha256(proposal.encode()).hexdigest(),
        admission_execution_sha256=hashlib.sha256(admission.encode()).hexdigest(),
        thresholds_unchanged=True, minimum_centroid_gap=0., geometry_uses_target=False,
        heldout_used=False, original_source_masks=True, inferred_geometry=True,
        production_changed=False))
    exec(compile(proposal, __file__+':proposal', 'exec'), raw.__dict__)
    exec(compile(admission, __file__+':admission', 'exec'), guard.__dict__)
    raw.prepare(output); guard.prepare(output); fit.prepare(); audit.run()
    final = output/'interpolated'/FRAME
    atomic_json(root/'complete.json', dict(request_sha256=sha(root/'request.json'),
        result_sha256=sha(final/'result.json'), audit_sha256=sha(final/'audit.json'),
        mesh=str(final/'mesh.ply'), mesh_sha256=sha(final/'mesh.ply'),
        visual_status='pending', production_changed=False))


if __name__ == '__main__': run()
