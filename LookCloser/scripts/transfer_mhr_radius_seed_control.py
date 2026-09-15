"""Transfer the frozen all-radius experiment onto audited reusable completions.

No new prior fit, threshold change or silent production promotion. Exact-count
adapters replace source-layout/configuration only, preserving the numerical
producer and independent exhaustive-distance replay from the001193 experiment.
"""
import argparse
import hashlib
import inspect
import multiprocessing
from pathlib import Path
import sys
from study_multiview_face_prior import read, save, sha
import run_local_mhr_completion as reusable
import run_mhr_radius_seed_control as producer
import audit_mhr_radius_seed_control as auditor


def replace_once(source, before, after):
    if source.count(before) != 1: raise ValueError('Frozen adapter mismatch: ' + before)
    return source.replace(before, after)


def prepare_transfer(source, bindings):
    config = read(source / 'config.json')
    assert config == reusable.inspect_inputs(config['spec'], source)
    a = read(source / 'audit.json'); assert a['status'] == 'passed'
    assert a['config_sha256'] == sha(source / 'config.json')
    assert a['admission_audit_sha256'] == sha(source / 'admission/audit.json')
    for rel, digest in a['inventory'].items(): assert sha(source / rel) == digest, rel
    admission, _ = reusable.configure(config)
    cq, rows, depths, masks, names, binding = admission.inputs()
    del masks, names
    spec = config['spec']
    proof = dict(frame=spec['frame'], production_mesh=spec['mesh'],
        production_mesh_sha256=sha(spec['mesh']), production_request=spec['production_request'],
        original_completion_config_sha256=sha(source / 'config.json'),
        original_completion_audit_sha256=sha(source / 'audit.json'),
        frame_configured_transfer=True, production_modified=False)
    bindings.update(config['input_hashes'])
    for path in [source / 'config.json', source / 'audit.json', Path(__file__)]:
        bindings[str(path)] = sha(path)
    return admission, proof, cq, rows, depths, binding, spec


def adapted_sources():
    run = inspect.getsource(producer.main)
    old = """    prior = Path(read(source / 'candidates/request.json')['final_prior_root'])
    probe.PRIOR = prior; probe.ROOT = source; probe.configure(); probe.control.configure()
    proof = probe.control.binding(); assert read(source / 'request.json') == proof
    cq, rows, depths, masks, names, real_binding = admission.inputs(); del masks, names"""
    run = replace_once(run, old, '    admission, proof, cq, rows, depths, real_binding, spec = prepare_transfer(source, bindings)')
    run = run.replace("source / 'admission/certified_conic'", "source / 'admission/silhouette100'")
    run = run.replace("source / 'candidates/certified_conic", "source / 'candidates/silhouette100")
    run = replace_once(run, "frame='001193'", "frame=spec['frame']")
    run = replace_once(run, 'script_sha256=sha(__file__),', 'script_sha256=sha(__file__), transfer_adapter=adapter_proof(),')
    audit = inspect.getsource(auditor.main)
    old = """    source = Path(q['source_candidate_root']); prior = Path(read(source / 'candidates/request.json')['final_prior_root'])
    probe.PRIOR = prior; probe.ROOT = source; probe.configure(); probe.control.configure()
    assert probe.control.binding() == q['production_base_binding']
    cq, rows, depths, masks, names, binding = admission.inputs(); del masks, names"""
    audit = replace_once(audit, old, """    source = Path(q['source_candidate_root'])
    admission, proof, cq, rows, depths, binding, spec = prepare_transfer(source, {})
    assert proof == q['production_base_binding'] and q['transfer_adapter'] == adapter_proof()""")
    audit = audit.replace("source / 'admission/certified_conic", "source / 'admission/silhouette100")
    audit = audit.replace("source / 'candidates/certified_conic", "source / 'candidates/silhouette100")
    for code in [run, audit]: compile(code, '<radius-frame-transfer>', 'exec')
    return run, audit


def adapter_proof():
    run, audit = adapted_sources()
    return dict(wrapper_sha256=sha(__file__), producer_sha256=sha(producer.__file__),
        auditor_sha256=sha(auditor.__file__), reusable_sha256=sha(reusable.__file__),
        generated_producer_sha256=hashlib.sha256(run.encode()).hexdigest(),
        generated_auditor_sha256=hashlib.sha256(audit.encode()).hexdigest(),
        fitting_and_numerical_thresholds_unchanged=True)


def render_view(root, view, generated):
    import review_mhr_production_patch_control as render
    root = Path(root); q = read(root / 'request.json'); proof = q['production_base_binding']
    namespace = dict(render.__dict__, ROOT=root, OUT=root / 'admission', ARM='certified_conic',
        FRAME=q['frame'], PARENT=Path(proof['production_request']).parent,
        PRODUCTION_REQUEST=proof['production_request'],
        PREVIOUS=Path(q['source_candidate_root']) / 'admission/silhouette100/interpolated/mesh.ply',
        binding=lambda: proof)
    exec(compile(generated, '<radius-transfer-rgb>', 'exec'), namespace)
    namespace['render']([view])


def render(root):
    import review_mhr_production_patch_control as old
    q = read(root / 'request.json'); r = read(root / 'admission/result.json')
    assert r['request_sha256'] == sha(root / 'request.json')
    assert q['transfer_adapter'] == adapter_proof()
    for p, h in r['hashes'].items(): assert sha(root / p) == h, p
    prepare_transfer(Path(q['source_candidate_root']), {})
    assert not (root / 'admission/rgb_adapter.json').exists()
    generated = inspect.getsource(old.render)
    generated = replace_once(generated, "moving_path = Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')",
        "moving_path = Path(PRODUCTION_REQUEST)\n    old_moving = next(r['camera'] for r in read('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')['inventory'] if r['frame_id'] == FRAME)")
    generated = replace_once(generated, "camera = moving if view == 'old_moving' else next(r for r in rows if r['physical_camera'].startswith(view))",
        "camera = moving if view == 'current_moving' else old_moving if view == 'old_moving' else next(r for r in rows if r['physical_camera'].startswith(view))")
    generated = replace_once(generated, "for variant in ['baseline', 'strict', 'interpolated']:",
        "for variant in ['baseline', 'nearest24', 'interpolated']:")
    generated = replace_once(generated, "mesh = Path(record['mesh']) if variant == 'baseline' else OUT/ARM/variant/'mesh.ply'",
        "mesh = Path(record['mesh']) if variant == 'baseline' else PREVIOUS if variant == 'nearest24' else OUT/ARM/variant/'mesh.ply'")
    views = ['current_moving', 'F004_E', 'old_moving']
    save(root / 'admission/rgb_adapter.json', dict(views=views, source=generated,
        adapter=adapter_proof(), renderer_sha256=sha(old.__file__), frame=q['frame'],
        old_moving_request_sha256=sha('/mnt/data/dec5_elevated_camera_dynamic_150/request.json'),
        request_sha256=sha(root / 'request.json'), production_accepted=False))
    with multiprocessing.get_context('spawn').Pool(2, maxtasksperchild=1) as pool:
        pool.starmap(render_view, [(str(root), v, generated) for v in views])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['produce', 'audit', 'render', 'dry-run'])
    p.add_argument('--source', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.stage == 'dry-run': print(adapter_proof())
    elif args.stage == 'render': render(args.output.resolve())
    else:
        index = 0 if args.stage == 'produce' else 1; module = [producer, auditor][index]
        if index == 0:
            assert args.source is not None
            sys.argv = [__file__, '--candidate-root', str(args.source), '--output', str(args.output)]
        else: sys.argv = [__file__, '--root', str(args.output)]
        code = adapted_sources()[index]
        namespace = dict(module.__dict__, prepare_transfer=prepare_transfer, adapter_proof=adapter_proof)
        exec(compile(code, '<radius-transfer>', 'exec'), namespace)
        namespace['main']()
