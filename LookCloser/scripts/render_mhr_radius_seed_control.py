"""Matched two-view RGB for the explicit all-radius certificate experiment."""
import argparse
import inspect
import multiprocessing
from pathlib import Path
from study_multiview_face_prior import read, save, sha
import review_mhr_production_patch_control as renderer


def render_view(root, view, source):
    root = Path(root); q = read(root / 'request.json')
    namespace = dict(renderer.__dict__, OUT=root / 'admission', ROOT=root,
        ARM='certified_conic', binding=lambda: q['production_base_binding'])
    exec(compile(source, '<all-radius-current-recipe-rgb>', 'exec'), namespace)
    namespace['render']([view])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); args = p.parse_args()
    root = args.root.resolve(); q = read(root / 'request.json'); r = read(root / 'admission/result.json')
    assert r['request_sha256'] == sha(root / 'request.json')
    assert q['policy'] == 'all_radius_then_normal_filter' and not q['target_used']
    for rel, digest in r['hashes'].items(): assert sha(root / rel) == digest, rel
    for path, digest in q['source_inputs'].items(): assert sha(path) == digest, path
    proof = q['production_base_binding']
    assert sha(proof['production_mesh']) == proof['production_mesh_sha256']
    assert sha(Path(__file__).with_name('run_mhr_radius_seed_control.py')) == q['script_sha256']
    assert not (root / 'admission/rgb_adapter.json').exists() and not (root / 'admission/rgb').exists()
    source = inspect.getsource(renderer.render)
    old = "for variant in ['baseline', 'strict', 'interpolated']:"; assert source.count(old) == 1
    source = source.replace(old, "for variant in ['baseline', 'interpolated']:")
    views = ['F004_E', 'old_moving']
    save(root / 'admission/rgb_adapter.json', dict(views=views, generated_source=source,
        renderer_sha256=sha(renderer.__file__), adapter_sha256=sha(__file__),
        request_sha256=sha(root / 'request.json'), admission_result_sha256=sha(root / 'admission/result.json'),
        numerical_rendering_unchanged=True, certificate_policy_changed=True, production_accepted=False))
    with multiprocessing.get_context('spawn').Pool(2, maxtasksperchild=1) as pool:
        pool.starmap(render_view, [(str(root), v, source) for v in views])


if __name__ == '__main__': main()
