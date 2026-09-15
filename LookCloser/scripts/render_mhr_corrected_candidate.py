"""Matched native baseline/interpolated RGB for a terminal 001193 admission.

Uses the frozen CPU renderer and source wrappers unchanged; skips the unrequested
strict branch. This is diagnostic RGB, not a production-video publication.
"""
import argparse
import inspect
from pathlib import Path
import multiprocessing
import review_mhr_production_patch_control as renderer
import probe_mhr_conic_candidates as probe
from study_multiview_face_prior import read, save, sha


def render_view(source, root, prior, view):
    """Fresh process per view: renderer's CPU/global shims never share state."""
    root, prior = Path(root), Path(prior)
    probe.PRIOR = prior; probe.ROOT = root; probe.configure()
    assert read(root / 'request.json') == probe.control.binding()
    namespace = dict(renderer.__dict__, OUT=root / 'admission', ROOT=root,
                     ARM='certified_conic', binding=probe.control.binding)
    exec(compile(source, '<corrected-candidate-native-rgb>', 'exec'), namespace)
    namespace['render']([view])
    return view


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-root', type=Path, required=True)
    p.add_argument('--views', nargs='+', default=['F004_E', 'old_moving', 'M004_B', 'C004_E'])
    p.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = p.parse_args()
    root = args.candidate_root.resolve(); out = root / 'admission'
    assert len(root.parts) >= 4
    r = read(out / 'result.json'); q = read(out / 'request.json')
    assert q['frame'] == '001193' and r['request_sha256'] == sha(out / 'request.json')
    assert q['numerical_admission_unchanged'] and q['arms'] == ['certified_conic']
    assert set(args.views) <= {'F004_E', 'old_moving', 'M004_B', 'C004_E'}
    assert len(args.views) == len(set(args.views)) and args.views
    for name, h in q['helpers'].items():
        assert sha(Path(__file__).with_name(name)) == h, name
    assert sha(Path(__file__).with_name('admit_mhr_local_patch_depth.py')) == q['script_sha256']
    prior = Path(read(root / 'candidates/request.json')['final_prior_root']).resolve()
    probe.PRIOR = prior; probe.ROOT = root; probe.configure()
    assert read(root / 'request.json') == probe.control.binding()
    proof = out / 'rgb_adapter.json'
    assert not proof.exists() and not (out / 'rgb').exists()
    source = inspect.getsource(renderer.render)
    old = "for variant in ['baseline', 'strict', 'interpolated']:"
    assert source.count(old) == 1
    source = source.replace(old, "for variant in ['baseline','interpolated']:")
    save(proof, dict(admission_result_sha256=sha(out / 'result.json'),
        admission_request_sha256=sha(out / 'request.json'), views=args.views,
        frozen_renderer_sha256=sha(renderer.__file__), adapter_sha256=sha(__file__),
        generated_source=source, numerical_rendering_unchanged=True,
        only_branch_selection_and_process_scheduling_changed=True,
        cpu_view_workers=args.workers, isolated_spawn_processes=True, production_accepted=False))
    with multiprocessing.get_context('spawn').Pool(processes=args.workers, maxtasksperchild=1) as pool:
        jobs = [(source, str(root), str(prior), view) for view in args.views]
        for view in pool.starmap(render_view, jobs, chunksize=1):
            print('view terminal', view, flush=True)


if __name__ == '__main__':
    main()
