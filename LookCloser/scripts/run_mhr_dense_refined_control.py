"""Fresh dense8 fit after the saved failed cone problem passes tighter refinement."""
from pathlib import Path
import run_mhr_dense_sampling_control as dense_driver
import refined_conic_surface_step as backend
from study_multiview_face_prior import read, sha


if __name__ == '__main__':
    replay = Path('/mnt/data/dec5_mhr_dense8_solver_precision')
    result = read(replay / 'result.json'); request = read(replay / 'request.json')
    assert result['request_sha256'] == sha(replay / 'request.json')
    for p,h in request['inputs'].items(): assert sha(p) == h, p
    control = next(r for r in result['records'] if r['name'] == 'tighter_refinement')
    assert control['status'] == 'accepted_by_unchanged_certificate'
    assert control['settings'] == backend.SETTINGS
    source, driver = dense_driver.sources(8)
    old = 'import certified_conic_surface_step as backend'
    assert source.count(old) == 1
    source = source.replace(old, 'import refined_conic_surface_step as backend')
    old = "ROOT=Path('/mnt/data/dec5_mhr_sampling_dense8')"
    assert driver.count(old) == 1
    driver = driver.replace(old, "ROOT=Path('/mnt/data/dec5_mhr_sampling_dense8_refined')")
    old = ',DENSE_WRAPPER,DENSE_HELPER]'
    assert driver.count(old) == 1
    driver = driver.replace(old, ',DENSE_WRAPPER,DENSE_HELPER,REFINED_WRAPPER,PRECISION_REPLAY]')
    parent = dense_driver.parent
    exec(compile(driver, '<dense8-refined-clearance>', 'exec'), dict(
        __name__='__main__', __file__=parent.weight.clearance.__file__, ANATOMICAL_SOURCE=source,
        WEIGHT_WRAPPER=Path(parent.weight.__file__), SURFACE_WRAPPER=Path(parent.__file__),
        SURFACE_HELPER=Path(parent.surface.__file__), DENSE_WRAPPER=Path(dense_driver.__file__),
        DENSE_HELPER=Path(dense_driver.dense.__file__), REFINED_WRAPPER=Path(__file__),
        PRECISION_REPLAY=replay / 'result.json'))
