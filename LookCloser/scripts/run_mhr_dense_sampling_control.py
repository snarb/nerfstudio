"""Fixed order4/order8 quadrature controls, preserving the surface16+vertex16 fit."""
import argparse
from pathlib import Path
import run_mhr_surface_sampling_control as parent
import mhr_dense_surface_silhouette as dense


def sources(order):
    if order not in (4, 8):
        raise ValueError('Use one predeclared control: 4 or 8')
    source, driver = parent.sources('surface')
    source = 'import mhr_dense_surface_silhouette as dense_helper\n' + source
    changes = {
        "ns['surface_silhouette']=surface_helper.linearize":
        f"ns['surface_silhouette']=dense_helper.linearizer({order})",
        "'arm': 'surface'": f"'arm': 'dense{order}', 'lattice_order': {order}, 'samples_per_face': {len(dense.barycentric_lattice(order))}",
        'three edge midpoints and centroid of every face touching an active vertex':
        f'nested barycentric lattice denominator{order}, excluding vertices, plus centroid; every face touching an active vertex'}
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    old = "ROOT=Path('/mnt/data/dec5_mhr_sampling_surface')"
    assert driver.count(old) == 1
    driver = driver.replace(old, f"ROOT=Path('/mnt/data/dec5_mhr_sampling_dense{order}')")
    old = ',SURFACE_WRAPPER,SURFACE_HELPER]'
    assert driver.count(old) == 1
    driver = driver.replace(old, ',SURFACE_WRAPPER,SURFACE_HELPER,DENSE_WRAPPER,DENSE_HELPER]')
    compile(source, '<dense-surface-optimizer-adapter>', 'exec')
    compile(driver, '<dense-surface-driver-adapter>', 'exec')
    return source, driver


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--order', type=int, choices=[4, 8], required=True)
    p.add_argument('--dry-run', action='store_true'); args = p.parse_args()
    source, driver = sources(args.order)
    if args.dry_run:
        print('Validated dense control', args.order, len(dense.barycentric_lattice(args.order)), 'samples per face')
    else:
        exec(compile(driver, '<dense-surface-clearance>', 'exec'), dict(
            __name__='__main__', __file__=parent.weight.clearance.__file__, ANATOMICAL_SOURCE=source,
            WEIGHT_WRAPPER=Path(parent.weight.__file__), SURFACE_WRAPPER=Path(parent.__file__),
            SURFACE_HELPER=Path(parent.surface.__file__), DENSE_WRAPPER=Path(__file__),
            DENSE_HELPER=Path(dense.__file__)))
