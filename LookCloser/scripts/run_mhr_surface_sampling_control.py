"""Isolated surface16+vertex16 vs vertex32 silhouette controls; no defaults change."""
import argparse
from pathlib import Path
import run_mhr_silhouette_weight_control as weight
import mhr_surface_silhouette as surface


def sources(arm):
    if arm not in ('surface', 'vertex32'):
        raise ValueError(arm)
    source, driver = weight.sources()
    if arm == 'surface':
        marker = "source=source.replace(old_weight,'weights = np.sqrt(16.*robust/(len(rows)*count))/2.')"
        assert source.count(marker) == 1
        source = source.replace(marker, marker + '\n    source=surface_helper.augment_optimizer(source)')
        source = 'import mhr_surface_silhouette as surface_helper\n' + source
        marker = "ns['RECIPE']=dict(ns['RECIPE'],silhouette_weight=16.)"
        assert source.count(marker) == 1
        source = source.replace(marker, marker + "\n    ns['surface_silhouette']=surface_helper.linearize")
    else:
        assert source.count('16.') == 4
        source = source.replace('16.', '32.')
    proof = dict(arm=arm, vertex_weight=16 if arm == 'surface' else 32,
        additional_surface_weight=16 if arm == 'surface' else 0,
        sample_rule='three edge midpoints and centroid of every face touching an active vertex',
        normalization='camera count times fixed sample count; equal sample weight',
        masks_and_geometry_guards_unchanged=True, target_ray_samples_used=False,
        strength_control_not_identical_spatial_weighting=True)
    marker = 'normalization_uses_reduced_active_count=True,'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + f'surface_sampling={proof!r},')
    old = "ROOT=Path('/mnt/data/dec5_mhr_silhouette_weight16')"
    assert driver.count(old) == 1
    driver = driver.replace(old, f"ROOT=Path('/mnt/data/dec5_mhr_sampling_{arm}')")
    old = ',WEIGHT_WRAPPER]'
    assert driver.count(old) == 1
    driver = driver.replace(old, ',WEIGHT_WRAPPER,SURFACE_WRAPPER,SURFACE_HELPER]')
    compile(source, '<surface-control-anatomical>', 'exec')
    compile(driver, '<surface-control-driver>', 'exec')
    return source, driver


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['surface', 'vertex32'], required=True)
    p.add_argument('--dry-run', action='store_true'); args = p.parse_args()
    source, driver = sources(args.arm)
    if args.dry_run:
        print('Validated isolated control', args.arm)
    else:
        exec(compile(driver, '<surface-control-clearance>', 'exec'), dict(
            __name__='__main__', __file__=weight.clearance.__file__, ANATOMICAL_SOURCE=source,
            WEIGHT_WRAPPER=Path(weight.__file__), SURFACE_WRAPPER=Path(__file__),
            SURFACE_HELPER=Path(surface.__file__)))
