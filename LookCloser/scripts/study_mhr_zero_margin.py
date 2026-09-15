"""One explicit offset-only control; frozen solver, observations and stop rule."""
from pathlib import Path
import argparse
import hashlib
import inspect
import numpy as np
import fit_mhr_silhouette_conformance as fit
import continue_mhr_silhouette_convergence as continuation
from study_multiview_face_prior import read, save, sha

ROOT = Path('/mnt/data/dec5_mhr_silhouette_zero_margin')
CONTROL = Path('/mnt/data/dec5_mhr_silhouette_convergence')
TEN = ROOT/'control10'
FINAL = ROOT/'fit100'
FROZEN_OPTIMIZE = fit.optimize
FROZEN_STATS = fit.silhouette_stats
FROZEN_MAKE = continuation.make_optimizer


def replace_exact(source, replacements):
    for old, new, count in replacements:
        assert source.count(old) == count, (old, source.count(old), count)
        source = source.replace(old, new)
    return source


def adapter(function, replacements, namespace):
    source = inspect.getsource(function)
    generated = replace_exact(source, replacements)
    exec(compile(generated, '<explicit_zero_margin_adapter>', 'exec'), namespace)
    proof = dict(original_source=source, generated_source=generated,
                 original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                 generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                 replacements=replacements)
    return namespace[function.__name__], proof


def configure():
    fit.RECIPE = dict(fit.RECIPE, boundary_tolerance_pixels=0.)
    fit.silhouette_stats, stats_proof = adapter(FROZEN_STATS, [
        ('record[1] > 2', 'record[1] > 0', 1),
        ('record[1]-2, 0', 'record[1]-0, 0', 1)], fit.__dict__)
    zero_source = replace_exact(inspect.getsource(FROZEN_OPTIMIZE), [
        ('excess = np.maximum(values-2, 0)', 'excess = np.maximum(values-0, 0)', 1)])
    fit.optimize, optimizer_proof = adapter(FROZEN_OPTIMIZE, [
        ('excess = np.maximum(values-2, 0)', 'excess = np.maximum(values-0, 0)', 1)], fit.__dict__)
    continuation.CONTROL = TEN
    continuation.ROOT = FINAL
    namespace = dict(continuation.__dict__, ZERO_SOURCE=zero_source)
    continuation.make_optimizer, observer_proof = adapter(FROZEN_MAKE, [
        ('excess = np.maximum(values-2,0)', 'excess = np.maximum(values-0,0)', 1),
        ('source = inspect.getsource(FROZEN_OPTIMIZER)', 'source = ZERO_SOURCE', 1)], namespace)
    return dict(wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
        frozen_fit_path=str(Path(fit.__file__).resolve()), frozen_fit_sha256=sha(fit.__file__),
        frozen_continuation_path=str(Path(continuation.__file__).resolve()), frozen_continuation_sha256=sha(continuation.__file__),
        old_control=str(CONTROL), old_control_fit_sha256=sha(CONTROL/'fit.npz'),
        original_base_path=str(fit.SOURCE/'smooth100/fit.npz'), original_base_sha256=sha(fit.SOURCE/'smooth100/fit.npz'),
        optimizer=optimizer_proof, diagnostics=stats_proof, observer=observer_proof,
        changed_parameter='silhouette boundary offset 2 -> 0 pixels',
        unchanged_normalization_pixels=2., unchanged_weight=4., unchanged_robust_scale_pixels=8.,
        same_original_base_reference=True, no_target_input=True, production_modified=False)


def fit_stage(steps):
    proof = configure()
    original_save = fit.save
    def annotated(path, value):
        if Path(path).name == 'protocol.json': value = dict(value, zero_margin_adapter=proof)
        original_save(path, value)
    fit.save = annotated
    try:
        if steps == 10:
            fit.ROOT = TEN
            fit.RECIPE = dict(fit.RECIPE, outer_iterations=10)
            fit.main()
        else:
            continuation.main()
    finally:
        fit.save = original_save


def review():
    configure()
    fit.ROOT = FINAL
    import review_mhr_silhouette_conformance as native
    import probe_mhr_silhouette_locality as locality
    native.main()
    locality.main()
    save(ROOT/'review_wrapper.json', dict(wrapper_sha256=sha(__file__), root=str(FINAL),
        protocol_sha256=sha(FINAL/'protocol.json'), target_used_posthoc_only=True,
        helpers={str(Path(m.__file__).resolve()):sha(m.__file__) for m in [native, locality]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit10', 'fit100', 'review'])
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.stage == 'fit10': fit_stage(10)
    elif args.stage == 'fit100': fit_stage(100)
    else: review()


if __name__ == '__main__': main()
