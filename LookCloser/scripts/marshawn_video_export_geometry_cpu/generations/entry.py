"""I/O-only entry for unchanged GPU/CPU shaders and explicit generation roots."""
from pathlib import Path
import argparse
import importlib.util
import runpy
import sys

from policy import CPU_SHA, GPU_SHA, PROFILE_SHA, bound, manifest_row, read, validate_native


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--dev3-entry', type=Path)
    p.add_argument('--dev3-sha')
    a, rest = p.parse_known_args()
    rest = rest[1:] if rest[:1] == ['--'] else rest
    mode = rest[0]
    if mode not in ('gpu', 'cpu', 'gt'):
        raise ValueError('Explicit GPU/CPU/GT mode required')
    spec_path = Path(rest[rest.index('--spec') + 1])
    spec = read(spec_path)
    index = int(rest[rest.index('--index') + 1])
    code = Path(rest[rest.index('--code') + 1])
    sys.path[:0] = [str(code), str(code / 'LookCloser/scripts'), str(a.worker.parent)]
    if mode == 'gt':
        if not 72 <= index < 120:
            raise ValueError('Frozen actual-GT branch only accepts indices72..119')
        # Actual2020 photographs use common source transforms, no2020 dataset or
        # geometry. The globally hashed spec stays identical across all120 rows.
        bound(a.manifest, spec['generation_manifest_sha256'])
        bound(a.worker, '57734bb49579e465269e0c00e8ce343936998b77c30a4871ca91feda5bca1c93')
        sys.argv = [str(a.worker), '--cpu', *rest[1:]]
        runpy.run_path(str(a.worker), run_name='__main__')
        return
    _, row = manifest_row(a.manifest, index, spec=spec, check_local_inputs=mode == 'cpu')
    if mode == 'gpu':
        directory = validate_native(row)
        if code != directory.parents[1] / 'code':
            raise ValueError('GPU model code root differs from actual native campaign')
        bound(a.worker, GPU_SHA)
        # Keep the original loader's native YAML/checkpoint/dataparser checks.
        # The narrow root extension follows frame-specific validation above.
        import cap_loader
        original = cap_loader.load_pipeline
        cap_loader.ROOTS = {directory.parents[1]}
        def load(frame_root, *args, **kwargs):
            if Path(frame_root).resolve() != directory.resolve():
                raise ValueError('Shader attempted another model frame')
            return original(frame_root, *args, **kwargs)
        cap_loader.load_pipeline = load
        if a.dev3_entry:
            bound(a.dev3_entry, a.dev3_sha)
            sys.argv = [str(a.dev3_entry), '--worker', str(a.worker),
                        '--expected-sha256', GPU_SHA, '--', *rest]
            runpy.run_path(str(a.dev3_entry), run_name='__main__')
        else:
            sys.argv = [str(a.worker), *rest]
            runpy.run_path(str(a.worker), run_name='__main__')
        return
    bound(a.worker, CPU_SHA)
    bound(a.worker.parent / 'hybrid_params.json', PROFILE_SHA)
    worker = module(a.worker, 'unchanged_generation_cpu_shader')
    original_read = worker.read
    def resolved_read(path):
        value = original_read(path)
        if Path(path) == spec_path:
            if value['data_template'] != '/__generation_entry_required__/{frame_id}/data':
                raise ValueError('Generation-aware spec sentinel missing')
            value = dict(value, data_template=row['local_data_template'])
        return value
    worker.read = resolved_read
    # The original shader reads its own immutable profile/release and uses the
    # actual spec_path for receipts; no completion metadata is restamped.
    args = argparse.Namespace(code=code, spec=spec_path,
                              path=Path(rest[rest.index('--path') + 1]), index=index)
    worker.cpu(args)


if __name__ == '__main__':
    main()
