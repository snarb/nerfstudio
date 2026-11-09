"""Separate zero-update source loader and three-model-view identity gate."""
import argparse
import json
import os
from pathlib import Path
import signal
from prepare_marshawn_frozen_semantic_cpu import sha, write


def run(path, execute):
    spec = json.loads(path.read_text())
    if not execute:
        print(json.dumps(dict(phase=spec['phase'], source_updates=0, optimizer_created=False,
            TRAIN_capture=False, head_loaded=False, selectors_rendered=False, CUDA_started=False)))
        return
    release = json.loads(Path(spec['root_release']).read_text())
    if release.get('reviewer') != 'root' or release.get('LC_source_parity_authorized') is not True or release['spec_sha256'].get(str(path)) != sha(path):
        raise ValueError('Exact independent root source-parity release required')
    for p, digest in release['bound_files'].items():
        if sha(p) != digest:
            raise ValueError('Bound input changed: ' + p)
    from marshawn_gpu_isolation import require_controller_lease
    lease = require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    out = Path(spec['output']); out.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Source parity deadline')))
    signal.alarm(spec['max_seconds'])
    from marshawn_LC_frozen_source import load_source
    source = load_source(spec, out, lease, release)
    checks = source['assert_unchanged']()
    source['guard_done'].set()
    write(out/'complete.json', dict(passed=True, phase=spec['phase'],
        source_checks=checks, source_FHD_proof=source['FHD_proof'],
        pipeline_digest=source['pipeline_digest'], source_updates=0, optimizer_created=False,
        TRAIN_capture=False, head_loaded=False, selectors_rendered=False,
        manifest_sha256=sha(spec['LC_source_plan']), spec_sha256=sha(path)))
    signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.spec, args.execute)
