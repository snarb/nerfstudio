"""Versioned1680-only release adapter over the unchanged V5 native factory."""
from pathlib import Path
import argparse, sys, time
import marshawn_sam3_assist as io
import marshawn_mask_assist_full_v5 as original

ORIGINAL_START = original.start


def guard(V, frame):
    if frame != 1680:
        raise ValueError('This versioned release authorizes only the1680 canary')
    release = io.read(V / 'operations/sam3_assist/v5/canary1680_release.json')
    for path, digest in release['files_sha256'].items():
        if io.sha(path) != digest:
            raise ValueError('1680 canary release source changed: ' + path)
    gate_path = V / 'operations/v5_1675_final_visual_release.json'
    gate = io.read(gate_path)
    if (gate.get('kind') != 'ROOT_REVIEWED_V5_1675_NATIVE_SEMANTICS_RELEASE'
            or gate.get('authorized') is not True or gate.get('frame') != 1675 or gate.get('cameras') != 66
            or gate.get('future_scope') != {'canary_frame': 1680, 'continuous_authorized': False}):
        raise ValueError('Root source/W64 gate does not authorize1680-only canary')
    bound = dict(release['files_sha256'])
    for path, key in [(V / 'frames/001675/native_semantic_ready.json', 'native_semantic_ready_sha256'),
                      (V / 'frames/001675/data/prepared_ready.json', 'prepared_ready_sha256')]:
        if io.sha(path) != gate[key]:
            raise ValueError('1675 semantic input changed after visual release: ' + key)
        bound[str(path)] = gate[key]
    for row in gate.get('visual_evidence', []):
        if io.sha(row['path']) != row['sha256']:
            raise ValueError('1675 visual release evidence changed')
        bound[row['path']] = row['sha256']
    bound[str(gate_path)] = io.sha(gate_path)
    return dict(release, files_sha256=bound, future_frames_authorized=True)


def start(argv, log):
    # Keep the versioned policy adapter on every detached factory descendant.
    # The original runtime, math, input factory and queue functions are reused.
    argv = list(argv)
    if len(argv) > 1 and Path(argv[1]).resolve() == Path(original.__file__).resolve():
        argv[1] = str(Path(__file__).resolve())
    return ORIGINAL_START(argv, log)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['run', 'coordinate', 'watchdog'], nargs='?', default='run')
    p.add_argument('--campaign', type=Path, required=True); p.add_argument('--frame', type=int, required=True); a = p.parse_args()
    V = a.campaign.resolve(); guard(V, a.frame)
    original.guard = guard; original.start = start
    root = V / 'operations/sam3_assist/v5' / ('frame%06d' % a.frame); root.mkdir(parents=True, exist_ok=True)
    if a.action == 'coordinate':
        original.coordinate(V, a.frame)
    elif a.action == 'watchdog':
        original.watchdog(V, a.frame)
    else:
        fleet = original.fleet_for(V); identity = root / 'factory_watchdog_identity.json'
        if not fleet.alive(io.read(identity) if identity.exists() else None):
            start([sys.executable, __file__, 'watchdog', '--campaign', str(V), '--frame', str(a.frame)], root / 'factory_watchdog.log')
        ready = root / 'all66_controller/all66_review_ready.json'
        while not ready.exists():
            time.sleep(5)
        print(str(ready), flush=True)


if __name__ == '__main__':
    main()
