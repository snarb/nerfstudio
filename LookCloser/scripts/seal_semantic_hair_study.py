"""Independent finite-study audit; does not approve production geometry/video."""
import argparse
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from build_train_hair_semantics import ROOT as SEM, read, sha, write
from study_semantic_hair_sources import BASE, ROOT as PILOT, FRAMES
from study_semantic_hair_temporal import ROOT as TEMPORAL, FRAMES as TIMES

HELD = {'F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'}
SCRIPTS = ['build_train_hair_semantics.py', 'study_semantic_hair_sources.py',
           'study_semantic_hair_temporal.py', 'seal_semantic_hair_study.py']


def verify():
    import render_smooth_temporal_mesh_video as engine
    from joint_temporal_texture import ROOT as COLOR, cameras
    source_hashes = {}
    for root, frames in [(SEM, FRAMES), (TEMPORAL/'semantics', TIMES)]:
        request = read(root/'request.json')
        assert request['frames'] == frames and not request['heldout_used']
        assert not request['mesh_or_target_view_used'] and not request['geometry_changed']
        assert request['script_sha256'] == sha(Path(__file__).with_name(SCRIPTS[0]))
        assert request['profiles_sha256'] == sha(COLOR/'parameters.npz')
        assert request['exposure_sha256'] == sha(COLOR/'exposure.json')
        for item in request['models'].values():
            assert sha(item['file']) == item['sha256']
        for frame in frames:
            folder = root/frame
            stage = read(folder/'stage.json'); complete = read(folder/'complete.json')
            assert stage['request_sha256'] == complete['request_sha256'] == sha(root/'request.json')
            assert complete['stage_sha256'] == sha(folder/'stage.json')
            names = [r['camera'] for r in stage['records']]
            rows, _, _ = cameras(frame)
            assert len(names) == len(set(names)) == 62 and not (set(names) & HELD)
            assert names == [r['physical_camera'] for r in rows]
            assert len(complete['records']) == 62
            for item, result, row in zip(stage['records'], complete['records'], rows):
                assert item['source_path'] == row['file_path']
                assert result['camera'] == item['camera']
                assert result['request_sha256'] == sha(root/'request.json')
                assert sha(item['input_path']) == item['input_sha256'] == result['input_sha256']
                path = item['source_path']
                if path not in source_hashes:
                    source_hashes[path] = sha(path)
                assert source_hashes[path] == item['source_sha256']
                pred = folder/'predictions'/item['camera']
                assert read(pred.with_suffix('.json')) == result
                assert sha(pred.with_suffix('.npz')) == result['output_sha256']
                with np.load(pred.with_suffix('.npz')) as archive:
                    values = archive['confidence']
                    assert values.shape == (3, 1250, 1080) and values.dtype == np.uint8
            print('semantic audit', frame, flush=True)
    summaries = []
    for root, semantics, frames in [(PILOT, SEM, FRAMES), (TEMPORAL/'render', TEMPORAL/'semantics', TIMES)]:
        request = engine.verify_request(root)
        assert not request['geometry_changed'] and not request['production_promoted']
        assert request['matched_parent_sha256'] == sha(BASE/'request.json')
        assert request['semantic_request_sha256'] == sha(semantics/'request.json')
        assert [r['frame_id'] for r in request['inventory']] == frames
        for frame in frames:
            assert request['semantic_frame_bindings'][frame] == sha(semantics/frame/'complete.json')
            old, new = BASE/'frames'/frame, root/'frames'/frame
            for parent, folder in [(BASE, old), (root, new)]:
                receipt = read(folder/'complete.json')
                assert receipt['request_sha256'] == sha(parent/'request.json')
                for name, digest in receipt['hashes'].items():
                    assert sha(folder/name) == digest
            assert 'semantic_decisions.npz' in read(new/'complete.json')['hashes']
            for name, key in [('target_depth.npz', 'depth'), ('face_source_labels.npy', None)]:
                a, b = np.load(old/name), np.load(new/name)
                np.testing.assert_array_equal(a[key] if key else a, b[key] if key else b)
                if key:
                    a.close(); b.close()
            ids = [np.asarray(Image.open(p/'source_ids.png')) for p in [old, new]]
            changed = ids[0] != ids[1]
            np.testing.assert_array_equal(ids[0] == 255, ids[1] == 255)
            with np.load(new/'semantic_decisions.npz') as decision:
                np.testing.assert_array_equal(changed, decision['changed'].astype(bool))
                assert not (changed & ~decision['gate'].astype(bool)).any()
                assert not (changed & (decision['protected_votes'] >= 3)).any()
            rgb = [np.asarray(Image.open(p/'prediction_native.png')) for p in [old, new]]
            assert all(x.shape == (1080, 1920, 3) and x.dtype == np.uint8 for x in rgb)
            np.testing.assert_array_equal(rgb[0][~changed], rgb[1][~changed])
            summaries.append(dict(root=str(root), frame=frame, changed=int(changed.sum()),
                new_zero_rgb=int(((rgb[0].max(2)>0)&(rgb[1].max(2)==0)).sum())))
    # The overlapping time is an independent repeat with the same frozen inputs.
    for name in ['frame.png', 'source_ids.png', 'prediction_native.png', 'face_source_labels.npy']:
        assert sha(PILOT/'frames/001083'/name) == sha(TEMPORAL/'render/frames/001083'/name)
    config = read(TEMPORAL/'configuration.json')
    for name, digest in config['script_hashes'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    old = Path('/mnt/data/dec5_semantic_hair_temporal')
    assert sha(old/'controller_original.py') == read(old/'configuration.json')['script_hashes']['study_semantic_hair_temporal.py']
    package = read(TEMPORAL/'review/package.json'); movie = TEMPORAL/'review/diagnostic_12f.mp4'
    assert package['frames'] == TIMES and package['movie_sha256'] == sha(movie)
    inventory = [r for r in read(BASE/'request.json')['inventory'] if r['frame_id'] in TIMES]
    positions = np.array([r['camera']['transform_matrix'] for r in inventory])[:, :3, 3]
    assert len(inventory) == len(set(TIMES)) == len(np.unique(positions, axis=0)) == 12
    env = dict(os.environ, LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    import json
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-of', 'json', str(movie)], env=env))['streams'][0]
    assert (probe['width'], probe['height'], probe['r_frame_rate'], probe['nb_read_frames']) == (1080, 1920, '24/1', '12')
    assert abs(float(probe['duration'])-.5) < .001
    decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(movie),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], env=env)
    arrays = np.frombuffer(decoded, np.uint8).reshape(12, 1920, 1080, 3)
    assert all(np.any(x) for x in arrays)
    return summaries


def seal():
    summaries = verify()
    images = sorted((SEM/'review').glob('*.png')) + sorted((PILOT/'review').glob('*.png'))
    images += sorted((TEMPORAL/'review').glob('*.png'))
    images += [TEMPORAL/'render/review'/name for name in ['001075_crown.png', '001077_face.png', '001093_crown.png']]
    assert len(images) == 23
    write(PILOT/'visual_review.json', dict(reviewer='main LLM',
        inspected_images={str(p): sha(p) for p in images},
        status='partial_texture_improvement_not_artifact_free',
        findings='Reduced tan hair fringe in both pilots and twelve consecutive times; no new broad face/neck change seen. Jagged crown geometry, stretched hair, residual brown regions and existing face lines remain.',
        temporal_scope='Twelve distinct times and camera poses, 0.5 seconds. Consecutive PNG/contact inspection; not continuous movie playback. Fine temporal flicker and full-video quality not approved.',
        black_pixels='Seven newly zero RGB pixels across twelve times; unchanged missing-source masks. Only pilot 001083 sample traced to negative source RGB interpolation/clamping.',
        production_promoted=False, geometry_repaired=False, independent_checks=summaries))
    bindings = {}
    for root in [SEM, PILOT, TEMPORAL]:
        for path in root.rglob('*'):
            if path.is_file() and path.name != 'artifact_manifest.json':
                bindings[str(path)] = sha(path)
    for name in SCRIPTS:
        path = Path(__file__).with_name(name).resolve(); bindings[str(path)] = sha(path)
    for name in ['tests/test_semantic_hair_sources.py', 'experiments/dec5_semantic_hair_sources.md']:
        path = Path(__file__).resolve().parents[1]/name; bindings[str(path)] = sha(path)
    write(PILOT/'artifact_manifest.json', dict(status='finite_experiment_complete_full_goal_not_achieved', bindings=bindings))
    print('sealed', len(bindings), 'bindings', flush=True)


def check():
    verify()
    manifest = read(PILOT/'artifact_manifest.json')
    for path, digest in manifest['bindings'].items():
        assert sha(path) == digest, path
    print('checked', len(manifest['bindings']), 'bindings', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['seal', 'check'])
    seal() if parser.parse_args().action == 'seal' else check()
