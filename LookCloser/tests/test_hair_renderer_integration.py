"""Exercise the real CLI on two crossing translucent synthetic strands."""
import gzip
import json
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
pytest.importorskip('nerfstudio')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from hair_depth_override import digest


def test_actual_renderer_preserves_nonhair_pixels_and_composites_once(tmp_path):
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    data = tmp_path/'data'
    (data/'images').mkdir(parents=True)
    (data/'depth').mkdir()
    shape = (48, 64)
    frames, rows = [], []
    for name in ('frame_train_0', 'frame_train_1', 'frame_eval_0'):
        image = np.broadcast_to(np.array([51, 102, 153], np.uint8), (*shape, 3)).copy()
        cv2.imwrite(str(data/'images'/(name+'.exr')), image.astype(np.float32)/255)
        pose = np.diag([1., -1., -1., 1.])
        if name == 'frame_train_1':
            pose[0, 3] = .1
        frames.append(dict(file_path='images/'+name+'.exr', transform_matrix=pose.tolist()))
        path = data/'depth'/(name+'.npy.gz')
        with gzip.open(path, 'wb') as stream:
            np.save(stream, np.ones(shape, np.float32)*2)
        rows.append(dict(image='images/'+name+'.exr', depth='depth/'+name+'.npy.gz'))
    (data/'transforms.json').write_text(json.dumps(dict(camera_model='OPENCV', w=64, h=48,
        fl_x=80., fl_y=80., cx=32., cy=24., frames=frames,
        train_filenames=[f['file_path'] for f in frames[:2]],
        val_filenames=[frames[2]['file_path']], test_filenames=[frames[2]['file_path']])))
    manifest = data/'depths.json'
    manifest.write_text(json.dumps(dict(images=rows)))
    roi = np.zeros(shape, bool)
    roi[20:23, 12:52] = True
    roi[8:40, 30:33] = True
    alpha = np.ones(shape, np.float32)*.25
    alpha[20:23, 30:33] = .75
    foreground = np.zeros((*shape, 3), np.float32)
    foreground[..., 0] = alpha
    background = np.zeros_like(foreground)
    background[..., 2] = 1
    archive = tmp_path/'hair.npz'
    np.savez(archive, front_depth=np.ones(shape, np.float32)*1.5, back_depth=np.ones(shape, np.float32)*2.5,
             alpha=alpha, foreground_premultiplied=foreground, background_rgb=background,
             roi=roi, confidence=np.ones(shape, np.float32))
    override = tmp_path/'hair.json'
    override.write_text(json.dumps(dict(archive='hair.npz', archive_sha256=digest(archive),
        target_image_name='frame_eval_0.exr', depth_convention='OpenCV camera-z', color_space='scene_linear',
        base_depth_sha256=digest(data/'depth/frame_eval_0.npy.gz'), mesh_depth_manifest_sha256=digest(manifest))))
    script = Path(__file__).resolve().parents[1]/'scripts/render_mesh_image_blend.py'
    common = [sys.executable, str(script), '--data', str(data), '--mesh-depth-manifest', str(manifest),
              '--neighbors', '1', '--aggregation-modes', 'nearest-fill', '--blend-alphas', '1',
              '--orientation-method', 'none', '--center-method', 'none', '--no-auto-scale-poses',
              '--downscale-factor', '1', '--device', 'cpu', '--skip-ground-truth-copy', '--pixel-center-offset', '.5']
    for name, extra in [('base', []), ('changed', ['--hair-override-manifest', str(override)])]:
        result = subprocess.run(common+['--output-dir', str(tmp_path/name)]+extra, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout[-2000:]+result.stderr[-2000:]
    base = cv2.imread(str(tmp_path/'base/nearest_fill1/eval_pred_0000.exr'), cv2.IMREAD_UNCHANGED)[..., ::-1]
    changed = cv2.imread(str(tmp_path/'changed/nearest_fill1/eval_pred_0000.exr'), cv2.IMREAD_UNCHANGED)[..., ::-1]
    np.testing.assert_array_equal(changed[~roi], base[~roi])
    np.testing.assert_allclose(changed[roi, 0], alpha[roi], atol=1e-6)
    np.testing.assert_allclose(changed[roi, 2], 1-alpha[roi], atol=1e-6)
