"""External regression for zero-distortion COLMAP undistorter cropping."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_hair import mvs_camera_model
from export_nerfstudio_colmap_model import export_model
from import_colmap_mvs_depth_dataset import load_binary_pinhole_calibration
from prepare_native_colmap_workspace import prepare_native_workspace


def test_pinhole_undistorter_preserves_native_pixel_grid(tmp_path):
    colmap = shutil.which('colmap')
    if colmap is None:
        pytest.skip('COLMAP is not installed')
    data = tmp_path/'data'
    (data/'images').mkdir(parents=True)
    yy, xx = np.indices((48, 64))
    pixels = np.stack((xx*3, yy*4, (xx+yy)*2), -1).astype(np.uint8)
    frames = []
    for index in range(2):
        name = 'images/frame_train_%d.png'%index
        cv2.imwrite(str(data/name), pixels)
        pose = np.diag([1., -1., -1., 1.])
        pose[0, 3] = index*.1
        frames.append(dict(file_path=name, transform_matrix=pose.tolist()))
    payload = dict(camera_model='OPENCV', w=64, h=48, fl_x=80., fl_y=80., cx=32., cy=24.,
                   k1=0., k2=0., p1=0., p2=0., frames=frames,
                   train_filenames=[f['file_path'] for f in frames])
    (data/'transforms.json').write_text(json.dumps(payload))
    camera_model = mvs_camera_model(frames, payload)
    assert camera_model == 'PINHOLE'
    export_model(data, tmp_path/'model', split='train', camera_model=camera_model)
    prepare_native_workspace(data, tmp_path/'model', tmp_path/'dense', Path(colmap))
    calibration = load_binary_pinhole_calibration(tmp_path/'dense/sparse')
    for frame in frames:
        camera = calibration[frame['file_path']]
        assert (camera['w'], camera['h']) == (64, 48)
        np.testing.assert_allclose([camera['fl_x'],camera['fl_y'],camera['cx'],camera['cy']], [80,80,32,24])
        actual = cv2.imread(str(tmp_path/'dense/images'/frame['file_path']))
        np.testing.assert_array_equal(actual, pixels)
        for folder in ('depth_maps','normal_maps','consistency_graphs'):
            assert (tmp_path/'dense/stereo'/folder/Path(frame['file_path']).parent).is_dir()
    if os.environ.get('LOOKCLOSER_RUN_CUDA_SMOKE') == '1':
        lines = []
        for index, frame in enumerate(frames):
            lines.extend([frame['file_path'], frames[1-index]['file_path']])
        (tmp_path/'dense/stereo/patch-match.cfg').write_text('\n'.join(lines)+'\n')
        result = subprocess.run([colmap,'patch_match_stereo','--workspace_path',str(tmp_path/'dense'),
            '--workspace_format','COLMAP','--PatchMatchStereo.max_image_size','64',
            '--PatchMatchStereo.depth_min','1','--PatchMatchStereo.depth_max','3',
            '--PatchMatchStereo.num_iterations','1','--PatchMatchStereo.geom_consistency','0',
            '--PatchMatchStereo.filter','0'],capture_output=True,text=True)
        assert result.returncode == 0, result.stderr[-2000:]
        for frame in frames:
            assert (tmp_path/'dense/stereo/depth_maps'/(frame['file_path']+'.photometric.bin')).is_file()
