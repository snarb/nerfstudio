import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from hair_depth_override import digest, load_override, apply_layers


def fixture(tmp_path):
    base = np.ones((8, 8), np.float32)*2
    depth = tmp_path/'base.gz'
    depth.write_bytes(b'base depth provenance')
    mesh = tmp_path/'mesh.json'
    mesh.write_text('{}')
    roi = np.zeros((8, 8), bool)
    roi[2:6, 2:6] = True
    proposed = base.copy()
    proposed[roi] = 1
    # A candidate may contain arbitrary values outside ROI; none can escape it.
    proposed[~roi] = 100
    archive = tmp_path/'hair.npz'
    np.savez(archive, front_depth=proposed, roi=roi, confidence=np.ones((8, 8)),
             back_depth=np.ones((8, 8))*3, alpha=np.ones((8, 8))*.25,
             foreground_premultiplied=np.broadcast_to([.25, 0, 0], (8, 8, 3)),
             background_rgb=np.broadcast_to([0., 0, 1.], (8, 8, 3)))
    record = dict(depth_convention='OpenCV camera-z', base_depth_sha256=digest(depth),
                  mesh_depth_manifest_sha256=digest(mesh), target_image_name='eval.png',
                  archive='hair.npz', archive_sha256=digest(archive), color_space='scene_linear')
    manifest = tmp_path/'override.json'
    manifest.write_text(json.dumps(record))
    return base, roi, depth, mesh, manifest


def test_hash_bound_roi_override_and_single_alpha_composite(tmp_path):
    base, roi, depth, mesh, manifest = fixture(tmp_path)
    refined, arrays, receipt = load_override(manifest, base, base_depth_path=depth,
                                           mesh_manifest_path=mesh, image_name='eval.png')
    np.testing.assert_array_equal(refined[~roi], base[~roi])
    np.testing.assert_array_equal(refined[roi], 1)
    assert receipt['changed_depth_pixels'] == 16
    rgb = np.ones((8, 8, 3), np.float32)*.5
    result, gate = apply_layers(rgb, arrays)
    np.testing.assert_array_equal(result[~roi], rgb[~roi])
    np.testing.assert_allclose(result[roi], np.tile([.25, 0, .75], (16, 1)))


def test_rejects_changed_camera_or_base_before_applying_depth(tmp_path):
    base, _, depth, mesh, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match='incompatible'):
        load_override(manifest, base, base_depth_path=depth, mesh_manifest_path=mesh, image_name='different.png')
    depth.write_bytes(b'another normalization')
    with pytest.raises(ValueError, match='incompatible'):
        load_override(manifest, base, base_depth_path=depth, mesh_manifest_path=mesh, image_name='eval.png')


def test_near_background_layer_falls_back_instead_of_depth_average(tmp_path):
    base, roi, depth, mesh, manifest = fixture(tmp_path)
    _, arrays, _ = load_override(manifest, base, base_depth_path=depth, mesh_manifest_path=mesh, image_name='eval.png')
    arrays['back_depth'][roi] = .5
    rgb = np.ones((8, 8, 3))*.3
    result, gate = apply_layers(rgb, arrays)
    np.testing.assert_array_equal(result, rgb)
    assert not gate.any()
