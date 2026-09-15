import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from plane_patch_evidence import patch_points, project_rgb, photographed_rgb, patch_ncc, evidence_decision, compare_planes


def camera():
    return dict(transform_matrix=np.eye(4), fl_x=100., fl_y=120., cx=20., cy=15.)


def test_plane_roundtrip_and_parallel_depth():
    row = camera();near = np.array([[.1, .03, -2.]])
    points, uv, valid = patch_points(row, near, np.array([[.2, .1, 1.]]), 3)
    assert valid.all()
    np.testing.assert_allclose(project_rgb(points, row)[0], uv, atol=1e-12)
    np.testing.assert_allclose(np.einsum('nki,ni->nk', points-near[:, None], np.array([[.2, .1, 1.]])), 0, atol=1e-12)
    _, uv2, valid2 = patch_points(row, near*1.2, np.array([[.2, .1, 1.]]), 3)
    np.testing.assert_allclose(uv2, uv)
    assert valid2.all()


def test_unavailable_and_flat_are_not_evidence():
    im = np.ones((20, 30, 3))
    values, good = photographed_rgb(im, np.array([[[2., 2.], [-.1, 3.]]]))
    assert not good[0] and np.isnan(values[0, 1]).all()
    assert np.isnan(patch_ncc(np.ones((1, 9, 3)), np.ones((1, 9, 3)))).all()
    _, _, valid = patch_points(camera(), np.array([[0., 0., -2.]]), np.array([[0., 0., 0.]]), 2)
    assert not valid[0]


def test_consensus_requires_same_witnesses_across_planes_and_scales():
    near = np.full((4, 3, 5), .9);far = np.full_like(near, .1)
    known = np.ones_like(near, bool);std = np.full((4, 3), 3.)
    known[0, 1, :3] = False
    near[0, 2, 3:] = .1;far[0, 2, 3:] = .9
    out = evidence_decision(near, far, known, std)
    assert out['reject_far'].tolist() == [True, False, False]
    std[:, 0] = 1.
    assert not evidence_decision(near, far, known, std)['reject_far'].any()


def test_known_plane_translation_preserves_rgb_correspondence():
    query = dict(transform_matrix=np.eye(4), fl_x=80., fl_y=80., cx=50.5, cy=40.5, physical_camera='query')
    witness = dict(query, transform_matrix=np.eye(4), physical_camera='source')
    witness['transform_matrix'][0, 3] = .2
    image = np.random.default_rng(4).uniform(0, 255, (80, 100, 3))
    other = np.zeros_like(image);other[:, :-8] = image[:, 8:]
    result = compare_planes(query, [witness], {'query': image, 'source': other}, {'source': np.zeros((80, 100))},
                            np.array([[0., 0., -2.]]), np.array([[0., 0., -2.8]]), np.array([[0., 0., 1.]]), 3)
    assert result['available'][0, 0]
    assert result['near'][0, 0] > .99999
    assert result['far'][0, 0] < .3
