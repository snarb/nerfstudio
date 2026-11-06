import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from marshawn_softplus_prefix_candidate import (
    SoftplusPrefixCandidate, load_reviewed_artifact, sha256, validate_geometry,
)


def geometry():
    o = np.array([[.1, .5, .1], [.5, .5, .1], [.9, .5, .1]])
    xyz = np.repeat([[.5, .5, .8]], 3, axis=0)
    depth = np.linalg.norm(xyz-o, axis=1)
    return dict(origins=o, directions=(xyz-o)/depth[:, None], witness_xyz=xyz,
                witness_depth=depth, t_start=np.full(3, .02),
                t_end=depth-np.maximum(.01, .05*depth),
                camera_idx=np.arange(3), point_idx=np.zeros(3, dtype=np.int64),
                source_track_view_count=np.full(3, 3, dtype=np.int64),
                camera_names=np.array(['AA', 'AB', 'AC']))


BOX = [[0., 0., 0.], [1., 1., 1.]]
CAMS = ['AA', 'AB', 'AC']


class Field(torch.nn.Module):
    density_activation = 'softplus'

    def __init__(self, raw=2.):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor(raw))
        self.aabb = torch.tensor(BOX)

    def density_fn(self, points):
        return torch.nn.functional.softplus(self.raw+1.).expand(len(points), 1)


def test_optical_integral_matches_native_softplus_and_finite_difference():
    obj = SoftplusPrefixCandidate(geometry(), BOX, CAMS, rays=6, samples=64)
    field = Field()
    loss, diagnostic = obj(field, 100)
    length = obj.sample(100)['length'].mean()
    expected = .01*length*torch.nn.functional.softplus(field.raw+1.)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    loss.backward()
    expected_gradient = .01*length*torch.sigmoid(field.raw.detach()+1.)
    assert field.raw.grad.item() == pytest.approx(expected_gradient.item(), rel=1e-6)
    eps = .001
    plus = obj(Field(2.+eps), 100)[0].item()
    minus = obj(Field(2.-eps), 100)[0].item()
    assert (plus-minus)/(2*eps) == pytest.approx(field.raw.grad.item(), rel=.001)
    assert diagnostic['prefix_tau_mean'] > 0


def test_sampling_balanced_repeatable_and_preserves_global_rng():
    obj = SoftplusPrefixCandidate(geometry(), BOX, CAMS, rays=128)
    rng = torch.random.get_rng_state().clone()
    first = obj.sample(4096)
    second = obj.sample(4096)
    assert torch.equal(first['points'], second['points'])
    assert torch.equal(rng, torch.random.get_rng_state())
    counts = torch.bincount(first['camera_groups'])
    assert (counts.max()-counts.min()).item() == 1
    assert not torch.equal(first['points'], obj.sample(4097)['points'])


def test_geometry_rejects_wrong_frame_unqualified_camera_and_surface_crossing():
    arr = geometry()
    arr['witness_xyz'] = arr['witness_xyz']+.1
    with pytest.raises(ValueError, match='coordinate chain'):
        validate_geometry(arr, BOX, CAMS)
    arr = geometry()
    arr['t_end'] = arr['witness_depth']-.001
    with pytest.raises(ValueError, match='uncertainty margin'):
        validate_geometry(arr, BOX, CAMS)
    with pytest.raises(ValueError, match='TRAIN camera'):
        validate_geometry(geometry(), BOX, ['AA', 'AB', 'HC'])
    arr = geometry()
    arr['point_idx'][2] = 1
    with pytest.raises(ValueError, match='three original TRAIN'):
        validate_geometry(arr, BOX, CAMS)


def test_runtime_activation_bounds_and_nonfinite_guards():
    obj = SoftplusPrefixCandidate(geometry(), BOX, CAMS)
    field = Field()
    field.density_activation = 'trunc_exp'
    with pytest.raises(ValueError, match='softplus only'):
        obj(field, 1)
    field = Field()
    field.aabb[1, 0] = 2.
    with pytest.raises(ValueError, match='AABB'):
        obj(field, 1)
    field = Field(float('nan'))
    with pytest.raises(FloatingPointError):
        obj(field, 1)


def test_review_and_hash_bound_artifact_refuses_drift(tmp_path):
    archive = tmp_path/'prefixes.npz'
    np.savez_compressed(archive, **geometry())
    bindings = {k: str(i)*64 for i, k in enumerate([
        'transforms_sha256', 'box_sha256', 'tracks_sha256', 'witnesses_sha256',
        'semantic_source_sha256', 'visibility_review_sha256'])}
    meta = dict(kind='reviewed_train_clothing_visible_prefixes',
                coordinate_system='accepted_floor_world', uses_eval_supervision=False,
                original_observed_track_graph_only=True, manual_clothing_visibility_review=True,
                distance_scale=1., bindings=bindings, aabb=BOX,
                archive=archive.name, archive_sha256=sha256(archive))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(meta))
    _, receipt = load_reviewed_artifact(manifest, expected_bindings=bindings,
                                        bounds=BOX, train_cameras=CAMS)
    assert receipt['observations'] == 3
    meta['uses_eval_supervision'] = True
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='TRAIN-only'):
        load_reviewed_artifact(manifest, expected_bindings=bindings, bounds=BOX, train_cameras=CAMS)
    meta['uses_eval_supervision'] = False
    manifest.write_text(json.dumps(meta))
    archive.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        load_reviewed_artifact(manifest, expected_bindings=bindings, bounds=BOX, train_cameras=CAMS)
