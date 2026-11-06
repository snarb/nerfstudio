import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def load():
    path = Path(__file__).resolve().parents[1]/'scripts/marshawn_frequency_residual_evidence.py'
    spec = importlib.util.spec_from_file_location('residual_evidence', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.bias = torch.nn.Parameter(torch.tensor([.1, .2, .3]))
        def render_masked(self, uv, max_active_level):
            assert max_active_level == 15
            return uv[:, :1].expand(-1, 3)*0+self.bias
    source = tmp_path/'source'; source.write_bytes(b'bound source')
    spec = {'image': str(source), 'postfit_evidence': {'source_image_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rois_xyxy': {'patch': [0, 0, 16, 8]}}}
    prep = SimpleNamespace(compute_ssim=lambda gt, pred, **kw: 1-(gt-pred).square().mean((1, 2, 3)), to_uint8_np=lambda x: np.clip(x*255+.5, 0, 255).astype(np.uint8))
    return Model(), torch.zeros(16, 16, 3), spec, prep


def test_readonly_capture_preserves_weights_rng_and_training_mode(tmp_path, monkeypatch):
    model, image, spec, prep = setup(tmp_path, monkeypatch)
    result = load().save_evidence(model, image, prep, spec, tmp_path)
    assert result['before'] == result['after'] and model.training
    raw = np.load(tmp_path/'residual_evidence/patch.npz')
    assert raw['ssim8'].shape == (2,)
    assert raw['prediction'].shape == (8, 16, 3)
    np.testing.assert_allclose(raw['residual'], raw['prediction']-raw['gt'])


def test_rng_change_is_detected(tmp_path, monkeypatch):
    model, image, spec, prep = setup(tmp_path, monkeypatch)
    original = prep.compute_ssim
    def bad(*args, **kw):
        torch.rand(1)
        return original(*args, **kw)
    prep.compute_ssim = bad
    with pytest.raises(ValueError, match='changed model or RNG'):
        load().save_evidence(model, image, prep, spec, tmp_path)
