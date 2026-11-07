import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


spec = importlib.util.spec_from_file_location('point_probe', Path(__file__).parents[1] / 'scripts/probe_marshawn_sourcepoint_counterfactual.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def arrays(root, sigma):
    folder = root / 'test_trace/rays'
    folder.mkdir(parents=True)
    np.savez(folder / 'pixel0000_uniform2x.npz', xyz=np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32), sigma=np.array(sigma, dtype=np.float32))


def test_actual_density_removal_and_admitted_identity(tmp_path):
    arrays(tmp_path / 'first', [2., 3.])
    arrays(tmp_path / 'changed', [2., 0.])
    result = probe.verify_density_restriction(tmp_path / 'first', tmp_path / 'changed', {'test': ''}, lambda p: p[:, 0] == 0, torch.device('cpu'))
    assert result['passed'] and result['records'][0]['nonzero_removed'] == 1


@pytest.mark.parametrize('sigma', [[2., 1.], [2.1, 0.], [np.nan, 0.]])
def test_incorrect_selector_density_rejected(tmp_path, sigma):
    arrays(tmp_path / 'first', [2., 3.])
    arrays(tmp_path / 'changed', sigma)
    with pytest.raises(ValueError):
        probe.verify_density_restriction(tmp_path / 'first', tmp_path / 'changed', {'test': ''}, lambda p: p[:, 0] == 0, torch.device('cpu'))


def test_missing_dense_reference_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='Missing dense'):
        probe.verify_density_restriction(tmp_path / 'first', tmp_path / 'changed', {'test': ''}, lambda p: p[:, 0] == 0, torch.device('cpu'))
