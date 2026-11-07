import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('tau_verifier', Path(__file__).parents[1] / 'scripts/verify_marshawn_quarter_tau_technical.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_successful_updates_and_skips_are_counted_separately():
    rows = [dict(step=1, Adam_updated=True, combined_gradients_finite=True),
            dict(step=2, Adam_updated=False, combined_gradients_finite=False)]
    result = module.check_segment(rows, 1, 2, {'fields':[10.]}, {'fields':[11.]})
    assert result['successful_updates'] == 1 and result['skipped'] == 1


@pytest.mark.parametrize('after,finite', [(12., True), (11., False)])
def test_inconsistent_counter_or_nonfinite_update_rejected(after, finite):
    with pytest.raises(ValueError):
        module.check_segment([dict(step=1, Adam_updated=True, combined_gradients_finite=finite)],
                             1, 1, {'fields':[10.]}, {'fields':[after]})


def test_checkpoint_deletion_invalidates_retention_contract(tmp_path):
    path = tmp_path / 'worker.py'
    path.write_text('gates=(500,1000,2000,4000,8000,12000)\nif gate or partial or terminal or stop: save(step)\nold.unlink()\n')
    with pytest.raises(ValueError, match='retention'):
        module.retention_proof(path)
