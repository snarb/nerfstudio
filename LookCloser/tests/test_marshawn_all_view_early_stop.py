import csv
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from marshawn_all_view_early_stop import (
    STOP_LOSS_SOURCE, all_view_eval_rows, configure_quiet_all_view_stop,
    geometric_mse_from_mean_psnr,
)


def write_rows(path, rows):
    fields = ['step', 'eval_loss', 'eval_loss_delta', 'status',
              'eval_all_psnr', 'eval_all_ssim', 'eval_all_lpips']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def complete(step, psnr, sampled='.001'):
    return dict(step=step, eval_loss=sampled, eval_all_psnr=psnr,
                eval_all_ssim='.8', eval_all_lpips='.3',
                eval_loss_delta='.123', status='original_batch_status')


def test_loss_is_geometric_mean_not_arithmetic_mean():
    image_mses = [.01, .001, .0001]
    mean_psnr = sum(-10 * math.log10(value) for value in image_mses) / 3
    value = geometric_mse_from_mean_psnr(mean_psnr)
    assert value == pytest.approx(math.prod(image_mses) ** (1 / 3))
    assert value != pytest.approx(sum(image_mses) / 3)


def test_full_view_improvement_ignores_sampled_regression_without_mutating_csv(tmp_path):
    path = tmp_path / 'metrics.csv'
    write_rows(path, [complete(6000, '29.0220', '.0012135'),
                      complete(8000, '29.1284', '.00125084')])
    original = path.read_bytes()
    rows = all_view_eval_rows(path)
    assert path.read_bytes() == original
    assert float(rows[-1]['eval_loss']) < float(rows[-2]['eval_loss'])
    assert rows[-1]['sampled_eval_loss'] == '.00125084'
    assert rows[-1]['sampled_eval_loss_delta'] == '.123'
    assert rows[-1]['sampled_eval_status'] == 'original_batch_status'
    assert rows[-1]['eval_loss_source'] == STOP_LOSS_SOURCE
    assert rows[-1]['status'] == 'improving'


@pytest.mark.parametrize('next_psnr', ['29', '30'])
def test_true_full_view_regression_or_plateau_still_stops(tmp_path, next_psnr):
    path = tmp_path / 'metrics.csv'
    write_rows(path, [complete(1, '30', '.02'), complete(2, next_psnr, '.01')])
    rows = all_view_eval_rows(path)
    assert float(rows[-1]['eval_loss']) >= float(rows[-2]['eval_loss'])
    assert rows[-1]['status'] == 'no_improve'


def test_partial_batch_only_nonfinite_and_duplicate_rows(tmp_path):
    path = tmp_path / 'metrics.csv'
    rows = [complete(1, '30'), complete(1, '30'),
            dict(step=2, eval_loss='.002'),
            dict(complete(3, '31'), eval_all_lpips=''),
            dict(complete(4, '31'), eval_all_ssim='nan'),
            dict(complete(5, 'inf')), complete(6, '32')]
    write_rows(path, rows)
    actual = all_view_eval_rows(path)
    assert [int(row['step']) for row in actual] == [1, 6]
    assert actual[0]['eval_loss_delta'] == ''
    assert actual[1]['status'] == 'improving'
    assert all_view_eval_rows(tmp_path / 'missing.csv') == []


def test_all_view_row_does_not_require_batch_evaluation(tmp_path):
    path = tmp_path / 'metrics.csv'
    write_rows(path, [complete(1, '30', '')])
    assert len(all_view_eval_rows(path)) == 1


def test_hook_retains_default_stop_and_native_selection():
    parse = lambda: SimpleNamespace(stop_on_no_improve=True)
    command = object()
    native_select = lambda *args: ('native.ckpt', 'all_psnr_lpips')
    quiet = SimpleNamespace(eval_rows=lambda path: [], print_eval_row=print,
                            best_eval_checkpoint=lambda *args: None,
                            parse_args=parse, train_command=command)
    configure_quiet_all_view_stop(quiet, native_select)
    assert quiet.eval_rows is all_view_eval_rows
    assert quiet.best_eval_checkpoint is native_select
    assert quiet.parse_args is parse and quiet.parse_args().stop_on_no_improve
    assert quiet.train_command is command


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -4000, 4000])
def test_invalid_or_unrepresentable_loss_is_rejected(value):
    with pytest.raises(ValueError):
        geometric_mse_from_mean_psnr(value)
