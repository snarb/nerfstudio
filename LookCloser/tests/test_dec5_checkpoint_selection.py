"""Regression: the moving PSNR tie band can select an older saved gate."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_dec5_hair_baseline import publish_best


def test_new_psnr_maximum_can_select_a_previously_unselected_older_checkpoint(tmp_path):
    history = [dict(step=1, eval_all_psnr=20., eval_all_lpips=.1),
               dict(step=2, eval_all_psnr=20.04, eval_all_lpips=.2)]
    for step in (1, 2, 3):
        (tmp_path/('gate_%06d.pt'%step)).write_bytes(('weights%d'%step).encode())
    assert publish_best(history, tmp_path)['step'] == 1
    history.append(dict(step=3, eval_all_psnr=20.10, eval_all_lpips=.3))
    assert publish_best(history, tmp_path)['step'] == 2
    assert (tmp_path/'best.pt').read_bytes() == b'weights2'
