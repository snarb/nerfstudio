"""Pruning must preserve candidates that can win later PSNR tie windows."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pytest
from luster_checkpoint_retention import retained_checkpoints, prune_dominated


def row(path, psnr, lpips):
    return dict(checkpoint=str(path), eval_all_psnr=psnr, eval_all_lpips=lpips)


def test_preserves_tradeoffs_and_latest():
    history=[row('a',30,.04),row('b',30.05,.05),row('c',29.9,.055)]
    assert retained_checkpoints(history,'c') == {'a','b','c'}
    history.append(row('d',30.11,.06))
    assert retained_checkpoints(history,'d') == {'a','b','d'}


def test_identical_scores_keep_first_and_latest():
    history=[row('a',30,.04),row('b',30,.04),row('c',29,.08)]
    assert retained_checkpoints(history,'c') == {'a','c'}


def test_nonfinite_metrics_fail():
    with pytest.raises(ValueError,match='Nonfinite'):
        retained_checkpoints([row('a',float('nan'),.04)],'a')


def test_prunes_only_listed_dominated_states(tmp_path):
    frame=tmp_path/'frames/000470'
    folder=frame/'runs/stage/trainer/nerfstudio_models';folder.mkdir(parents=True)
    a,b,c=[folder/f'{name}.ckpt' for name in 'abc']
    for path in [a,b,c]:path.write_bytes(path.name.encode())
    keep=prune_dominated(tmp_path,frame,[row(a,29,.05),row(b,30,.04)],str(b))
    assert keep=={str(b)} and not a.exists() and b.exists() and c.exists()
    assert (tmp_path/'checkpoint_pruning.jsonl').exists()


def test_outside_frame_is_never_deleted(tmp_path):
    outside=tmp_path/'outside.ckpt';outside.write_bytes(b'keep')
    with pytest.raises(ValueError):
        prune_dominated(tmp_path,tmp_path/'frames/000470',[row(outside,29,.05),row('winner',30,.04)],'winner')
    assert outside.read_bytes()==b'keep'
