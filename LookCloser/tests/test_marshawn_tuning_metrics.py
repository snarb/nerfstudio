"""Actual LPIPS/background independence and restart-safe saturation policy."""
import sys
from pathlib import Path

import pytest
import torch
from torchmetrics.functional.image.lpips import _NoTrainLpips

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_tuning_metrics import (DOMAIN, PlateauTracker, aggregate_actor_metrics,
                                     masked_actor_metrics)


@pytest.fixture(scope='module')
def lpips_model():
    torch.set_num_threads(2)
    return _NoTrainLpips(net='alex').eval()


def test_thin_actor_background_invariance_and_damage(lpips_model):
    g = torch.Generator().manual_seed(100)
    gt = torch.rand((1, 3, 192, 192), generator=g)
    pred = gt*.9
    mask = torch.zeros((1, 1, 192, 192), dtype=torch.bool)
    mask[:, :, 25:170, 80:110] = True
    def score(p, t):
        return masked_actor_metrics(p, t, mask, lpips_model, crop_xyxy=(20, 10, 180, 190))
    a = score(pred, gt)
    outside = ~mask.expand_as(gt)
    pred2, gt2 = pred.clone(), gt.clone()
    pred2[outside], gt2[outside] = 0, 1
    b = score(pred2, gt2)
    for k in ('psnr', 'ssim', 'lpips', 'validation_loss'):
        assert a[k] == pytest.approx(b[k], abs=1e-8)
    pred2[mask.expand_as(gt)] = 0
    damaged = score(pred2, gt2)
    assert damaged['lpips'] > a['lpips']
    assert damaged['psnr'] < a['psnr']
    assert damaged['ssim'] < a['ssim']
    assert damaged['validation_loss'] > a['validation_loss']
    assert a['lpips'] > 0  # strict RF-only metric is unavailable for this thin mask
    with pytest.raises(ValueError, match='excludes actor'):
        masked_actor_metrics(pred, gt, mask, lpips_model, crop_xyxy=(0, 0, 100, 100))


def test_full_mask_agrees_with_standard_lpips(lpips_model):
    gt = torch.rand((1, 3, 128, 160), generator=torch.Generator().manual_seed(3))
    pred = gt*.7
    mask = torch.ones((1, 1, 128, 160), dtype=torch.bool)
    result = masked_actor_metrics(pred, gt, mask, lpips_model, crop_xyxy=(0, 0, 160, 128))
    with torch.inference_mode():
        expected = float(lpips_model(pred, gt, normalize=True).mean())
    assert result['lpips'] == pytest.approx(expected, abs=1e-8)


def test_equal_actor_view_aggregation_rejects_missing_or_duplicate():
    records = [dict(view=v, actor=a, domain=DOMAIN, psnr=x, ssim=x, lpips=x,
                    validation_loss=x, actor_pixels=10**(i+1))
               for i, (v, a, x) in enumerate([('DC','black',1),('DC','grey',3),
               ('HC','black',5),('HC','grey',7),('KA','black',9),('KA','grey',11)])]
    result = aggregate_actor_metrics(records)
    assert result['metrics']['lpips'] == 6
    assert result['by_actor']['black']['lpips'] == 5
    for bad in (records[:-1], records+[records[0]]):
        with pytest.raises(ValueError, match='exactly one'):
            aggregate_actor_metrics(bad)


def test_mandatory_ancestry_then_four_joint_misses(tmp_path):
    p = PlateauTracker()
    for step in (15188, 30376, 45564, 60752, 75940, 91128, 106316):
        assert not p.observe(step, .1, .2)['stopped']
    for i in range(1, 4):
        assert not p.observe(106316+5000*i, .1, .2)['stopped']
    path = tmp_path/'state.json'
    p.save(path)
    resumed = PlateauTracker.load(path)
    assert resumed.to_dict() == p.to_dict()
    state = resumed.observe(126316, .1, .2)
    assert state['stopped']
    assert resumed.observe(126316, .1, .2) == state  # restart replay doesn't add a miss
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        resumed.observe(126316, .09, .2)


def test_cumulative_small_gains_reset_only_matching_signal():
    p = PlateauTracker()
    p.observe(106316, .1, .2)
    p.observe(111316, .09996, .2)
    p.observe(116316, .09992, .2)
    p.observe(121316, .09988, .2)
    assert p.loss_misses == 0 and p.lpips_misses == 3
    assert not p.observe(126316, .09988, .2)['stopped']
    p.observe(131316, .09988, .1996)
    p.observe(136316, .09988, .1992)
    assert not p.observe(141316, .09988, .1988)['stopped']
    assert p.loss_misses == 4 and p.lpips_misses == 0


def test_missing_gates_invalid_data_and_corrupt_restore_are_rejected():
    p = PlateauTracker()
    with pytest.raises(ValueError, match='Missing anchor'):
        p.observe(111316, .1, .2)
    p.observe(106316, .1, .2)
    for args in ((116316, .1, .2), (111316, float('nan'), .2)):
        with pytest.raises(ValueError):
            p.observe(*args)
    state = p.to_dict()
    state['stopped'] = True
    with pytest.raises(ValueError, match='stop flag'):
        PlateauTracker.from_dict(state)


def test_one_signal_keeps_improving_no_fixed_step_limit():
    p = PlateauTracker()
    p.observe(106316, .1, .5)
    for i in range(1, 101):
        assert not p.observe(106316+5000*i, .1, .5-i*.002)['stopped']
    assert p.last_step > 600000 and p.loss_misses == 100


def test_actor_guard_catches_hidden_regression_and_requires_explicit_control():
    from marshawn_tuning_metrics import actor_regression_guard
    control={'selection':{'by_actor':{'black':{'lpips':.1},'grey':{'lpips':.2}}}}
    candidate={'by_actor':{'black':{'lpips':.104},'grey':{'lpips':.1}}}
    guard=actor_regression_guard(candidate,control)
    assert guard['guard_status']=='regression'
    assert not guard['automatic_promotion_allowed']
    candidate['by_actor']['black']['lpips']=.103
    assert actor_regression_guard(candidate,control)['guard_status']=='pass'
    assert actor_regression_guard(candidate,None)['guard_status']=='unassessed'
    assert 'automatic_promotion_allowed' not in actor_regression_guard(candidate,None)
