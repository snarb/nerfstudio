"""Plateau/selection/replay checks for native continuation's durable gate state."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_tuning_metrics import PlateauTracker
from run_marshawn_native_continuation import select_best,advance_gate,atomic,validate_release


def row(step,lpips=.05,psnr=30.,value=.02):
    return dict(step=step,metrics=dict(lpips=lpips,psnr=psnr,ssim=.8,validation_loss=value))


def state():
    baseline=row(20000);tracker=PlateauTracker(anchor_step=20000)
    tracker.observe(20000,.02,.05)
    return dict(evaluations=[baseline],selection=baseline,plateau=tracker.to_dict())


def test_baseline_frontier_and_lpips_tie():
    assert select_best([row(20000),row(25000,lpips=.052,psnr=40.)])['step']==20000
    assert select_best([row(20000),row(25000,lpips=.0509,psnr=31.)])['step']==25000
    assert select_best([row(20000),row(25000)])['step']==20000


def test_four_gates_and_idempotent_resume():
    current=state()
    for step in [25000,30000,35000,40000]:
        before=json.dumps(current,sort_keys=True);new=advance_gate(current,row(step))
        assert json.dumps(current,sort_keys=True)==before
        assert advance_gate(new,row(step))==new
        assert new['plateau']['stopped']==(step==40000)
        current=new


def test_subthreshold_gains_accumulate_and_one_signal_prevents_stop():
    current=state()
    for step,lpips in [(25000,.0496),(30000,.0492),(35000,.0488),(40000,.0484)]:
        current=advance_gate(current,row(step,lpips=lpips))
    assert not current['plateau']['stopped'] and current['plateau']['lpips_misses']==1


def test_incomplete_or_conflicting_gate_cannot_commit():
    original=state()
    for bad in [row(30000),row(25000,lpips=float('nan'))]:
        try:advance_gate(original,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid gate accepted')
    assert len(original['evaluations'])==1
    advanced=advance_gate(original,row(25000))
    try:advance_gate(advanced,row(25000,lpips=.04))
    except ValueError:pass
    else:raise AssertionError('Conflicting replay accepted')
