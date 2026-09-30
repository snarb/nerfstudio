"""Exercise campaign decisions using the measured seed and first-pilot behavior."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_luster_video_campaign import gate_action,numeric_pass


def result(psnr,lpips,ssim=.944,foreground=25):
    return dict(eval_all_psnr=psnr,eval_all_lpips=lpips,eval_all_ssim=ssim,
                per_view=[dict(split='eval',foreground_psnr=foreground)])


def test_growing_poor_pilot_continues_but_poor_plateau_stops():
    early=result(27.916,.10109,.9218);later=result(28.503,.08739,.9303)
    assert gate_action([early,later],later)=='continue'
    plateau=result(28.510,.0871,.931)
    assert gate_action([later,plateau],plateau)=='review'


def test_measured_pilot_exports_before_polish_then_accepts_improvement():
    earlier=result(29.164,.07219,.9424);selected=result(29.117,.07188,.9434)
    history=[earlier,selected]
    assert gate_action(history,selected)=='export'
    first_export=result(29.985,.04735,.9541)
    assert gate_action(history,selected,first_export)=='polish'
    final=result(30.146,.04463,.9556)
    assert gate_action(history,selected,final,polished=True)=='accept'


def test_failed_polish_or_exhausted_budget_returns_for_review():
    selected=result(29.117,.07188,.9434);export=result(29.985,.04735,.9541)
    assert gate_action([selected],selected,export,polished=True)=='review'
    assert gate_action([selected],selected,export,at_limit=True)=='review'


def test_successful_export_skips_unnecessary_polish():
    selected=result(29.716,.05792,.9529);export=result(30.659,.03690,.9621)
    assert gate_action([selected],selected,export)=='accept'


def test_empty_foreground_cannot_pass_on_black_background_alone():
    assert not numeric_pass(result(120,0,1,foreground=None))


def test_budget_cap_does_not_continue_even_when_metrics_improve():
    previous=result(29.1,.078);current=result(29.4,.070)
    assert gate_action([previous,current],current,at_limit=True)=='export'
