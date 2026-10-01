"""Exercise campaign decisions using the measured seed and first-pilot behavior."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_luster_video_campaign import gate_action,numeric_pass
import json
import pytest
import run_luster_video_campaign as campaign


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


@pytest.mark.parametrize('rejected',[False,True])
def test_resume_never_retrains_completed_or_rejected_snapshot(tmp_path,monkeypatch,rejected):
    def write(relative,value):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write('visual_reviews/000471.json',dict(accepted=True))
    if rejected:write('visual_reviews/000472.json',dict(accepted=False))
    write('selection.json',result(30.5,.04,.96))
    write('snapshots/000472.json',dict(selection=str(tmp_path/'selection.json'),run='/completed/run',archived_checkpoint=dict(sha256='verified')))
    write('frames/000472/finish_complete.json',dict(run='/completed/run',checkpoint_sha256='verified'))
    monkeypatch.setattr(sys,'argv',['campaign',str(tmp_path),'--start','472','--end','472'])
    def unexpected(*a,**kw):raise AssertionError('Must not launch or restore pruned earlier stages')
    monkeypatch.setattr(campaign.subprocess,'run',unexpected)
    if rejected:
        with pytest.raises(SystemExit) as error:campaign.main()
        assert error.value.code==2
    else:campaign.main()
    status=json.loads((tmp_path/'campaign_status.json').read_text())
    assert status['phase']==('quality_review_required' if rejected else 'visual_review_required')


@pytest.mark.parametrize('reviewed_sha',['verified','wrong',None])
def test_numeric_exception_requires_review_of_the_exact_checkpoint(tmp_path,monkeypatch,reviewed_sha):
    def write(relative,value):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write('visual_reviews/000472.json',dict(accepted=True))
    review=dict(accepted=True)
    if reviewed_sha:review['numeric_gate_override']=dict(checkpoint_sha256=reviewed_sha,reason='Near-threshold PSNR; foreground and visual review support acceptance')
    write('visual_reviews/000473.json',review)
    write('selection.json',result(29.985,.04616,.95325))
    write('snapshots/000473.json',dict(selection=str(tmp_path/'selection.json'),run='/completed/run',archived_checkpoint=dict(sha256='verified')))
    write('frames/000473/finish_complete.json',dict(run='/completed/run',checkpoint_sha256='verified'))
    monkeypatch.setattr(sys,'argv',['campaign',str(tmp_path),'--start','473','--end','473'])
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**kw:pytest.fail('No training should be launched'))
    if reviewed_sha=='verified':campaign.main()
    else:
        with pytest.raises(SystemExit) as error:campaign.main()
        assert error.value.code==2


def test_resume_uses_latest_complete_stage_when_early_checkpoints_were_pruned(tmp_path,monkeypatch):
    def write(relative,value):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write('visual_reviews/000471.json',dict(accepted=True))
    write('snapshots/000471.json',dict(run='/parent'))
    write('frames/000472/data/frequency_complete.json',{})
    selected=dict(result(28.,.10),step=10000,checkpoint='/latest_selected')
    for step in (6000,10000):
        prefix=f'frames/000472/runs/s{step:06d}'
        write(prefix+'/complete.json',dict(latest_checkpoint='/latest'))
        write(prefix+'/request.json',dict(lr=.002))
        write(prefix+'/history.json',[selected])
        write(prefix+'/selection.json',selected)
    monkeypatch.setattr(sys,'argv',['campaign',str(tmp_path),'--start','472','--end','472'])
    restored=[]
    monkeypatch.setattr(campaign,'restore',lambda path:restored.append(path))
    monkeypatch.setattr(campaign,'prune_dominated',lambda *a:None)
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**kw:pytest.fail('Do not retrain a completed stage'))
    with pytest.raises(SystemExit) as error:campaign.main()
    assert error.value.code==2
    assert restored==['/latest_selected']
    assert json.loads((tmp_path/'campaign_status.json').read_text())['run'].endswith('s010000')
