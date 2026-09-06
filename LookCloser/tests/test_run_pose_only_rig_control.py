from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_pose_only_rig_control import evaluate_gate


def fixture():
    def rows(error):return [dict(frame_id=t,left_camera='A',right_camera='B',points=10,
        block_median_absolute_error=error) for t in ['000979','001219']]
    scores=dict(held_frames=['000979','001219'],results={'old':rows(.5),'new':rows(.4)},summaries=[
        dict(calibration='old',pair_block_median=.5,pair_block_p90=.6,fraction_pairs_improved=0.),
        dict(calibration='new',pair_block_median=.4,pair_block_p90=.5,fraction_pairs_improved=1.)])
    manifest=dict(solver_report='Termination: CONVERGENCE',camera_changes=[dict(rotation_degrees=.1,center_shift_world=.02)])
    return scores,manifest


def test_improved_shared_inventory_is_eligible_not_a_surface_pass():
    scores,manifest=fixture();gate=evaluate_gate(scores,'new',manifest,'old')
    assert gate['eligible_for_dense_control'] and all(gate['checks'].values())
    assert 'visual_pass' not in gate


def test_one_worse_held_time_rejects_candidate():
    scores,manifest=fixture();scores['results']['new'][1]['block_median_absolute_error']=.7
    assert not evaluate_gate(scores,'new',manifest,'old')['eligible_for_dense_control']


def test_changed_pair_inventory_fails_closed():
    scores,manifest=fixture();scores['results']['new'][0]['points']=9
    with pytest.raises(ValueError,match='identical unique'):evaluate_gate(scores,'new',manifest,'old')


def test_unconverged_or_deformed_rig_is_rejected():
    scores,manifest=fixture();bad=deepcopy(manifest);bad['solver_report']='Termination: NO_CONVERGENCE'
    assert not evaluate_gate(scores,'new',bad,'old')['eligible_for_dense_control']
    bad=deepcopy(manifest);bad['camera_changes'][0]['rotation_degrees']=1.
    assert not evaluate_gate(scores,'new',bad,'old')['eligible_for_dense_control']
