from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_smooth_temporal_video import validate_review_inventory, validate_opt_in_delivery_claim


def review(ids,status='accepted_with_known_artifacts'):
    return (Path('review.json'),{'frame_ids':ids,'status':status})


def test_all_frames_need_explicit_review():
    with pytest.raises(ValueError,match='incomplete'):
        validate_review_inventory(['000899','000901'],[review(['000899'])])


def test_duplicate_reviews_are_not_silently_overwritten():
    with pytest.raises(ValueError,match='duplicate'):
        validate_review_inventory(['000899'],[review(['000899']),review(['000899'])])


@pytest.mark.parametrize('status',['pending','uncertain'])
def test_unfinished_review_is_not_a_pass(status):
    with pytest.raises(ValueError,match='Unfinished'):
        validate_review_inventory(['000899'],[review(['000899'],status)])


def test_explicit_fail_is_retained_not_hidden():
    result=validate_review_inventory(['000899'],[review(['000899'],'fail')])
    assert result['000899'][1]['status']=='fail'


def test_delivery_completion_is_opt_in_not_implied_by_no_catastrophe():
    assert not validate_opt_in_delivery_claim({},set(),0)


@pytest.mark.parametrize('notable,failures', [({'000971'},0),(set(),1)])
def test_delivery_claim_cannot_hide_notable_or_catastrophic_failure(notable,failures):
    with pytest.raises(ValueError):
        validate_opt_in_delivery_claim({'goal_fully_achieved_claimed':True,'artifact_tolerant_preview_accepted':True},notable,failures)


def test_explicit_artifact_tolerant_delivery_does_not_require_perfect_geometry():
    assert validate_opt_in_delivery_claim({'goal_fully_achieved_claimed':True,'artifact_tolerant_preview_accepted':True,
                                          'strict_artifact_free':False},set(),0)
