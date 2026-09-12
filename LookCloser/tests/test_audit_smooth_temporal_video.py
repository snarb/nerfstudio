from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_smooth_temporal_video import validate_review_inventory


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
