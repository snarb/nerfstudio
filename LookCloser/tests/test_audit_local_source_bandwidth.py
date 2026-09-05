from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_local_source_bandwidth import source_pairs, spatial_partition, stratify_depth, summarize_observations


def test_all_pairs_includes_observations_without_angular_primary():
    assert source_pairs(4) == [(0,1),(0,2),(0,3)]
    pairs = source_pairs(4, True)
    assert len(pairs) == len(set(pairs)) == 6 and (2,3) in pairs
    assert all(a < b for a,b in pairs)
    with pytest.raises(ValueError): source_pairs(1)


@pytest.mark.parametrize('half_width', [20,32])
def test_fit_and_held_footprints_cannot_overlap(half_width):
    patches = [(x,y,spatial_partition(x,y,half_width=half_width))
               for x in range(32,768,16) for y in range(32,512,16)]
    patches = [(x,y,p) for x,y,p in patches if p is not None]
    assert {p['held'] for _,_,p in patches} == {False,True}
    for x,y,p in patches:
        for u,v,q in patches:
            if p['held'] != q['held']:
                assert abs(x-u) >= 2*half_width or abs(y-v) >= 2*half_width
    assert spatial_partition(128,128) is None


def test_many_overlapping_patches_do_not_invent_independent_blocks():
    rows = [{'held': i%2 == 0, 'block': [i%2,0], 'relative_blur_variance': -1.} for i in range(100)]
    result = summarize_observations(rows)
    assert result['fit']['patches'] == 50 and result['fit']['blocks'] == 1
    assert not result['qualified']


def test_held_depths_do_not_set_bin_boundaries():
    rows = [{'held': False, 'depth': float(i), 'block': [i,0], 'relative_blur_variance': -1.} for i in range(1,9)]
    a = stratify_depth(rows)
    b = stratify_depth(rows + [{'held': True, 'depth': 100., 'block': [20,0], 'relative_blur_variance': 1.}])
    assert a['boundaries'] == b['boundaries']
    assert b['bins'][-1]['held']['patches'] == 1


def test_qualification_requires_agreement_on_multiple_held_blocks():
    rows = [{'held': i>=3, 'depth': 1., 'block': [i,0], 'relative_blur_variance': -1.} for i in range(5)]
    assert summarize_observations(rows)['qualified']
    rows[-1]['relative_blur_variance'] = 1.
    assert not summarize_observations(rows)['qualified']


def test_invalid_input_fails():
    with pytest.raises(ValueError): spatial_partition(32,32,block_size=16)
    with pytest.raises(ValueError): stratify_depth([],bins=0)
    with pytest.raises(ValueError): stratify_depth([{'held': False,'depth': float('nan')}])
