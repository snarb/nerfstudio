from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_source_detail_control import decode_eight_source_labels,held_patch_check


def test_categorical_labels_are_decoded_without_reassignment():
    image=np.array([[[230,25,75],[60,180,75],[240,50,230],[0,0,0]]],np.uint8)
    np.testing.assert_array_equal(decode_eight_source_labels(image),[[0,1,7,-1]])
    image[0,0]=[1,2,3]
    with pytest.raises(ValueError,match='palette'):decode_eight_source_labels(image)


def test_held_ncc_comparison_is_fixed_and_fit_rows_cannot_change_readout():
    rng=np.random.default_rng(7)
    rgb=rng.uniform(.2,.7,(2,64,64,3)).astype(np.float32)
    row=dict(x=24,y=24,primary_rank=0,source_rank=1,dx=0.,dy=0.,held=True,block=[0,0])
    observations=dict(patch_size=24,observations=[dict(row) for _ in range(3)])
    a=held_patch_check(rgb,rgb,observations)
    assert a['held_pair_blocks']==1 and a['patches']==3
    assert a['median_block_delta']==0 and a['median_ncc_before']==a['median_ncc_after']
    observations['observations'] += [dict(row,held=False,x=-100,y=-100) for _ in range(100)]
    assert held_patch_check(rgb,rgb,observations)==a


def test_pair_blocks_not_overlapping_patch_counts_define_the_summary():
    rng=np.random.default_rng(9)
    rgb=rng.uniform(.2,.7,(2,80,80,3)).astype(np.float32);after=rgb.copy()
    after[1,8:40,8:40]=after[0,8:40,8:40]
    rows=[dict(x=x,y=x,primary_rank=0,source_rank=1,dx=0.,dy=0.,held=True,block=[i,i])
          for i,x in enumerate([24,56]) for _ in range(3)]
    a=held_patch_check(rgb,after,dict(patch_size=24,observations=rows))
    b=held_patch_check(rgb,after,dict(patch_size=24,observations=rows+rows[:3]*100))
    assert a['median_block_delta']==b['median_block_delta']
    assert a['held_pair_blocks']==b['held_pair_blocks']==2
