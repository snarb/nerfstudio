import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_marshawn_margin_continuation import validate_spec
from prepare_marshawn_margin_continuation import STOP,SOURCE_SHA
from verify_marshawn_margin_continuation_phase import verify_cross_arm


def spec():
    return dict(output='out',parent_checkpoint='source',parent_checkpoint_sha256=SOURCE_SHA,parent_spec='parent',parent_spec_sha256='x',anchor_audit='anchor',anchor_audit_sha256='x',mode='zero',arm='source_margin002',release='release',worker_sha256='x',frozen_source_hashes={},source_render_sha256={c+x:'x'for c in ['DC','HC','KA']for x in ['.png','_alpha.png']},margin_manifest='margin',margin_manifest_sha256='x',stopping=copy.deepcopy(STOP),density_selector='source_margin002',zero_reference='own_B002',zero_reference_sha256='x')


def test_exact_recipe_lineage_contract():
    assert validate_spec(spec())['zero_reference']=='own_B002'
    for key,value in [('parent_checkpoint_sha256','wrong'),('density_selector','original_mapped'),('mode','debug')]:
        changed=spec();changed[key]=value
        with pytest.raises(ValueError):validate_spec(changed)
    changed=spec();changed['stopping']['hard_iteration_limit']=22000
    with pytest.raises(ValueError):validate_spec(changed)


def test_first_batch_equal_but_later_rng_need_not_equal():
    reports={'A':dict(first_batch_indices='same',first32_RNG='x'),'B':dict(first_batch_indices='same',first32_RNG='different')}
    assert verify_cross_arm(reports,'technical')['later_cross_arm_RNG_equality_required']is False
    reports['B']['first_batch_indices']='different'
    with pytest.raises(ValueError):verify_cross_arm(reports,'technical')
