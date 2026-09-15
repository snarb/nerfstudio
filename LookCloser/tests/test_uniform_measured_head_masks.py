import inspect
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import refine_measured_head_masks as masks
from apply_measured_head_mask_completion import inject_refined_masks
from transfer_close_boundary_completion import optional_override
import guard_poisson_jaw_completion as guard


def test_foreground_witnesses_match_names_not_array_order_and_do_not_mutate(monkeypatch):
    rows=[{'physical_camera':'a'},{'physical_camera':'b'}]
    data=np.zeros((2,1080,1920),bool);data[1,4,3]=True
    before=data.copy()
    monkeypatch.setattr(masks,'project_integer',lambda row,p:(np.array([[3,4],[-1,4],[1920,4]]),np.ones(3)))
    result=masks.foreground_witnesses(np.zeros((3,3)),rows,data,['b','a'])
    np.testing.assert_array_equal(result,[[True,False,False],[False,False,False]])
    np.testing.assert_array_equal(data,before)


def test_refined_mask_adapter_only_injects_one_explicit_load():
    original=optional_override(inspect.getsource(guard.prepare));updated=inject_refined_masks(original)
    compile(updated,'<test>','exec')
    assert updated.replace('    masks=load_refined_masks(FRAME,names,masks)\n','')==original
    with pytest.raises(ValueError):
        inject_refined_masks('def prepare(root):\n    pass\n')
