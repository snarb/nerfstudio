import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_measured_source_visibility import MeasuredGate,inject


def test_native_center_offset_and_no_unknown_admission():
    depth=torch.zeros(1,5,5);depth[0,2,3]=2
    q=torch.tensor([[[[2.5,1.5],[1.5,1.5],[2.5,1.5],[2.5,1.5]]]])
    z=torch.tensor([[2.,2.,1.,3.]])
    assert MeasuredGate(depth)(q,z).tolist()==[[True,False,False,False]]


def test_invalid_and_outside_projections_cannot_reuse_clamped_pixel():
    d=torch.ones(1,3,3);q=torch.tensor([[[[-9.,0.],[float('nan'),0.],[0.,0.]]]])
    z=torch.tensor([[1.,1.,-1.]])
    assert not MeasuredGate(d)(q,z).any()
    with pytest.raises(ValueError):MeasuredGate(d,tolerance=np.nan)


def test_injection_rejects_unexpected_layout_and_gates_shared_query():
    import inspect
    import render_smooth_temporal_mesh_video as engine
    original=inspect.getsource(engine.render_one)
    text=inject(original)
    assert text.count('valid&=_measured_gate(q,zq)')==1
    assert 'q,z,valid=query(centroid)' in text and 'q,z,valid=query(points)' in text
    compile(text,'measured_source_test','exec')
    # Like its parent source installer, this adapter is invoked once per fresh
    # worker. Validate template drift, not unsupported repeated installation.
    with pytest.raises(ValueError):inject(original.replace('return q,zq,valid','return altered_layout'))
