import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_marshawn_source_gate_replays import record

def test_reports_original_actual_weights_without_opacity_renormalization(tmp_path):
 p=tmp_path/'ray.npz';np.savez(p,t_start=np.array([0.]),t_end=np.array([1.]),sigma=np.array([1.]),weights=np.array([.9]),actual_weights=np.array([.25]),transmittance=np.array([1.]),rgb=np.array([[.4,.8,.2]]))
 r=record(tmp_path,dict(pixel={},profiles={'native':dict(trace=p.name)}),'native')
 assert r['opacity']==.25 and r['transmission']==.75
 np.testing.assert_allclose(r['RGB'],[.1,.2,.05])
