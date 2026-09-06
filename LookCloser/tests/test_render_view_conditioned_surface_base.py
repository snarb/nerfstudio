from pathlib import Path
import sys
import json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_view_conditioned_surface_base import evaluate_base,validate_directional_manifest
from colmap_patchmatch_tsdf_campaign_common import sha256


def test_known_linear_direction_base_and_smooth_camera_motion():
    coeff=np.zeros((2,4,3),np.float32);coeff[:,0]=.4;coeff[:,1,0]=.1
    points=np.array([[0,0,0],[0,0,0]],np.float32)
    np.testing.assert_allclose(evaluate_base(coeff,points,[1,0,0],1),[[.5,.4,.4]]*2,atol=1e-7)
    np.testing.assert_allclose(evaluate_base(coeff,points,[0,1,0],1),[[.4,.4,.4]]*2,atol=1e-7)
    a=evaluate_base(coeff,points,[1,1,0],1);b=evaluate_base(coeff,points,[1,1.001,0],1)
    assert np.max(np.abs(a-b))<1e-4


def test_zero_angular_terms_do_not_depend_on_query():
    coeff=np.zeros((2,9,3),np.float32);coeff[:,0]=[.1,.2,.3];p=np.zeros((2,3),np.float32)
    np.testing.assert_array_equal(evaluate_base(coeff,p,[1,2,3],2),evaluate_base(coeff,p,[-3,1,-2],2))


def test_nonfinite_or_wrong_shape_base_fails():
    coeff=np.zeros((2,4,3));p=np.zeros((2,3));coeff[0,0,0]=float('nan')
    with pytest.raises(ValueError):evaluate_base(coeff,p,[1,1,1],1)
    with pytest.raises(ValueError):evaluate_base(np.zeros((2,4,3)),p,[0,0,0],1)


@pytest.mark.parametrize('change',[
    {'uses_eval_rgb':True},{'uses_semantic_masks':True},{'query_dependent_base':False},
    {'fit':{'converged':False}},{'mesh_sha256':'wrong'},
    {'source_base_manifest_sha256':'wrong'},{'degree':3}
])
def test_directional_provenance_rejected(tmp_path,change):
    base=tmp_path/'base.json';base.write_text('{}');mesh=tmp_path/'mesh.ply';mesh.write_bytes(b'fixture')
    manifest=dict(uses_eval_rgb=False,uses_semantic_masks=False,query_dependent_base=True,
        fit={'converged':True},degree=2,mesh_sha256=sha256(mesh),source_base_manifest_sha256=sha256(base))
    validate_directional_manifest(manifest,base,mesh)
    with pytest.raises(ValueError):validate_directional_manifest({**manifest,**change},base,mesh)
