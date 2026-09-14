import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_forearm_color_witnesses import patch_chroma,witness_errors


def setup_scene():
    rows=[]
    for i,center in enumerate([[0,0,0],[.04,0,0],[-.04,0,0],[0,.04,0]]):
        pose=np.eye(4);pose[:3,3]=center
        rows.append(dict(physical_camera=str(i),transform_matrix=pose.tolist(),fl_x=100,fl_y=100,cx=50,cy=50))
    depths=[np.ones((1080,1920),np.float32) for _ in rows]
    images={r['physical_camera']:np.broadcast_to(np.array([120,80,40],np.uint8),(1080,1920,3)) for r in rows}
    return rows,depths,images


def test_geometrically_consistent_background_is_not_color_evidence_for_skin():
    rows,depths,images=setup_scene()
    for name in ['1','2','3']:images[name]=np.broadcast_to(np.array([40,80,120],np.uint8),(1080,1920,3))
    errors=witness_errors(np.array([[0,0,-1.]]),rows[0],rows,depths,images)
    assert np.isfinite(errors).sum()==3 and (errors<=.04).sum()==0


def test_matching_color_retains_three_witnesses_and_excludes_query_camera():
    rows,depths,images=setup_scene()
    errors=witness_errors(np.array([[0,0,-1.]]),rows[0],rows,depths,images)
    assert np.isnan(errors[0,0]) and (errors<=.04).sum()==3
    depths[3][:]=0
    errors=witness_errors(np.array([[0,0,-1.]]),rows[0],rows,depths,images)
    assert (errors<=.04).sum()==2


def test_chromaticity_is_invariant_to_unsaturated_scalar_brightness():
    image=np.broadcast_to(np.array([50,30,20],np.uint8),(20,20,3))
    np.testing.assert_allclose(patch_chroma(image,[[5,5]]),patch_chroma(image*2,[[5,5]]))
