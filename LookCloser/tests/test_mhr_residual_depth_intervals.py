from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_mhr_residual_depth_intervals import segments,greedy_veto_cover,refine_edge
from diagnose_local_mhr_residual_rays import landscape_pixel


def test_pixel_rotation_is_exact_for_residual_and_controls():
    image=np.arange(1080*1920).reshape(1080,1920)
    for x,y in [(685,1148),(696,1150),(682,1144),(696,1154)]:
        u,v=landscape_pixel(x,y);assert image[v,u]==np.rot90(image)[y,x]


def test_segments_and_cover_are_explicit_not_minimum_core():
    assert segments([False,True,True,False,True])==[(1,2),(4,4)]
    ids,complete=greedy_veto_cover(np.array([[1,1,0],[0,1,1]],bool));assert ids==[0,1] and complete
    assert greedy_veto_cover(np.array([[1,0]],bool))==([0],False)


def test_refinement_preserves_actual_transition():
    class Probe:
        def accepted(self,p,mode):return p[0]>=.00012345
    lo,hi=refine_edge(Probe(),np.zeros(3),np.array([1.,0,0]),.00012,.00013,'binary',False)
    assert lo<.00012345<=hi and hi-lo<=1e-8


def test_exact_saved_ray_depth_and_direction():
    root=Path('/mnt/data/dec5_mhr_residual_depth_intervals_v2')
    if not root.exists():return
    q=np.load(root/'verification.npz');ray=q['rays'].astype(np.float64)
    np.testing.assert_array_equal(q['prior_points'],ray[:,:3]+q['prior_depth'].astype(np.float64)[:,None]*ray[:,3:])
    old=np.load('/mnt/data/dec5_mhr_production_patch_001193/residual_hole/evidence.npz')
    np.testing.assert_array_equal(q['portrait_xy'][:30],old['portrait_xy'])
    np.testing.assert_array_equal(q['prior_triangle_ids'][:30],old['prior_face'])
