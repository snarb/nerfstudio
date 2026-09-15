import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_frame_adapter_changes_no_numeric_geometry_gates():
    import inspect
    import patch_mhr_transfer_001195 as t
    import build_mhr_silhouette_patch_candidates as base
    source=inspect.getsource(t.candidate_builder)
    assert 'inherited=CONFORM' in source and 'frame=FRAME' in source
    assert 'maximum_edge' not in source and 'minimum_centroid_distance' not in source
    assert 'local&(gap>=.00002)' in inspect.getsource(base.build)


def test_same_silhouette_objective_and_termination_constants():
    import fit_mhr_silhouette_conformance as fit
    import continue_mhr_silhouette_convergence as continuation
    assert fit.RECIPE['silhouette_weight']==4
    assert fit.RECIPE['maximum_step']==.001
    assert fit.RECIPE['boundary_tolerance_pixels']==2
    assert continuation.STOP['maximum_outer_iterations']==100
    assert continuation.STOP['unconstrained_step_tolerance']==1e-5
    assert continuation.STOP['consecutive_small_steps']==3


def test_new_frame_initialization_does_not_read_old_fitted_pose():
    import inspect
    import transfer_mhr_001195 as t
    source=inspect.getsource(t.init)
    assert "canonical.fit_once('similarity')" in source
    assert "['canonical.npz','canonical_face_model.obj','LICENSE']" in source
    assert "old/'similarity/fit.npz'" not in source
    assert t.FRAME=='001195' and t.HEAD.parent==t.ROOT
