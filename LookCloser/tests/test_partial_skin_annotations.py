import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_partial_roi_unknown_does_not_disable_measured_free_space(monkeypatch):
    import ordered_forearm_admission as module
    rows = [dict(physical_camera=str(i), w=20, h=20) for i in range(3)]
    names = [r['physical_camera'] for r in rows]
    masks = {n: np.full((20, 20), n != '2', bool) for n in names}
    data = {n + '_trusted': np.full((20, 20), n == '2', bool) for n in names}
    depths = [np.ones((20, 20)) for _ in rows]
    monkeypatch.setattr(module, 'project_integer', lambda row, p: (np.full((len(p), 2), 5.), np.full(len(p), .5)))
    args = (np.zeros((1, 3)), rows, names, masks, data, depths, lambda row, p, inside: inside)
    support, negative, free = module.point_votes(*args)
    assert (support[0], negative[0], free[0]) == (2, 1, 1)
    support, negative, free = module.point_votes(*args, positive_only_annotations=True)
    assert (support[0], negative[0], free[0]) == (2, 0, 1)


def test_partial_annotation_requires_protected_composition(tmp_path):
    import pytest
    from study_coherent_forearm_replacement import prepare
    with pytest.raises(ValueError, match='requires production protection'):
        prepare(tmp_path / 'unused', '001037', positive_only_annotations=True)
    assert not (tmp_path / 'unused').exists()
