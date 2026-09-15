import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from study_forearm_independent_patch_evidence import select_witnesses


def row(name, angle):
    pose = np.eye(4);radians = np.deg2rad(angle)
    pose[:3, 3] = [np.sin(radians), 0, np.cos(radians)]
    return dict(physical_camera=name, transform_matrix=pose)


def test_prior_query_and_near_duplicate_are_excluded():
    query = row('query', 0)
    rows = [query, row('prior', 3), row('duplicate', .2), row('far', 50), row('b', 5), row('a', 2)]
    result = select_witnesses(query, rows, {'prior'}, np.zeros(3))
    assert [r['physical_camera'] for r in result] == ['a', 'b']
