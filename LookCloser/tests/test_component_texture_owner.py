import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from component_texture_owner import components, choose_owners


def test_disconnected_old_and_new_face_components():
    triangles = np.array([[0, 1, 2], [3, 4, 5], [5, 4, 6], [7, 8, 9]])
    ids = components(triangles, 1)
    assert ids[0] == ids[1] and ids[2] != ids[0]


def test_coverage_owner_preserves_original_and_invisible_faces():
    labels = np.array([1, 0, 0, 0]);quality = np.array([[1., 10., 10., 0.], [2., 1., 1., 1.]])
    selected, record = choose_owners(labels, quality, np.zeros(3, int), np.ones(4), 1, minimum_faces=1)
    assert selected.tolist() == [1, 1, 1, 1] and record[0]['owner'] == 1
    quality[1, 3] = 0
    selected, record = choose_owners(labels, quality, np.zeros(3, int), np.ones(4), 1, minimum_faces=1)
    np.testing.assert_array_equal(selected, labels)
    assert not record[0]['accepted']
