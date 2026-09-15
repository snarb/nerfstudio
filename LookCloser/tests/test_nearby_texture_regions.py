import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from nearby_texture_regions import texture_regions, visible_face_weights


def test_only_near_parallel_original_faces_join():
    patch = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
    v = np.concatenate([patch+[0, 0, .001], patch+[0, 0, .02], patch])
    t = np.arange(9).reshape(3, 3)
    region = texture_regions(v, t, 2, join_original=True)
    assert region.tolist() == [0, -1, 0]
    assert texture_regions(v, t, 2).tolist() == [-1, -1, 0]


def test_hidden_and_unrelated_faces_have_zero_weight():
    ids = np.array([[0, 1], [2, 2]], np.uint32);depth = np.array([[1., 1.], [1., np.inf]])
    weights = visible_face_weights(ids, depth, 4, np.array([-1, 0, 0, 1]))
    assert weights.tolist() == [0., 1., 1., 0.]
