from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from lock_dec5_hair_selection import admission


def baseline():
    return dict(hair=dict(lpips=.2), face_neck=dict(lpips=.1),
                holes_fraction=.02, witness_deficit_fraction=.05, silhouette_recall=.96)


def test_rejects_apparent_stability_from_erasing_hair():
    base = baseline()
    candidate = deepcopy(base)
    candidate.update(holes_fraction=.001, witness_deficit_fraction=.001, silhouette_recall=.8)
    assert not admission(base, candidate)['passes']


def test_rgb_sharpness_alone_is_not_geometry_improvement():
    base = baseline()
    candidate = deepcopy(base)
    candidate['hair']['lpips'] = .15
    assert not admission(base, candidate)['passes']
    candidate['witness_deficit_fraction'] = .04
    assert admission(base, candidate)['passes']


def test_no_holes_in_either_variant_is_not_twenty_percent_improvement():
    base = baseline()
    base['holes_fraction'] = 0
    assert not admission(base, deepcopy(base))['passes']


def test_camera_popping_alternative_still_needs_independent_geometry_gain():
    base = baseline()
    base['camera_popping_fraction'] = .1
    candidate = deepcopy(base)
    candidate['camera_popping_fraction'] = .07
    assert not admission(base, candidate)['passes']
    candidate['witness_deficit_fraction'] = .04
    assert admission(base, candidate)['passes']
