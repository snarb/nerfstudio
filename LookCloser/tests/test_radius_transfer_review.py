import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_radius_seed_transfer import counts


def test_keeps_fixed_reference_holes_and_reports_new_holes():
    old = np.ones((7,7)); old[3,3] = 0
    new = old.copy(); new[3,3] = 1; new[2,2] = 0
    rgb0 = np.full((7,7,3), 100, np.uint8); rgb0[3,3] = 0
    rgb1 = np.full((7,7,3), 100, np.uint8); rgb1[2,2] = 0
    r = counts(rgb0, old, rgb1, new, [0,0,7,7])
    assert r['original_enclosed_misses'] == 1 and r['remaining_enclosed_misses'] == 0
    assert r['new_geometry'] == r['lost_geometry'] == r['newly_black_rgb'] == 1


def test_new_geometry_without_texture_is_not_counted_as_color_repair():
    old = np.ones((7,7)); old[3,3] = 0; new = np.ones((7,7))
    rgb = np.full((7,7,3), 100, np.uint8); rgb[3,3] = 0
    r = counts(rgb, old, rgb, new, [0,0,7,7])
    assert r['new_geometry'] == r['new_uncolored_geometry'] == 1
