"""Optical/layer invariants: thin topology and premultiplication matter here."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "layered_dof", Path(__file__).parents[1] / "scripts/layered_dof.py")
dof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dof)


def test_thin_opaque_strand_and_hole_survive_exactly():
    rng = np.random.default_rng(12)
    alpha = np.zeros((65, 65), np.float32)
    alpha[8:57, 8:57] = 1
    alpha[:, 32] = 1  # one-pixel strand
    alpha[30:33, 22:25] = 0  # fine internal opening
    foreground = rng.random((65, 65, 3), dtype=np.float32)
    background = rng.random((65, 65, 3), dtype=np.float32)
    output = dof.compose_in_focus(foreground * alpha[..., None], alpha,
                                 background, np.full_like(alpha, 8))
    output_srgb = dof.linear_to_srgb(output)
    np.testing.assert_allclose(output_srgb[alpha == 1], foreground[alpha == 1], atol=2e-7)
    background_only = dof.gaussian_defocus(dof.srgb_to_linear(background),
                                          np.full_like(alpha, 8))
    np.testing.assert_allclose(output[alpha == 0], background_only[alpha == 0])


def test_partial_coverage_is_decoded_after_unassociation():
    alpha = np.full((9, 9), .5, np.float32)
    # White foreground with 50% coverage against black.
    premult = np.full((9, 9, 3), .5, np.float32)
    output = dof.compose_in_focus(premult, alpha, np.zeros_like(premult),
                                 np.zeros_like(alpha))
    np.testing.assert_allclose(output, .5, atol=1e-7)
    np.testing.assert_allclose(dof.linear_to_srgb(output), .73535698, atol=1e-7)


def test_native_zero_blur_reconstructs_original_composite():
    rng = np.random.default_rng(8)
    a = rng.random((17, 19), dtype=np.float32)
    fg = rng.random((17, 19, 3), dtype=np.float32)
    bg = rng.random((17, 19, 3), dtype=np.float32)
    expected = fg * a[..., None] + bg * (1 - a[..., None])
    actual = dof.compose_in_focus(fg * a[..., None], a, bg, np.zeros_like(a),
                                  composite_space="srgb")
    np.testing.assert_array_equal(actual, expected)


def test_background_blur_does_not_borrow_subject_color():
    a = np.zeros((65, 65), np.float32)
    a[16:49, 16:49] = 1
    fg = np.zeros((65, 65, 3), np.float32)
    fg[..., 0] = a
    bg = np.full_like(fg, .2)
    result = dof.compose_in_focus(fg, a, bg, np.full_like(a, 8),
                                 composite_space="srgb")
    np.testing.assert_allclose(result[a == 0], .2, atol=2e-7)


def test_background_coc_is_not_subject_coc_at_partial_coverage():
    alpha = np.zeros((17, 19), np.float32)
    alpha[4:13, 4:15] = 1
    alpha[4, 4:15] = .25
    mixed_coc = np.full_like(alpha, 24.)
    mixed_coc[alpha > 0] = 0  # mixed depth reports the focused foreground
    saved_alpha = alpha.copy()
    background_coc = dof.extend_background_coc(mixed_coc, alpha)
    np.testing.assert_array_equal(background_coc, 24.)
    np.testing.assert_array_equal(alpha, saved_alpha)


def test_missing_layers_fail_instead_of_silent_alpha_clipping():
    a = np.full((9, 9), .1, np.float32)
    with pytest.raises(ValueError, match="premultiplied"):
        dof.compose_in_focus(np.ones((9, 9, 3), np.float32), a,
                             np.zeros((9, 9, 3), np.float32), np.zeros_like(a))
    with pytest.raises(ValueError, match="No background"):
        dof.extend_background_coc(a, np.ones_like(a))


def test_background_guard_rejects_mixed_depth_without_eroding_strands():
    alpha = np.zeros((25, 25), np.float32)
    alpha[8:17, 12] = 1
    original_alpha = alpha.copy()
    coc = np.full_like(alpha, 24.)
    coc[7:18, 11:14] = 0  # contaminated depth extends outside opaque support
    assert dof.extend_background_coc(coc, alpha)[7, 11] == 0
    guarded = dof.extend_background_coc(coc, alpha, background_guard_pixels=3)
    np.testing.assert_array_equal(guarded, 24.)
    np.testing.assert_array_equal(alpha, original_alpha)
    np.testing.assert_array_equal(coc[7:18, 11:14], 0)
    with pytest.raises(ValueError, match="nonnegative"):
        dof.extend_background_coc(coc, alpha, background_guard_pixels=-1)
    with pytest.raises(ValueError, match="No background"):
        dof.extend_background_coc(coc, alpha, background_guard_pixels=100)


def test_distant_aperture_preserves_opaque_detail_and_thin_gaps():
    rng = np.random.default_rng(28)
    a = np.zeros((65, 65), np.float32)
    a[8:57, 8:57] = 1
    a[:, 32] = 1
    a[25:29, 23:25] = 0
    a[8, 8:32] = .25
    fg = rng.random((65, 65, 3), dtype=np.float32)
    bg = rng.random((65, 65, 3), dtype=np.float32)
    saved_a = a.copy()
    output = dof.compose_distant_in_focus(fg*a[..., None], a, bg, 6.3)
    np.testing.assert_allclose(dof.linear_to_srgb(output[a == 1]), fg[a == 1], atol=2e-7)
    blurred = dof.disk_defocus(dof.srgb_to_linear(bg), 6.3)
    np.testing.assert_array_equal(output[a == 0], blurred[a == 0])
    expected = dof.srgb_to_linear(fg)*a[..., None] + (1-a[..., None])*blurred
    np.testing.assert_allclose(output, expected, atol=2e-7)
    np.testing.assert_array_equal(a, saved_a)


def test_aperture_plate_conserves_constant_and_impulse_energy():
    image = np.zeros((101, 101, 3), np.float32)
    image[50, 50] = 1
    blurred = dof.disk_defocus(image, 7.4)
    np.testing.assert_allclose(blurred.sum(axis=(0, 1)), 1, atol=2e-6)
    np.testing.assert_allclose(blurred, blurred[::-1, ::-1], atol=2e-8)
    np.testing.assert_allclose(dof.disk_defocus(np.full_like(image, .7), 7.4), .7, atol=2e-7)
    np.testing.assert_array_equal(dof.disk_defocus(image, 0), image)
    with pytest.raises(ValueError, match="nonnegative"):
        dof.disk_defocus(image, -1)
