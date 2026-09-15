"""Pixel-center and optical-budget controls for the opt-in 6K renderer."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_cinematic_6k_output import scale_camera,SCALE


def camera():
    return dict(w=1920,h=1080,fl_x=1600.,fl_y=1610.,cx=959.25,cy=541.75,
        transform_matrix=np.eye(4).tolist())


def test_output_rays_preserve_continuous_hd_coordinates():
    old=camera();new=scale_camera(old,6144,3456)
    for key in ['fl_x','fl_y','cx','cy']:assert new[key]==old[key]*3.2
    for p in [0,159,3071,6143]:
        u_hd=(p+.5)/3.2-.5
        assert np.isclose((p+.5-new['cx'])/new['fl_x'],(u_hd+.5-old['cx'])/old['fl_x'],atol=1e-14)
    assert new['transform_matrix']==old['transform_matrix'] and old['w']==1920


def test_native_source_uses_anisotropic_crop_pixel_centers():
    old=camera();native=scale_camera(old,5461,3072)
    assert SCALE[0]!=SCALE[1]
    for q in [-.3,0.,.25]:
        u_hd=old['fl_x']*q+old['cx']-.5
        u_native=native['fl_x']*q+native['cx']-.5
        assert np.isclose(u_native,(u_hd+.5)*SCALE[0]-.5)
        v_hd=old['fl_y']*q+old['cy']-.5
        v_native=native['fl_y']*q+native['cy']-.5
        assert np.isclose(v_native,(v_hd+.5)*SCALE[1]-.5)


def test_ending_optical_budget_not_entire_6k_source():
    assert np.isclose(5461/1.9,2874.2105263157896)
    assert np.isclose(3072/1.9,1616.842105263158)
    # Correct source footprint remains bounded even when output is 6144 wide.
    source=camera();target=scale_camera(source,6144,3456)
    target['fl_x']*=1.9;target['fl_y']*=1.9
    native=scale_camera(source,5461,3072)
    uv=(np.array([0,6143])+.5-target['cx'])/target['fl_x']*native['fl_x']+native['cx']-.5
    assert np.all((uv>=0)&(uv<5461))


def test_parallel_publication_never_overwrites_even_empty_directory(tmp_path):
    from parallel_cinematic_6k_endings import publish_no_replace
    import pytest
    source=tmp_path/'source';dest=tmp_path/'dest';source.mkdir();dest.mkdir()
    with pytest.raises(FileExistsError):publish_no_replace(source,dest)
    assert source.is_dir() and dest.is_dir()
    destination=tmp_path/'new_destination';publish_no_replace(source,destination)
    assert destination.is_dir() and not source.exists()


def test_parallel_relative_symlink_publication_no_overwrite(tmp_path):
    from parallel_cinematic_6k_endings import publish_symlink_no_replace
    import pytest
    source=tmp_path/'payload';dest=tmp_path/'existing';source.mkdir();dest.mkdir()
    with pytest.raises(FileExistsError):publish_symlink_no_replace(source,dest)
    assert not dest.is_symlink()
    target=tmp_path/'published';publish_symlink_no_replace(source,target)
    assert target.is_symlink() and target.resolve()==source.resolve()
    assert not __import__('os').path.isabs(__import__('os').readlink(target))
