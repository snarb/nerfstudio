from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from texture_patchmatch_mesh_mvs import mve_camera
from render_textured_mesh_path import sample_atlas,geometry_audit


@pytest.mark.parametrize('w,h,fx,fy',[(1920,1080,1400,1500),(1080,1920,1500,1400)])
def test_mve_intrinsics_and_opengl_conversion(w,h,fx,fy):
    f=dict(w=w,h=h,fl_x=fx,fl_y=fy,cx=w*.49,cy=h*.51,transform_matrix=np.eye(4).tolist())
    ext,values=mve_camera(f);focal,_,_,aspect,px,py=values
    actual=(focal*h/aspect,focal*h) if w/h*aspect<1 else (focal*w,focal*w*aspect)
    np.testing.assert_allclose(actual,(fx,fy))
    np.testing.assert_allclose((px*w,py*h),(f['cx'],f['cy']))
    np.testing.assert_allclose(ext@np.array([1,2,-3,1]),[1,-2,3,1])


def test_obj_texture_pixel_centers_and_vertical_axis():
    image=np.arange(12).reshape(2,2,3)
    np.testing.assert_allclose(sample_atlas(image,np.array([[0.,1.],[.5,.5]])),image[[0,1],[0,1]])


def test_geometry_allows_reordering_but_rejects_missing_or_reversed_triangle():
    ref=SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]]),triangles=np.array([[0,1,2],[1,3,2]]))
    perm=np.array([2,3,0,1]);v=ref.vertices[perm];inv=np.argsort(perm)
    assert geometry_audit(ref,v,inv[ref.triangles[::-1]])['oriented_triangle_inventory_equal']
    with pytest.raises(ValueError,match='triangle inventory'):geometry_audit(ref,ref.vertices,ref.triangles[:1])
    with pytest.raises(ValueError,match='triangle inventory'):geometry_audit(ref,ref.vertices,ref.triangles[:,::-1])
