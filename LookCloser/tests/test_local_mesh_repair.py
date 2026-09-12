"""Explicit repair boundaries, fixed outside geometry and smooth camera contracts."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from local_mesh_repair import boundary_loops,close_selected_holes,project_crop,cylinder_geometry,conservative_depth_removal
from smooth_mesh_flythrough import central_path,ANCHORS
from bake_local_mesh_repair import extend_uv,neutral_metal_samples
from finalize_local_mesh_repair import verify_hashes


def test_boundary_inventory_is_exact_and_oriented():
    triangles=np.array([[0,1,2],[0,2,3]])
    loops,rejected=boundary_loops(triangles)
    assert not rejected and len(loops)==1
    assert loops[0].tolist()==[0,1,2,3]


def test_selected_hole_fill_preserves_all_other_geometry():
    import trimesh
    a=trimesh.creation.box();b=trimesh.creation.box();b.apply_translation([3,0,0])
    vertices=np.r_[a.vertices,b.vertices];t1=a.faces[a.face_normals[:,2]<.5];t2=b.faces[b.face_normals[:,2]<.5]+len(a.vertices)
    triangles=np.r_[t1,t2];loops,_=boundary_loops(triangles)
    chosen=next(loop for loop in loops if vertices[loop].mean(0)[0]<1)
    v,t,report=close_selected_holes(vertices,triangles,[chosen],edge_length=.3)
    np.testing.assert_array_equal(v[:len(vertices)],vertices)
    np.testing.assert_array_equal(t[:len(triangles)],triangles)
    remaining,_=boundary_loops(t)
    assert len(remaining)==1 and v[remaining[0]].mean(0)[0]>2
    assert report['other_boundary_edges_unchanged'] and report['closed_boundary_edges']==4


def test_crop_rotation_preserves_native_pixel_centers():
    row={'transform_matrix':np.eye(4),'fl_x':1000.,'fl_y':1000.,'cx':960.,'cy':540.}
    uv,z=project_crop(np.array([[0,0,-1.]]),row,[704,181,1216,693])
    np.testing.assert_allclose(uv,[[717.5,511.5]])
    np.testing.assert_allclose(z,[1.])


def test_uv_extension_keeps_original_texel_coordinates():
    old=np.array([[0.,0.],[.42,.37],[1.,1.]],np.float64)
    patch=np.array([[0.,0.],[1.,1.]],np.float64)
    a,b,(w,h)=extend_uv(old,patch,(4935,4937),(1200,1300))
    np.testing.assert_allclose(a[:,0]*w,old[:,0]*4935,atol=1e-10)
    np.testing.assert_allclose((1-a[:,1])*h,(1-old[:,1])*4937,atol=1e-10)
    assert ((1-b[:,1])*h>=4937+16-1e-10).all()


def test_cylinder_geometry_has_positive_radius_and_unit_axis():
    c,a,r=cylinder_geometry([.1,.2,.3,-.1,np.log(.001)],-.01)
    np.testing.assert_allclose(c,[-.01,.1,.2]);assert np.linalg.norm(a)==pytest.approx(1)
    assert r==pytest.approx(.001)


def fixture_rig():
    rows=[]
    for name,(x,y) in zip(ANCHORS,[(-.1,-.1),(.1,-.1),(.1,.1),(-.1,.1)]):
        pose=np.eye(4);pose[:3,3]=[x,y,2.]
        rows.append({'physical_camera':name,'transform_matrix':pose.tolist(),'fl_x':1000.,'fl_y':1000.,'cx':960.,'cy':540.,'w':1920,'h':1080})
    center=dict(rows[0]);center['physical_camera']='H004_C005_1210SZ';pose=np.eye(4);pose[2,3]=2.;center['transform_matrix']=pose.tolist()
    return rows+[center]


def test_camera_loop_is_slow_uniform_closed_without_anchor_jumps():
    frames,report=central_path(fixture_rig(),[0,0,0],count=360,fps=30,radius=.6)
    assert len(frames)==360 and report['duration_seconds']==12
    weights=np.array([f['convex_weights'] for f in frames]);assert (weights>=0).all()
    np.testing.assert_allclose(weights.sum(1),1)
    positions=np.array([f['transform_matrix'] for f in frames])[:,:3,3]
    steps=np.linalg.norm(np.roll(positions,-1,axis=0)-positions,axis=1)
    assert steps.max()/steps.min()<1.00001
    assert not np.allclose(positions[0],positions[-1])
    assert report['angular_speed_degrees_per_second_min_median_max'][-1]<5
    assert all(f['fl_x']==1000. and f['fl_y']==1000. for f in frames)


@pytest.mark.parametrize('radius',[0.,1.,float('nan')])
def test_camera_loop_rejects_outside_hull_options(radius):
    with pytest.raises(ValueError):central_path(fixture_rig(),[0,0,0],radius=radius)


def test_asymmetric_rig_does_not_move_reference_optical_target():
    rows=fixture_rig()
    for row in rows[:4]:row['transform_matrix'][0][3]+=.08
    frames,report=central_path(rows,[.02,.01,0.])
    np.testing.assert_allclose(report['target_world'],[0.,0.,0.],atol=1e-10)
    for row in frames:
        pose=np.array(row['transform_matrix']);delta=-pose[:3,3]
        assert np.dot(delta,pose[:3,2])<0
        assert np.linalg.norm(np.cross(delta,pose[:3,2]))<1e-10


def test_artifact_validation_rejects_mutations_and_path_escape(tmp_path):
    from joint_temporal_texture import sha
    path=tmp_path/'mesh.ply';path.write_bytes(b'checked artifact')
    expected=sha(path);verify_hashes(tmp_path,{'mesh.ply':expected})
    path.write_bytes(b'changed artifact')
    with pytest.raises(ValueError,match='checksum'):verify_hashes(tmp_path,{'mesh.ply':expected})
    with pytest.raises(ValueError,match='checksum'):verify_hashes(tmp_path,{'../escape':expected})


def test_camera_path_rejects_subject_behind_reference():
    with pytest.raises(ValueError,match='behind'):central_path(fixture_rig(),[0.,0.,3.])


def test_neutral_material_prior_excludes_skin_and_black_without_recoloring():
    color=np.array([[.9,.88,.86],[.7,.49,.3],[.05,.05,.05],[.4,.41,.39]])
    original=color.copy()
    assert neutral_metal_samples(color).tolist()==[True,False,False,True]
    np.testing.assert_array_equal(color,original)


def test_object_prior_cannot_remove_real_supported_finger_or_generated_patch():
    result=conservative_depth_removal(np.ones(4,bool),[0,1,2,-1],[0,2,1],[4,4,2])
    assert result.tolist()==[True,False,False,False]
