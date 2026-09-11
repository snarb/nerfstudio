"""Native pixel/gain contracts and portable-mesh regression tests."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import joint_temporal_texture as joint
import convert_exr_nerfstudio_to_jpeg as ingest


def test_projection_and_native_pixel_center_roundtrip():
    row={'transform_matrix':np.eye(4),'fl_x':800.,'fl_y':900.,'cx':960.,'cy':540.}
    uv,z=joint.project(np.array([[0.,0.,-2.],[.1,.2,-2.]],np.float32),[row])
    np.testing.assert_allclose(uv,[[[959.5,539.5],[999.5,449.5]]])
    np.testing.assert_allclose(z,2.)
    image=torch.arange(1920,dtype=torch.float32)[None,None,None].expand(1,1,1080,1920)
    result=joint.sample(image,torch.tensor([[[[10.,20.],[400.5,200.]]]]))
    torch.testing.assert_close(result,torch.tensor([[[[10.,400.5]]]]),atol=1e-4,rtol=0)


def test_response_has_one_time_invariant_gauge_and_known_gain_recovery():
    true=torch.tensor([[.12,-.1,.2],[-.12,.1,-.2]])
    # Different poses/content at two times; same physical-camera response.
    colors=torch.tensor([.03,.15,.55])[None,:,None,None]*torch.ones(2,3,7,9)
    observations=[colors*true.exp()[:,:,None,None],colors.flip(1)*2*true.exp()[:,:,None,None]]
    fitted=torch.nn.Parameter(torch.zeros(2,3));opt=torch.optim.Adam([fitted],lr=.03)
    for _ in range(140):
        opt.zero_grad()
        loss=sum((joint.apply_response(x,fitted)[0].log()-joint.apply_response(x,fitted)[1].log()).square().mean() for x in observations)
        loss.backward();opt.step()
    torch.testing.assert_close(fitted-fitted.mean(0),-true,atol=5e-4,rtol=0)
    a=joint.apply_response(observations[0],fitted)
    b=joint.apply_response(observations[0],fitted+3)
    torch.testing.assert_close(a,b)


def test_warp_is_bounded_per_axis_and_differentiable():
    static=torch.full((2,2,12,20),100.,requires_grad=True)
    uv=torch.tensor([[[[100.,200.],[1800.,900.]]]]).expand(2,-1,-1,-1)
    shift=joint.bounded_warp(static,torch.zeros_like(static),uv)
    assert shift.abs().max()<=2.
    assert shift.norm(dim=-1).max()<=2*np.sqrt(2)+1e-6
    static2=torch.zeros_like(static,requires_grad=True)
    joint.bounded_warp(static2,torch.zeros_like(static2),uv).sum().backward()
    assert static2.grad.abs().sum()>0


def test_display_matches_existing_converter_for_fixed_gain():
    rng=np.random.default_rng(21);x=rng.uniform(-.02,3,(10,10,3)).astype(np.float32)
    np.testing.assert_array_equal(ingest.tone_map(x,4.3),np.rint(joint.display(x,4.3)*255).astype(np.uint8))


def test_fixed_ingest_never_estimates_camera_exposure_and_resume_pins_gain(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    names=['frame_train_00000.exr','frame_train_00001.exr','frame_eval_00001.exr']
    for name in names:(source/name).write_bytes(b'fixture')
    joint.atomic_json(source/'transforms.json',{'frames':[{'file_path':n} for n in names]})
    monkeypatch.setattr(ingest,'resolve_nerfstudio_dataset',lambda *a,**k:SimpleNamespace(train_images=[source/n for n in names[:2]],eval_images=[source/names[-1]]))
    def forbidden(*a,**k):raise AssertionError('Fixed mode must never estimate exposure')
    monkeypatch.setattr(ingest,'calibrate_exr_paths',forbidden)
    monkeypatch.setattr(ingest,'load_exr_image',lambda p:np.full((12,16,3),.1 if '00000' in p.name else .2,np.float32))
    dest=tmp_path/'output';args=['--input',str(source),'--output',str(dest),'--exposure-mode','fixed','--fixed-exposure-gain','4.3']
    assert ingest.main(args)==0
    manifest=joint.read(dest/'conversion_manifest.json')
    assert [r['exposure_gain'] for r in manifest['images']]==[4.3]*3
    assert manifest['tone_map']['calibration'] is None
    assert ingest.main(args+['--resume'])==0
    with pytest.raises(RuntimeError,match='request changed'):ingest.main(args[:-1]+['4.4','--resume'])


@pytest.mark.parametrize('gain',['0','-1','nan','inf'])
def test_fixed_gain_rejects_invalid_values(tmp_path,monkeypatch,gain):
    monkeypatch.setattr(ingest,'resolve_nerfstudio_dataset',lambda *a,**k:SimpleNamespace())
    with pytest.raises(ValueError,match='finite and positive'):
        ingest.main(['--input',str(tmp_path/'in'),'--output',str(tmp_path/'out'),
                     '--exposure-mode','fixed','--fixed-exposure-gain',gain])


def test_small_normalized_mesh_atlas_and_embedded_glb(tmp_path):
    import open3d as o3d
    import trimesh
    import bake_joint_temporal_mesh as baker
    mesh=o3d.geometry.TriangleMesh.create_box(.001,.001,.001)
    path=tmp_path/'mesh.ply';o3d.io.write_triangle_mesh(str(path),mesh)
    out=tmp_path/'asset';atlas=baker.atlas_geometry(path,out,resolution=128)
    assert len(atlas['pixels'])>1000  # Previously tiny triangles collapsed in xatlas.
    np.testing.assert_array_equal(atlas['mapping'][atlas['indices']],atlas['triangles'])
    Image.new('RGB',(int(atlas['width']),int(atlas['height'])),(180,70,40)).save(out/'texture_joint.png')
    baker.export_asset(out,atlas,'test',None)
    record=joint.read(out/'asset_manifest.json')
    assert record['roundtrip_vertices_max_error']<1e-6
    imported=trimesh.load(record['glb'],force='scene',process=False)
    geom=next(iter(imported.geometry.values()))
    assert geom.visual.material.baseColorTexture.size==(int(atlas['width']),int(atlas['height']))


def test_frozen_parameter_loader_rejects_exposure_or_profile_edits(tmp_path):
    np.savez_compressed(tmp_path/'parameters.npz',log_gain=np.zeros((62,3),np.float32),
                        static_warp=np.zeros((62,2,12,20),np.float32),residual_000899=np.zeros((62,2,12,20),np.float32))
    joint.atomic_json(tmp_path/'fit_request.json',{'fit_frames':['000899']})
    joint.atomic_json(tmp_path/'exposure.json',{'fixed_exposure_gain':4.3})
    joint.atomic_json(tmp_path/'camera_profiles.json',{'rgb_gain':np.ones((62,3)).tolist(),
                     'exposure_sha256':joint.sha(tmp_path/'exposure.json')})
    joint.atomic_json(tmp_path/'fit_result.json',{'parameters_sha256':joint.sha(tmp_path/'parameters.npz'),
                     'request_sha256':joint.sha(tmp_path/'fit_request.json')})
    a,b,c=joint.load_parameters(tmp_path,'000899',device='cpu')
    assert a.shape==(62,3) and torch.count_nonzero(c)==0
    joint.atomic_json(tmp_path/'exposure.json',{'fixed_exposure_gain':4.4})
    with pytest.raises(ValueError,match='exposure checksum'):joint.load_parameters(tmp_path,'000899',device='cpu')


def test_incomplete_prepared_cache_is_never_accepted(tmp_path):
    joint.atomic_json(tmp_path/'complete.json',{'hashes':{}})
    with pytest.raises(ValueError,match='Incomplete prepared'):joint.validate_cache(tmp_path)


def test_robust_linear_fusion_preserves_identical_views_and_ignores_zero_weight():
    import bake_joint_temporal_mesh as baker
    colors=torch.full((5,3,20),.2);weights=torch.ones((5,20))
    colors[4]=9.;weights[4]=0
    torch.testing.assert_close(baker.robust_fusion(colors,weights),torch.full((3,20),.2))


def test_calibrate_helper_prepares_all_times_then_fits_once(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(joint,'prepare_frame',lambda *args:calls.append(('prepare',*args)))
    monkeypatch.setattr(joint,'fit',lambda *args:calls.append(('fit',*args)))
    root=tmp_path/'new';fit=['000899','000973'];held=['001059']
    result=joint.calibrate(root,fit,held,patches=100,iterations=12)
    assert calls==[('prepare',root,f,100) for f in fit+held]+[('fit',root,fit,held,12)]
    assert result['status']=='calibrated_not_visually_approved'
    assert result['uses_eval_rgb'] is False and result['changes_geometry'] is False
    assert Path(result['report']).is_file()


def test_calibrate_dry_run_writes_nothing_and_loads_no_images(tmp_path,monkeypatch,capsys):
    import json
    def forbidden(*args,**kwargs):raise AssertionError('Dry run must not execute stages')
    monkeypatch.setattr(joint,'prepare_frame',forbidden);monkeypatch.setattr(joint,'fit',forbidden)
    root=tmp_path/'new'
    joint.main(['calibrate','--output',str(root),'--dry-run'])
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='planned' and len(result['stages'])==6
    assert not root.exists()


@pytest.mark.parametrize('fit,held',[
    (['000899'],['001059']),
    (['000899','000973'],[]),
    (['000899','000899'],['001059']),
    (['000899','000973'],['000973']),
    (['000899','../973'],['001059']),
])
def test_calibrate_rejects_invalid_time_inventory_before_writes(tmp_path,fit,held):
    root=tmp_path/'new'
    with pytest.raises(ValueError):joint.calibrate(root,fit,held,dry_run=True)
    assert not root.exists()


def test_calibrate_rejects_changed_existing_recipe_before_preparing(tmp_path,monkeypatch):
    root=tmp_path/'existing'
    joint.atomic_json(root/'fit_request.json',{'fit_frames':['000899','000973'],
                     'held_frames':['001059'],'iterations':160,'script_sha256':'old_code'})
    def forbidden(*args,**kwargs):raise AssertionError('Must fail before preparing')
    monkeypatch.setattr(joint,'prepare_frame',forbidden)
    with pytest.raises(ValueError,match='Existing calibration request differs'):
        joint.calibrate(root,['000899','000973'],['001059'])
    assert not (root/'cache').exists()


def test_calibrate_rejects_source_as_output(monkeypatch,tmp_path):
    monkeypatch.setattr(joint,'SOURCE',tmp_path/'source')
    with pytest.raises(ValueError,match='immutable source'):
        joint.calibrate(joint.SOURCE/'new',['000899','000973'],['001059'],dry_run=True)
    assert not joint.SOURCE.exists()
