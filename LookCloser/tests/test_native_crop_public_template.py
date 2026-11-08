"""CPU contracts for the reusable NN-frequency crop worker; no NeRF training."""
import copy
import ast
import hashlib
import importlib.util
import json
import random
import os
import types
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT=Path(os.environ.get('LOOKCLOSER_TEST_SOURCE_CODE','/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code'))
sys.path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'LookCloser/scripts'),str(ROOT/'scripts'),str(ROOT.parent)]
from native_crop_sampling import gather_rasters
from native_crop_sampling_v2 import PackedRaggedGather


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def worker(monkeypatch):
    if not (SOURCE_ROOT/'LookCloser/scripts/marshawn_iteration_budget.py').exists():
        pytest.skip('Set LOOKCLOSER_TEST_SOURCE_CODE to a complete sealed NN source release')
    monkeypatch.setenv('LOOKCLOSER_SEALED_CODE', str(SOURCE_ROOT))
    name = '_native_crop_public_worker_test'
    loader = importlib.util.spec_from_file_location(name, ROOT/'scripts/train_native_crop.py')
    module = importlib.util.module_from_spec(loader)
    sys.modules[name] = module
    loader.loader.exec_module(module)
    return module


@pytest.fixture
def spec(tmp_path, worker, monkeypatch):
    # Synthetic source/config receipts test validation, not real-data qualification.
    source = SOURCE_ROOT
    sentinel = source/'nerfstudio/engine/optimizers.py'
    seal = tmp_path/'source_seal.json'
    seal.write_text(json.dumps({'files_sha256': {'nerfstudio/engine/optimizers.py': sha(sentinel)}}))
    data = tmp_path/'data'; data.mkdir()
    original = tmp_path/'original'; original.mkdir()
    for folder in (data, original):
        (folder/'box.json').write_text(json.dumps({'aabb': [[-1,-1,-1],[1,1,1]]}))
    manifest = original/'manifest.json'; manifest.write_text('{}')
    plan = tmp_path/'stage_plan.json'; plan.write_text(json.dumps({'schema':1,'stop_after_iterations':32000}))
    return dict(data=str(data), output=str(tmp_path/'output'), source_code=str(source),
                source_seal=str(seal), source_seal_sha256=sha(seal), stage_plan=str(plan),
                code_sha256={name:sha(ROOT/'scripts'/name) for name in (
                    'train_native_crop.py','native_crop_sampling.py','native_crop_sampling_v2.py')},
                semantic_admission=dict(manifest=str(manifest),manifest_sha256=sha(manifest),data=str(original)))


def test_defaults_setup_real_cpu_adam_and_yaml(worker, spec):
    import yaml
    initial_cuda = torch.cuda.is_initialized()
    settings = worker.validate_spec(spec)
    cfg = worker.build_config(settings)
    assert (settings['distortion_final'], settings['radius'], settings['exposure_scale']) == (.1,0,1)
    assert cfg.pipeline.datamanager.train_num_rays_per_batch == 4096
    assert cfg.pipeline.model.occupancy_dilation_radius == 0
    assert cfg.pipeline.model.adaptive_warmup_steps == 4096
    assert cfg.pipeline.model.tcnn_network_jit is False
    for beta in (.999,.99):
        changed = worker.build_config(dict(settings,adam_beta2=beta))
        optimizer_cfg = yaml.load(yaml.dump(changed.optimizers['fields']['optimizer']), Loader=yaml.Loader)
        optimizer = optimizer_cfg.setup(params=[torch.nn.Parameter(torch.tensor([.2]))])
        assert optimizer.param_groups[0]['betas'] == (.9,beta)
        assert optimizer.param_groups[0]['lr'] == .01
        assert optimizer.param_groups[0]['eps'] == 1e-15
        assert changed.optimizers['fields']['scheduler'].lr_final == .0001
    assert torch.cuda.is_initialized() == initial_cuda


@pytest.mark.parametrize('radius', [0,1])
@pytest.mark.parametrize('weight', [0,.01,.03,.1])
def test_reviewed_geometry_knobs_and_phase(worker,spec,radius,weight):
    settings = worker.validate_spec(dict(spec,radius=radius,distortion_final=weight))
    cfg = worker.build_config(settings)
    assert cfg.pipeline.model.occupancy_dilation_radius == radius
    for scale in (1,2,4):
        virtual, fraction = worker.exposure_phase(4096//scale,scale)
        assert (virtual,fraction) == (4096,0.)
        assert .01+(weight-.01)*fraction == .01
        _, end_fraction = worker.exposure_phase(12288//scale,scale)
        assert .01+(weight-.01)*end_fraction == pytest.approx(weight)


@pytest.mark.parametrize('change', [
    {'radius':2},{'radius':True},{'distortion_final':.02},{'distortion_final':float('nan')},
    {'exposure_scale':3},{'adam_beta2':0},{'adam_beta2':1},{'initial_lr':0},
    {'initial_lr':.041},{'final_lr':float('inf')},{'extra_eval_interval':-1},
    {'frequency_source_policy':'uniform16'},
])
def test_numeric_guards_and_proxy_exclusion(worker,spec,change):
    with pytest.raises(ValueError): worker.validate_spec(dict(spec,**change))


def test_exposure_mapping_and_actual_evaluation_counts(worker,spec):
    settings = worker.validate_spec(spec)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    python_state = random.getstate(); numpy_state = np.random.get_state(); torch_state = torch.get_rng_state().clone()
    for scale in (1,2,4):
        cfg = worker.build_config(dict(settings,exposure_scale=scale))
        m = cfg.pipeline.model
        assert (m.adaptive_warmup_steps,m.occupancy_warmup_steps,m.occupancy_binary_warmup_steps) == (4096//scale,)*3
        assert m.occupancy_update_interval == 16//scale
        assert cfg.pipeline.grid_update_interval == 1024//scale
        for actual in (0,1024,2048,4096,8192,12288,32000):
            virtual, fraction = worker.exposure_phase(actual,scale)
            assert virtual == actual*scale
            assert fraction == min(1.,max(0.,(virtual-4096)/8192))
    assert not worker.is_eval_iteration(30000,5000)
    assert worker.is_eval_iteration(32000,0)
    assert worker.is_eval_iteration(35000,5000)
    assert not worker.is_eval_iteration(35000,0)
    assert worker.validate_limit(settings) == 140000
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1],numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(),torch_state)


def test_own_output_provenance_rejects_foreign_checkpoint(worker,tmp_path):
    clean = tmp_path/'own'; clean.mkdir()
    worker.bind_cold_output(clean,'spec','data',{'code':'sha'})
    worker.bind_cold_output(clean,'spec','data',{'code':'sha'})
    with pytest.raises(ValueError):worker.bind_cold_output(clean,'changed','data',{'code':'sha'})
    foreign = tmp_path/'foreign'; (foreign/'checkpoints').mkdir(parents=True)
    (foreign/'checkpoints/old.ckpt').write_bytes(b'not loaded')
    with pytest.raises(ValueError):worker.bind_cold_output(foreign,'spec','data',{})


def raster_fixture():
    rgb=[torch.arange(24).reshape(2,4,3).float(),torch.arange(27).reshape(3,3,3).float()+100]
    return dict(image_idx=torch.tensor([8,2]),image=rgb,
                mask=[torch.ones(2,4,1,dtype=torch.bool),torch.zeros(3,3,1,dtype=torch.bool)],
                depth_image=[rgb[0][...,:1]+7,rgb[1][...,:1]+7])


def assert_same_rasters(batch,index,gather):
    expected=gather_rasters(batch,index); actual=gather(batch,index)
    for key in expected:assert torch.equal(expected[key],actual[key])
    assert actual['indices'] is index


def test_packed_ragged_gather_identity_mutations_and_bounds():
    batch=raster_fixture(); index=torch.tensor([[2,1,2],[8,0,3]])
    fast=PackedRaggedGather(); assert_same_rasters(batch,index,fast)
    assert fast.builds==1
    assert_same_rasters(batch,index,fast); assert fast.builds==1
    batch['image'][0].add_(.25); assert_same_rasters(batch,index,fast); assert fast.builds==2
    batch['image'][0]=batch['image'][0].clone(); assert_same_rasters(batch,index,fast); assert fast.builds==3
    reordered={key:value.flip(0) if key=='image_idx' else list(reversed(value)) for key,value in batch.items()}
    assert_same_rasters(reordered,index,fast)
    assert fast(reordered,index,keep_full_image=True)['full_image'] is reordered['image']
    with pytest.raises(ValueError):fast(batch,torch.tensor([[8,2,0]]))
    with pytest.raises(ValueError):fast(batch,torch.tensor([[88,0,0]]))
    assert fast(batch,torch.empty(0,3,dtype=torch.long))['image'].shape == (0,3)


def test_gpu_metadata_contract_without_initializing_cuda():
    class SimulatedCUDAMetadata(torch.Tensor):
        @property
        def device(self):return torch.device('cuda:0')
        def cpu(self,*args,**kwargs):return self.as_subclass(torch.Tensor)
        def tolist(self):raise AssertionError('Copy small metadata to CPU first')
    initial_cuda=torch.cuda.is_initialized(); batch=raster_fixture()
    batch['image_idx']=batch['image_idx'].as_subclass(SimulatedCUDAMetadata)
    index=torch.tensor([[2,1,2],[8,0,3]]).as_subclass(SimulatedCUDAMetadata)
    fast=PackedRaggedGather(); assert_same_rasters(batch,index,fast)
    assert fast.cache_info()['source_camera_order']==[8,2]
    assert all(value.device.type=='cpu' for value in fast._cache['packed'].values())
    assert torch.cuda.is_initialized()==initial_cuda


def test_source_helper_import_rejects_cached_other_checkout(worker,monkeypatch,tmp_path):
    fake=types.ModuleType('marshawn_iteration_budget');fake.__file__=str(tmp_path/'other.py')
    monkeypatch.setitem(sys.modules,'marshawn_iteration_budget',fake)
    with pytest.raises(ImportError):worker._source_import('marshawn_iteration_budget')


def test_public_sampling_import_rejects_foreign_overlay(worker,monkeypatch,tmp_path):
    fake=types.ModuleType('native_crop_sampling_v2');fake.__file__=str(tmp_path/'other.py')
    monkeypatch.setitem(sys.modules,'native_crop_sampling_v2',fake)
    with pytest.raises(ImportError):worker._public_sampling_import('native_crop_sampling_v2')


def test_frozen_v7_recipe_and_loop_parity(worker,spec):
    """Public bootstrap may differ; every inherited v7 function keeps its body."""
    import yaml
    frozen=Path(os.environ.get('LOOKCLOSER_TEST_V7_WORKER',
        '/fsx/tmp/marshawn_native_crop_tuning_20261108/code_overlay_v7/train_native_crop.py'))
    if not frozen.exists():pytest.skip('Frozen v7 comparison fixture unavailable')
    class StripProvenanceChecks(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_source_import':
                return None
            return self.generic_visit(node)
    old_tree=ast.parse(frozen.read_text());new_tree=ast.parse((ROOT/'scripts/train_native_crop.py').read_text())
    old_functions={node.name:node for node in old_tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new_functions={node.name:node for node in new_tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for name,node in old_functions.items():
        assert ast.dump(StripProvenanceChecks().visit(new_functions[name])) == ast.dump(node),name
    original_path=list(sys.path)
    try:
        loader=importlib.util.spec_from_file_location('_native_crop_frozen_v7_test',frozen)
        baseline=importlib.util.module_from_spec(loader);loader.loader.exec_module(baseline)
    finally:sys.path[:]=original_path
    assert worker.DEFAULTS==baseline.DEFAULTS
    for change in ({},{'corrected_arm_allocator':True},{'train_rays_per_batch':16384,'exposure_scale':4,
        'lr_decay_steps':50000,'fr_switch_step':18986,'radius':1,'distortion_final':.01,'adam_beta2':.99}):
        settings=worker.validate_spec(dict(spec,**change))
        assert yaml.dump(worker.build_config(settings))==yaml.dump(baseline.build_config(settings))
    for name in ('marshawn_semantic_admission','marshawn_iteration_budget','nerfstudio.configs.method_configs'):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(SOURCE_ROOT)
    for name in ('native_crop_sampling','native_crop_sampling_v2'):
        assert Path(sys.modules[name].__file__).resolve()==ROOT/'scripts'/(name+'.py')
        assert sha(ROOT/'scripts'/(name+'.py'))==sha(frozen.parent/(name+'.py'))


@pytest.mark.parametrize('step',[0,8192])
def test_packed_body80_rng_and_proposal_parity(step):
    from native_crop_sampling import NativeCropSamplingAdapter
    from native_crop_sampling_v2 import NativeCropSamplingAdapterV2
    batch=raster_fixture();batch['image_idx']=torch.tensor([1,0])
    batch={key:value if key=='image_idx' else list(reversed(value)) for key,value in batch.items()}
    rows=[]
    for height,width in ((2,4),(3,3)):
        flat=np.arange(height*width,dtype=np.int64)
        rows.append(dict(core=np.ones((height,width),bool),pools=dict(core=flat[::2],band=flat[1::2],valid=flat),
            proposal_means=np.array([.7,.8,.6,.5]),
            fas=dict(probabilities=np.array([.5,.5]+[0.]*14),cells=[[flat[::2]],[flat[1::2]]]+[[] for _ in range(14)])))
    old=NativeCropSamplingAdapter.__new__(NativeCropSamplingAdapter)
    old.rows=rows;old.step=step;old.source_hits=np.zeros(4,np.int64)
    old.sampler=types.SimpleNamespace(num_rays_per_batch=4096,config=types.SimpleNamespace(keep_full_image=False))
    candidate=copy.copy(old);candidate.__class__=NativeCropSamplingAdapterV2
    candidate.source_hits=old.source_hits.copy();candidate.packed_gather=PackedRaggedGather()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    reference,mean=old._body_draw(batch)
    expected_rng=(random.random(),np.random.random(8),torch.rand(8))
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    result,candidate_mean=candidate._body_draw(batch)
    assert mean==candidate_mean
    for key in reference:assert torch.equal(reference[key],result[key])
    assert random.random()==expected_rng[0]
    assert np.array_equal(np.random.random(8),expected_rng[1])
    assert torch.equal(torch.rand(8),expected_rng[2])
    assert np.array_equal(old.source_hits,candidate.source_hits)
    assert old.last_counts==candidate.last_counts
