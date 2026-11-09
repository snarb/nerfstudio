"""Explicit GB300/native72 port of the frozen recipe; original helpers stay immutable."""
import copy
import hashlib
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import torch
from common import atomic, read, sha


def bootstrap(spec):
    root=Path(spec['source_code']);frozen=Path(spec['frozen_helpers'])
    sys.path[:0]=[str(frozen),str(root),str(root/'LookCloser/scripts')]
    # Load the already-built ARM64 extension; never trigger an unqualified JIT.
    if spec.get('nerfacc_extension') and 'nerfacc.csrc' not in sys.modules:
        s=importlib.util.spec_from_file_location('nerfacc_cuda',spec['nerfacc_extension'])
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m);sys.modules['nerfacc_cuda']=m
        import nerfacc
        nerfacc.csrc=m;sys.modules['nerfacc.csrc']=m
    return root,frozen


def load_support(manifest_path,expected_sha,data_dir,bounds,resolution,identity=False):
    if identity:raise ValueError('Identity admission is not a production setting')
    path=Path(manifest_path);root=Path(data_dir);m=read(path)
    if sha(path)!=expected_sha or sha(path.parent/'grid.npz')!=m['grid_sha256']:raise ValueError('Admission changed')
    if sha(root/'ready.json')!=m['ready_sha256'] or sha(root/'transforms.json')!=m['transforms_sha256']:raise ValueError('Admission dataset binding differs')
    tf=read(root/'transforms.json');names=sorted(f['physical_camera'] for f in tf['frames'] if f['file_path'] in tf['train_filenames'])
    if names!=sorted(r['camera'] for r in m['records']):raise ValueError('Admission train leakage')
    for r in m['records']:
        if sha(root/'semantics'/(r['camera']+'.npz'))!=r['semantic_sha256']:raise ValueError('Admission masks changed')
    with np.load(path.parent/'grid.npz') as z:mask=z['mapped'].copy();box=z['aabb'].copy()
    np.testing.assert_allclose(bounds,box,rtol=0,atol=2e-6)
    if tuple(resolution)!=(128,)*3 or mask.dtype!=bool:raise ValueError('Expected original128cube grid')
    if hashlib.sha256(mask.tobytes()).hexdigest()!=m['mapped_sha256']:raise ValueError('Admission mapping changed')
    return mask,dict(manifest_sha256=expected_sha,train_camera_count=len(names),frame=read(root/'ready.json')['frame'],identity_control=False,mapped_bool_sha256=m['mapped_sha256'])


def config(spec,resume=None):
    bootstrap(spec)
    import native63_geometry
    native63_geometry.load_support=load_support
    from native63_semantic_admission import SemanticAdmissionModelConfig
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
    from nerfstudio.data.scene_box import SceneBox
    class Parser(Nerfstudio):
        def _generate_dataparser_outputs(self,split='train'):
            c=self.config
            if c.auto_scale_poses or c.center_method!='none' or c.orientation_method!='none' or c.scale_factor!=1:raise ValueError('Identity normalization required')
            outputs=super()._generate_dataparser_outputs(split)
            outputs.scene_box=SceneBox(aabb=torch.tensor(read(Path(c.data)/'box.json')['aabb'],dtype=torch.float32))
            return outputs
    # Give the config a stable, reloadable class path for checkpoint consumers.
    Parser.__module__=__name__;Parser.__qualname__='SequenceParser';globals()['SequenceParser']=Parser
    cfg=copy.deepcopy(method_configs['lookcloser'])
    cfg.output_dir=Path(spec['output']);cfg.experiment_name='worker';cfg.timestamp='frozen'
    cfg.vis='tensorboard';cfg.max_num_iterations=spec['hard_max_iterations'];cfg.machine.seed=42;cfg.machine.num_devices=1
    cfg.logging.local_writer.enable=False;cfg.logging.profiler='none';cfg.logging.csv_writer.enable=False
    cfg.save_only_latest_checkpoint=False;cfg.viewer.quit_on_train_completion=True
    cfg.load_checkpoint=Path(resume) if resume else None
    cfg.load_dir=cfg.load_config=cfg.eval_checkpoint=cfg.load_step=None
    pipe=cfg.pipeline;pipe.training_seed=42
    model=pipe.model=SemanticAdmissionModelConfig(**{k:v for k,v in vars(pipe.model).items() if k!='_target'},semantic_manifest=spec['semantic_admission']['manifest'],semantic_manifest_sha256=spec['semantic_admission']['manifest_sha256'],semantic_data=spec['data'],semantic_identity_control=False)
    model.training_seed=42;model.correct_sh_directions=True;model.density_activation='trunc_exp'
    model.log2_hashmap_size=25;model.hash_features_per_level=2;model.max_res=spec.get('max_res',8192)
    model.occupancy_dilation_radius=0;model.adaptive_coarse_step_size=spec.get('coarse_step',.0015625)
    model.adaptive_warmup_steps=model.occupancy_warmup_steps=model.occupancy_binary_warmup_steps=4096
    model.occupancy_update_interval=16;model.corrected_arm_allocator=True;model.adaptive_min_step_size=1e-4
    model.max_steps_per_ray=spec.get('max_steps_per_ray',1024);model.adaptive_interval_level_mode=spec.get('adaptive_interval_level_mode','midpoint')
    model.adaptive_max_step_size=.1;model.adaptive_min_frequency_level=0.;model.adaptive_max_frequency_level=None
    model.occupancy_occ_thre=.01;model.occupancy_ema_decay=.95;model.eval_num_rays_per_chunk=4096
    if spec.get('format')=='exr':
        model.rgb_output_parameterization='linear_softplus'
        model.hdr_linear_scale=1.;model.hdr_initial_radiance=.18;model.pq_nits_per_scene_unit=52.13360318135631
    pipe.frequency_projection_mode='legacy';pipe.grid_update_interval=spec.get('frequency_grid_update_interval',1024)
    model.feature_reweighting_strength=spec.get('fr_initial',1.)
    pipe.feature_reweighting_switch_step=75941;pipe.feature_reweighting_after_switch=.3
    pipe.target_num_samples_per_batch=0;pipe.dynamic_rays_start_step=4096;pipe.dynamic_rays_max=32768
    opt=cfg.optimizers['fields'];opt['optimizer'].betas=(.9,.99);opt['optimizer'].lr=.01
    opt['scheduler'].lr_final=.0001;opt['scheduler'].max_steps=200000
    dm=pipe.datamanager;dm.train_num_rays_per_batch=spec.get('train_rays_per_batch',8192);dm.cpu_fas_prefetch=False;dm.cache_train_rays=False
    if spec.get('sampling_memmap'):
        dm.images_on_gpu=True;dm.masks_on_gpu=True
    dm.data=Path(spec['data']);dp=dm.dataparser;dp._target=Parser;dp.data=dm.data
    dp.orientation_method=dp.center_method='none';dp.auto_scale_poses=False;dp.scale_factor=dp.scene_scale=1.;dp.downscale_factor=1;dp.eval_mode='filename'
    return cfg


def install_sampling(pipeline,spec):
    """Keep archived proposal/RNG/normalizer math; adapt camera and actor membership."""
    _,frozen=bootstrap(spec)
    source=frozen/'native63_sampling_experiment.py';text=source.read_text()
    replacements={
        "for actor in ('black','grey'):":"for actor in rec['actors']:",
        "row[actor]&row['valid']":"row['actor_union']&row['valid']",
        "if c in {'DC','HC','KA'}:":"if c in HELDOUT:",
        'balanced_actor_bbox_top30_proxy':'balanced_current_SAM_actor_head',
    }
    for before,after in replacements.items():
        if before not in text:raise ValueError('Frozen adapter source changed: '+before)
        text=text.replace(before,after)
    module=types.ModuleType('sequence_sampling_runtime');module.__file__=str(source)
    sys.modules[module.__name__]=module
    exec(compile(text,str(source),'exec'),module.__dict__)
    ready=read(Path(spec['data'])/'ready.json');tf=read(Path(spec['data'])/'transforms.json')
    train_names={Path(p).stem for p in tf['train_filenames']}
    heldout=set(ready['validation'])-train_names
    module.HELDOUT=heldout
    def validate(parameters):
        if parameters.get('head_fraction') not in (.1,.15,.2):raise ValueError('ExplicitSAM proposal fraction required')
        path=Path(parameters['head_proxy_manifest'])
        if sha(path)!=parameters['head_proxy_manifest_sha256']:raise ValueError('Head manifest changed')
        m=read(path)
        if set(m['train_cameras'])!=train_names or m['kind']!='SEQUENCE_NATIVE_SAM3_HEAD_FACE_IMPORTANCE_V1':raise ValueError('Head source membership differs')
        return parameters
    module.validate=validate
    original_load=np.load
    scratch=Path(spec['sampling_memmap']) if spec.get('sampling_memmap') else None
    class ArrayProxy:
        def __init__(self,path):self.path=path
        def copy(self):return original_load(self.path,mmap_mode='c',allow_pickle=False)
    class ArchiveProxy:
        def __init__(self,path):self.path=Path(path)
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,key):
            directory=scratch/(self.path.stem+'_'+sha(self.path)[:12]);directory.mkdir(parents=True,exist_ok=True)
            dest=directory/(key+'.npy')
            if not dest.exists():
                with original_load(self.path,allow_pickle=False) as z:
                    tmp=dest.with_suffix('.tmp.npy');np.save(tmp,z[key],allow_pickle=False);tmp.replace(dest)
            return ArrayProxy(dest)
    def mapped_load(path,*args,**kwargs):
        if scratch and str(Path(spec['data'])/'semantics') in str(path) and str(path).endswith('.npz'):return ArchiveProxy(path)
        return original_load(path,*args,**kwargs)
    try:
        if scratch:np.load=mapped_load
        adapter=module.ExperimentalSampling(pipeline,spec['data'],'actor','body80',64,spec)
    finally:np.load=original_load
    pipeline.marshawn_sampling_adapter=adapter
    if spec.get('format')=='exr':
        from marshawn_tuning_sampling import weighted_charbonnier
        original=pipeline.model.get_loss_dict
        def loss(outputs,batch,metrics_dict=None):
            result=original(outputs,batch,metrics_dict)
            if 'actor_confidence' in batch:
                result['rgb_loss']=weighted_charbonnier(display_rgb(outputs['rgb']),display_rgb(batch['image'].to(outputs['rgb'].device)),batch['actor_confidence'],batch['actor_proposal_mean_weight'])
            return result
        pipeline.model.get_loss_dict=loss
    if len(adapter.rows)!=ready['train']:raise ValueError('Actual sampler camera count differs')
    atomic(Path(spec['output'])/'sampling_port.json',dict(original_sha256=sha(source),executed_source_sha256=hashlib.sha256(text.encode()).hexdigest(),replacements=replacements,train_count=len(adapter.rows),heldout=sorted(heldout),head_source='four current SAM person head pools; no bbox proxy',actor_storage='black=union four performers; grey=empty compatibility slot; metric instances separate'))
    return adapter


def display_rgb(linear):
    value=linear.float().clamp_min(0)*1.6;value=value/(1+value)
    return torch.where(value<=.0031308,12.92*value,1.055*value.clamp_min(1e-8).pow(1/2.4)-.055)


class GPUCacheGather:
    """One immutable GPU copy of RGB rasters; original sample indices unchanged."""
    def __init__(self):self.source=None;self.data=None;self.signature=None
    def __call__(self,batch,indices,keep_full_image=False):
        if isinstance(batch['image'],torch.Tensor):
            from native_crop_sampling import gather_rasters
            return gather_rasters(batch,indices,keep_full_image=keep_full_image)
        signature=(tuple(batch['image_idx'].detach().cpu().tolist()),tuple((k,tuple((v.data_ptr(),v._version,tuple(v.shape)) for v in values)) for k,values in batch.items() if k!='image_idx' and values is not None))
        if self.source is not batch or self.signature!=signature:
            images=batch['image'];ids=batch['image_idx'].detach().cpu().long().tolist()
            if not isinstance(images,list):raise ValueError('Expected native variable-size list')
            if len(ids)!=len(images) or len(set(ids))!=len(ids):raise ValueError('Unique actual camera IDs required')
            self.lookup={c:i for i,c in enumerate(ids)}
            self.widths=np.array([v.shape[1] for v in images]);self.heights=np.array([v.shape[0] for v in images]);sizes=np.array([v.shape[0]*v.shape[1] for v in images]);self.offsets=np.r_[0,np.cumsum(sizes)[:-1]]
            self.data={k:torch.cat([v.reshape(-1,*v.shape[2:]) for v in values]).to('cuda') for k,values in batch.items() if k!='image_idx' and values is not None}
            # With native images_on_gpu, point source rasters at packed views to
            # release the duplicate allocation. Tensor values/order are unchanged.
            if images[0].device.type=='cuda':
                for key,values in batch.items():
                    if key=='image_idx' or values is None:continue
                    packed=self.data[key]
                    for i,value in enumerate(values):
                        start=int(self.offsets[i]);stop=start+int(self.heights[i]*self.widths[i])
                        values[i]=packed[start:stop].reshape(value.shape)
                signature=(tuple(batch['image_idx'].detach().cpu().tolist()),tuple((k,tuple((v.data_ptr(),v._version,tuple(v.shape)) for v in values)) for k,values in batch.items() if k!='image_idx' and values is not None))
            self.source=batch;self.signature=signature
        a=indices.detach().cpu().long().numpy();pos=np.array([self.lookup[int(c)] for c in a[:,0]])
        if ((a[:,1]<0)|(a[:,2]<0)|(a[:,1]>=self.heights[pos])|(a[:,2]>=self.widths[pos])).any():raise ValueError('Sampled pixel outside native crop')
        flat=torch.as_tensor(self.offsets[pos]+a[:,1]*self.widths[pos]+a[:,2],device='cuda')
        result={k:v[flat] for k,v in self.data.items()};result['indices']=indices
        if keep_full_image:result['full_image']=batch['image']
        return result


def install_core(pipeline,spec):
    from native_crop_opaque_core_prior import core_loss
    root=Path(spec['data']);names=[Path(p).stem for p in pipeline.datamanager.train_dataset.image_filenames]
    manifest=read(root/'cores/manifest.json')
    if set(manifest['train_cameras'])!=set(names) or set(manifest['records'])!=set(names):raise ValueError('Opaque core train membership differs')
    packed=[];widths=[];sizes=[]
    for name in names:
        rec=manifest['records'][name]
        if sha(root/'cores'/(name+'.npz'))!=rec['core_sha256'] or sha(root/'semantics'/(name+'.npz'))!=rec['source_semantic_sha256']:raise ValueError('Opaque core/source changed')
        with np.load(root/'cores'/(name+'.npz')) as z:core=z['core']
        packed.append(core.reshape(-1));widths.append(core.shape[1]);sizes.append(core.size)
    masks=np.concatenate(packed);offsets=np.r_[0,np.cumsum(sizes)[:-1]];widths=np.array(widths)
    model=pipeline.model;original=model.get_loss_dict
    def loss(outputs,batch,metrics_dict=None):
        result=original(outputs,batch,metrics_dict)
        if model.training:
            ids=batch['indices'].detach().cpu().long().numpy();c,y,x=ids.T
            flag=torch.from_numpy(masks[offsets[c]+y*widths[c]+x])
            result['opaque_core_deficit_loss']=core_loss(outputs['accumulation'],flag,.01,pipeline.marshawn_sampling_adapter.step)
        return result
    model.get_loss_dict=loss
