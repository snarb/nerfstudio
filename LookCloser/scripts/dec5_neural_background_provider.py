"""Hash-bound frozen background volume layered over the existing ray fallback."""
from pathlib import Path
import json
import random
import weakref
import numpy as np
import torch
import yaml
from nerfstudio.cameras.rays import RayBundle
from train_dec5_neural_background import install_empty_subject_box
from dec5_sequence_worker import sha


class HybridFrozenBackground:
    def __init__(self,base,foreground,dependency):
        from dec5_background_separation import network_fingerprint
        self.base=base;self.frames=base.frames;self.checkpoint_sha256=base.checkpoint_sha256
        self.foreground_model=weakref.ref(foreground.model)
        self.start_step=int(dependency.get('start_step',0))
        if self.start_step<0:raise ValueError('Negative background switch step')
        checkpoint=Path(dependency['checkpoint']);config=Path(dependency['config'])
        source=Path(dependency['source_metadata'])
        for path,key in [(checkpoint,'checkpoint_sha256'),(config,'config_sha256'),(source,'source_metadata_sha256')]:
            if sha(path)!=dependency[key]:raise ValueError('Neural background dependency changed: '+key)
        data=Path(foreground.datamanager.config.data)
        bg_meta=json.loads(source.read_text());fg_meta=json.loads((data/'transforms.json').read_text())
        if bg_meta['frame_id']!=fg_meta['frame_id']:raise ValueError('Background volume belongs to another frame')
        bg={f['physical_camera']:f for f in bg_meta['frames'] if f['file_path'] in bg_meta['train_filenames']}
        fg={f['physical_camera']:f for f in fg_meta['frames'] if f['file_path'] in fg_meta['train_filenames']}
        if set(bg)!=set(fg) or len(bg)!=64:raise ValueError('Background/foreground train splits differ')
        for name,f in fg.items():
            b=bg[name];np.testing.assert_allclose(f['transform_matrix'],b['transform_matrix'],atol=1e-8,rtol=0)
            np.testing.assert_allclose([f['fl_x']/f['w'],f['fl_y']/f['h'],f['cx']/f['w'],f['cy']/f['h']],
                [b['fl_x']/b['w'],b['fl_y']/b['h'],b['cx']/b['w'],b['cy']/b['h']],atol=1e-6,rtol=0)
        cfg=yaml.load(config.read_text(),Loader=yaml.Loader)
        cfg.pipeline.datamanager.data=data;cfg.pipeline.datamanager.dataparser.data=data
        cfg.pipeline.model.eval_num_rays_per_chunk=8192
        python_state=random.getstate();numpy_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self.pipeline=cfg.pipeline.setup(device=foreground.device,test_mode='inference')
        finally:
            random.setstate(python_state);np.random.set_state(numpy_state)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        hole=state.get('runtime',{}).get('background_empty_subject_half_width')
        if hole!=.18:raise ValueError('Missing exact background empty-volume convention')
        self.pipeline.load_pipeline(state['pipeline'],state['step']);del state
        install_empty_subject_box(self.pipeline.model.field,hole)
        self.pipeline.eval();self.pipeline.model.requires_grad_(False)
        # scene_scale changes only the scene bounds; ray coordinates must agree.
        child={Path(p).stem:i for i,p in enumerate(self.pipeline.datamanager.train_dataset.image_filenames)}
        for i,path in enumerate(foreground.datamanager.train_dataset.image_filenames):
            j=child[Path(path).stem]
            torch.testing.assert_close(foreground.datamanager.train_dataset.cameras.camera_to_worlds[i].cpu(),
                self.pipeline.datamanager.train_dataset.cameras.camera_to_worlds[j].cpu(),rtol=0,atol=1e-5)
        self.before=network_fingerprint(self.pipeline.model);self.dependency=dependency

    @torch.no_grad()
    def predict(self,origins,directions):
        fallback=self.base.predict(origins,directions)
        if self.foreground_model().current_train_step<self.start_step:return fallback
        rays=RayBundle(origins=origins.detach(),directions=directions.detach(),
            pixel_area=torch.ones_like(origins[...,:1]),camera_indices=torch.zeros_like(origins[...,:1],dtype=torch.long))
        values=self.pipeline.model(rays);alpha=values['accumulation']
        if alpha.ndim==values['rgb'].ndim-1:alpha=alpha[...,None]
        return values['rgb']+(1-alpha).clamp(0,1)*fallback

    def verify(self):
        from dec5_background_separation import network_fingerprint
        self.base.verify()
        if network_fingerprint(self.pipeline.model)!=self.before:
            raise ValueError('Frozen neural background weights or buffers changed')
