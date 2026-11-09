"""Opt-in continuous field visibility for rejected hair rays; no fitting."""
from pathlib import Path
import json
import numpy as np,torch,yaml
from dec5_hair import ray_to_z,packed_depth_quantile_vectorized
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import sha


class DirectHairVisibility:
    def __init__(self,data,training,checkpoint,source_frames,expected_checkpoint_sha,max_queries=2000000):
        if sha(checkpoint)!=expected_checkpoint_sha:raise ValueError('Direct visibility must use the same target/source checkpoint')
        request=json.load(open(training/'request.json'))
        if request['data_sha256']!=sha(data/'transforms.json'):raise ValueError('Direct field dataset mismatch')
        cfg=yaml.load(next(training.rglob('config.yml')).read_text(),Loader=yaml.Loader)
        cfg.pipeline.datamanager.data=data;cfg.pipeline.datamanager.dataparser.data=data;cfg.pipeline.datamanager.cache_images_type='uint8'
        pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();self.pipe=pipe;self.model=pipe.model;self.model.current_train_step=state['step'];self.before=fingerprint(self.model)
        if self.model.config.ray_sampling_mode!='adaptive':raise ValueError('Native adaptive field required')
        d=json.load(open(data/'transforms.json'));by={Path(f['file_path']).name:f for f in d['frames']};train=pipe.datamanager.train_dataset;frames=[by[Path(p).name] for p in train.image_filenames];names={f['physical_camera']:i for i,f in enumerate(frames)}
        raw=np.array([f['transform_matrix'] for f in frames]);normal=train.cameras.camera_to_worlds.cpu().numpy();R=normal[0,:3,:3]@raw[0,:3,:3].T;scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);offset=normal[0,:3,3]-scale*R@raw[0,:3,3]
        if np.max(np.linalg.norm(scale*raw[:,:3,3]@R.T+offset-normal[:,:3,3],axis=-1))>1e-4:raise ValueError('Camera alignment mismatch')
        self.scale=float(scale);self.cameras=train.cameras.to('cuda');self.sources=[]
        for source in source_frames:
            i=names[source['physical_camera']];f=frames[i];np.testing.assert_allclose(source['transform_matrix'],f['transform_matrix'],atol=1e-10,rtol=0)
            self.sources.append((i,f,np.array([f['w']/source['w'],f['h']/source['h']],np.float32)))
        self.queries=0;self.nonzero=0;self.max_queries=max_queries;self.checkpoint_sha=sha(checkpoint)

    @torch.inference_mode()
    def __call__(self,index,uv,observed,semantics,eligible):
        selected=torch.nonzero(eligible&(semantics[:,0]>.5),as_tuple=False).flatten()
        if not len(selected):return observed
        if self.queries+len(selected)>self.max_queries:raise RuntimeError('Direct-source query cap reached; no candidate promoted')
        self.queries+=len(selected);i,frame,ratio=self.sources[index];new=observed.clone()
        for begin in range(0,len(selected),256):
            positions=selected[begin:begin+256];coords=uv[positions]*uv.new_tensor(ratio)
            rays=self.cameras[i:i+1].generate_rays(0,coords=coords[:,[1,0]],keep_shape=False);v=self.model(rays)
            arrays=[v[k].detach().float().cpu().numpy() for k in ['packed_spacing_starts','packed_spacing_ends','packed_weights']]
            distance=packed_depth_quantile_vectorized(*arrays,v['packed_ray_indices'].cpu().numpy(),rays.nears.cpu().numpy(),rays.fars.cpu().numpy())
            z=ray_to_z(distance/self.scale,coords.cpu().numpy(),frame);valid=np.isfinite(z)&(z>0)&(v['accumulation'].reshape(-1).cpu().numpy()>=.1)
            self.nonzero+=int(valid.sum());new[positions]=torch.as_tensor(np.where(valid,z,0),device=uv.device,dtype=observed.dtype)
        return new

    def receipt(self):
        if fingerprint(self.model)!=self.before:raise ValueError('Direct visibility changed the frozen field')
        return dict(queries=self.queries,valid_depth_queries=self.nonzero,max_queries=self.max_queries,checkpoint_sha256=self.checkpoint_sha,
            frozen_state_unchanged=True,mode='rejected raster source rays only, semantic hair probability >.5, exact projected source coordinate',
            original_visibility_tolerance='abs(source_z-target_z) <= .01*target_z+.002',uses_validation_rgb=False,uses_test_rgb=False)
