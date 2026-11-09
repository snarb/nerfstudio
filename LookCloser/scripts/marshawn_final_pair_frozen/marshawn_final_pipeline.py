"""Matched native final recipe: scalar RGB, observed geometry, radius0/1 guard."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline
from nerfstudio.engine.callbacks import TrainingCallback,TrainingCallbackLocation
from marshawn_los_objective_frozen import FreeSpaceObjective
from marshawn_surface_band_objective import SurfaceBandObjective
from staged_native_hash import file_sha


class FinalNativePipeline(DynamicBatchPipeline):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        cfg=self.config;radius=int(cfg.final_grid_radius)
        if radius not in (0,1):raise ValueError('Only reviewed radius0/1 policies')
        for name,digest in cfg.final_source_hashes.items():
            if Path(name).name!=name or file_sha(Path(__file__).parent/name)!=digest:
                raise ValueError('Frozen final helper changed: '+name)
        data=Path(cfg.datamanager.dataparser.data)
        gate=json.loads(Path(cfg.final_color_gate).read_text())
        if file_sha(cfg.final_color_gate)!=cfg.final_color_gate_sha256 or gate['approved']is not True:
            raise ValueError('Scalar RGB gate mismatch')
        if data.resolve()!=Path(gate['data']).resolve() or file_sha(data/'transforms.json')!=gate['transforms_sha256']:
            raise ValueError('Final source/geometry changed')
        for name,digest in gate['files_sha256'].items():
            if file_sha(data/name)!=digest:raise ValueError('Final RGB input changed: '+name)
        contract=json.loads((data/'color_contract.json').read_text())
        if contract['eval_rgb_used_to_fit_gains']is not False:raise ValueError('No eval RGB gain fitting allowed')
        if file_sha(cfg.final_geometry_preparation)!=cfg.final_geometry_preparation_sha256:
            raise ValueError('Original geometry provenance changed')
        variant=cfg.final_geometry_variant
        if variant not in ('scene_tight','scene_medium','scene_loose'):raise ValueError('Explicit approved scene variant required')
        bounds=self.model.scene_box.aabb.detach().cpu().numpy()
        free=FreeSpaceObjective(cfg.final_geometry_manifest,variant,bounds,data/'transforms.json',
            cfg.final_geometry_preparation,weight=.01,rays=128,samples=64,seed=42,device=self.device)
        band=SurfaceBandObjective(cfg.final_geometry_manifest,variant,bounds,data/'transforms.json',
            cfg.final_geometry_preparation,weight=.01,rays=128,samples=16,seed=42,device=self.device)
        if free.receipt['archive_sha256']!=band.receipt['archive_sha256']:raise ValueError('Geometry archive/order mismatch')
        if list(free.parameters())or list(band.parameters()):raise ValueError('Auxiliaries must have no parameters')
        for name,value in [('_final_free',free),('_final_band',band),('_guard_radius',radius),
            ('_actual_core',None),('_actual_active',None),('_actual_threshold',None),('_last_core_step',None)]:
            object.__setattr__(self,name,value)

    def get_train_loss_dict(self,step):
        outputs,losses,metrics=super().get_train_loss_dict(step)
        free,free_metrics=self._final_free(self.model.field,int(step))
        ids=self._final_free.sample_batch(int(step))['indices']
        surface,surface_metrics=self._final_band(self.model.field,int(step),ids)
        losses['observed_sfm_free_space']=free
        losses['observed_sfm_surface_band']=surface
        metrics.update(free_metrics);metrics.update(surface_metrics)
        return outputs,losses,metrics

    @staticmethod
    def one_halo(core):
        return torch.nn.functional.max_pool3d(core[None].float(),3,stride=1,padding=1)[0].bool()

    @torch.no_grad()
    def guard_after_native_core(self,step):
        if step%16:raise ValueError('Guard only at native16-step update')
        if self._last_core_step==step:return
        grid=self.model.occupancy_grid
        threshold=torch.clamp(grid.occs[grid.occs>=0].mean(),max=.01)
        expected=(grid.occs>threshold).reshape_as(grid.binaries)
        if not torch.equal(expected,grid.binaries):
            raise RuntimeError('Native core not fresh; callback order or double-halo error')
        core=grid.binaries.clone()
        active=core if self._guard_radius==0 else self.one_halo(core)
        grid.binaries.copy_(active)
        object.__setattr__(self,'_actual_core',core.clone())
        object.__setattr__(self,'_actual_active',active.clone())
        object.__setattr__(self,'_actual_threshold',float(threshold))
        object.__setattr__(self,'_last_core_step',int(step))

    def get_training_callbacks(self,training_callback_attributes):
        callbacks=super().get_training_callbacks(training_callback_attributes)
        native=[c for c in callbacks if getattr(c.func,'__name__','')=='update_occupancy_grid']
        if len(native)!=1:raise RuntimeError('Expected exactly one frozen native occupancy callback')
        callbacks.append(TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            update_every_num_iters=16,func=self.guard_after_native_core))
        return callbacks

    def export_grid_for_checkpoint(self,checkpoint,step):
        """Called AFTER native synchronous save; external sidecars add no NN state."""
        checkpoint=Path(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)['pipeline']
        def value(suffix):
            arrays=[v for k,v in saved.items()if k.endswith(suffix)]
            if not arrays or not all(torch.equal(arrays[0],v)for v in arrays[1:]):
                raise ValueError('Checkpoint shared-grid aliases disagree '+suffix)
            return arrays[0]
        active=value('occupancy_grid.binaries');ema=value('occupancy_grid.occs')
        aabb=value('occupancy_grid.aabbs');resolution=value('occupancy_grid.resolution')
        if self._actual_core is None or self._actual_active is None:
            raise ValueError('Actual native core cache required; never infer authoritative core with CPUmean')
        core=self._actual_core.cpu();cached_active=self._actual_active.cpu()
        expected=core if self._guard_radius==0 else self.one_halo(core)
        if not torch.equal(active,expected):raise ValueError('Stored grid violates declared policy')
        if not torch.equal(active,cached_active):
            raise ValueError('Cached active mask differs from checkpoint')
        cpu_threshold=torch.clamp(ema[ema>=0].mean(),max=.01)
        cpu_diagnostic=(ema>cpu_threshold).reshape_as(active)
        folder=Path(self.config.final_grid_artifact_dir);folder.mkdir(parents=True,exist_ok=True)
        archive=folder/(checkpoint.stem+'_grid.npz')
        np.savez_compressed(archive,core=core.numpy(),active=active.numpy(),ema=ema.numpy(),
            world_aabb=aabb[0].reshape(2,3).numpy(),cascade_aabbs=aabb.numpy(),resolution=resolution.numpy())
        record=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),
            checkpoint_step=int(step),core_update_step=self._last_core_step,
            grid_archive=str(archive),grid_archive_sha256=file_sha(archive),radius=self._guard_radius,
            policy='NativeEMA core then one noncumulative cube-radius1 guard'if self._guard_radius else'Unchanged nativeEMA core',
            native_period=16,gpu_native_threshold=self._actual_threshold,
            cpu_threshold_diagnostic=float(cpu_threshold),cpu_core_difference_diagnostic=int((cpu_diagnostic!=core).sum()),
            core_cells=int(core.sum()),active_cells=int(active.sum()),
            world_aabb=aabb[0].reshape(2,3).tolist(),source_sha256=file_sha(__file__),
            geometry_variant=self.config.final_geometry_variant,
            geometry_archive_sha256=self._final_free.receipt['archive_sha256'],
            rgb_gate_sha256=self.config.final_color_gate_sha256,
            native_model_source_sha256=file_sha(inspect.getsourcefile(type(self.model))),
            native_field_source_sha256=file_sha(inspect.getsourcefile(type(self.model.field))),
            no_extra_nn_state=True,constructor_does_not_dilate_loaded_masks=True)
        (folder/(checkpoint.stem+'_grid.json')).write_text(json.dumps(record,indent=2)+'\n')
        return record
