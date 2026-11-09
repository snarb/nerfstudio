"""Actor-only confidence training, with unchanged RGB and explicit proposal units.

Pure contract/math helpers import without nerfstudio. Native classes are loaded
only on demand; no model, optimizer, GPU work or training starts on import.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np

POLICIES=('hard','W32','W64')

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def weighted_scalar_mean(values,confidence,proposal_mean_weight):
    """E_uniformCommon[w*f]/FIXED E_uniformCommon[w], never batch self-normalized."""
    import torch
    w=confidence.to(values).reshape(-1);v=values.reshape(-1)
    if w.shape!=v.shape or not torch.isfinite(w).all()or (w<0).any()or (w>1).any():
        raise ValueError('Matched finite confidence in[0,1] required')
    if not math.isfinite(float(proposal_mean_weight))or not 0<float(proposal_mean_weight)<=1:
        raise ValueError('Frozen positive global proposal mean required')
    return (v*w).mean()/float(proposal_mean_weight)

def draw_uniform_proposal(flat_eligible,batch_size,height,width,*,generator=None):
    """Global uniform pixels, not uniform cameras and not confidence sampling."""
    import torch
    if flat_eligible.ndim!=1 or not len(flat_eligible)or batch_size<1:
        raise ValueError('Nonempty frozen global proposal required')
    chosen=flat_eligible[torch.randint(len(flat_eligible),(batch_size,),generator=generator,device=flat_eligible.device)]
    image=chosen//(height*width);yx=chosen%(height*width)
    return torch.stack([image,yx//width,yx%width],-1).long()

class ActorConfidenceStore:
    """External immutable source data, not NN parameters/checkpoint buffers.

Manifest rows: camera, split(train/eval), rgb, rgb_sha256, archive,
archive_sha256. NPZ: hard/W32/W64 float32, proposal/source_ineligible bool.
The isolated transforms use the SAME common proposal as mask_path for every
policy. Semantic eval masks/native RGB have a separate, frozen domain.
"""
    def __init__(self,manifest,policy):
        if policy not in POLICIES:raise ValueError('Explicit hard/W32/W64 policy required')
        self.path=Path(manifest);self.manifest=json.loads(self.path.read_text());self.policy=policy
        self.rows={};self.normalizers={};self.sha256=file_sha(self.path)
        for split in ['train','eval']:
            rows=[r for r in self.manifest['rows']if r['split'].lower()==split]
            if not rows:continue
            out=[];sums={p:0. for p in POLICIES};count=0
            for row in rows:
                rgb=Path(row['rgb']);archive=Path(row['archive'])
                if file_sha(rgb)!=row['rgb_sha256']or file_sha(archive)!=row['archive_sha256']:
                    raise ValueError('Actor source/hash mismatch')
                with np.load(archive,allow_pickle=False)as z:
                    proposal=z['proposal'].copy();veto=z['source_ineligible'].copy()
                    if proposal.dtype!=np.bool_ or veto.dtype!=np.bool_ or proposal.shape!=veto.shape or proposal.ndim!=2:
                        raise ValueError('Equal boolean proposal/veto rasters required')
                    if (proposal&veto).any():raise ValueError('Source-ineligible pixels in proposal')
                    previous=None;selected=None
                    for p in POLICIES:
                        w=z[p]
                        if w.dtype!=np.float32 or w.shape!=proposal.shape or not np.isfinite(w).all()or(w<0).any()or(w>1).any():
                            raise ValueError('Valid unmodified float32 confidence raster required')
                        if (w[~proposal]!=0).any()or(w[veto]!=0).any():raise ValueError('Outside/veto confidence must be exactly zero')
                        if previous is not None and (w<previous-1e-7).any():raise ValueError('Confidence widths are not monotone')
                        previous=w.copy();sums[p]+=float(w[proposal].sum(dtype=np.float64))
                        if p==policy:selected=w.copy()
                count+=int(proposal.sum());out.append(dict(row,weights=selected,proposal=proposal,veto=veto))
            if count<1:raise ValueError('Empty common proposal')
            means={p:v/count for p,v in sums.items()}
            if any(not 0<v<=1 for v in means.values()):raise ValueError('Zero/invalid global policy mean')
            declared=self.manifest.get('normalizer_policy_'+split)
            if declared is not None and any(not np.isclose(means[p],declared[p],rtol=1e-7,atol=1e-9)for p in POLICIES):
                raise ValueError('Declared mean is not over the common global proposal')
            self.rows[split]=out;self.normalizers[split]=means

    def for_dataset(self,filenames,split):
        by={Path(r['rgb']).resolve():r for r in self.rows[split]}
        if len(by)!=len(filenames):raise ValueError('All split images must have a unique confidence row')
        return [by[Path(p).resolve()]for p in filenames]

def random_background_opaque_variance(alpha):
    """Minimum expected RGB3 random-BG error after unconstrained mean-color fit.

Uniform[0,1] RGB contributes (1-alpha)^2/12, hence favors opacity1 at every
positive confidence. This is not an alpha GT and must not be hidden by a taper.
"""
    return (1-alpha)**2/12

def actor_crop_rectangle(mask,margin=256,alignment=32):
    """Crop native coordinates, retaining the full mask and LPIPS lattice origin."""
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype!=np.bool_ or not mask.any():raise ValueError('Nonempty native boolean actor mask required')
    y,x=np.where(mask);h,w=mask.shape
    left=max(0,(int(x.min())-margin)//alignment*alignment)
    top=max(0,(int(y.min())-margin)//alignment*alignment)
    right=min(w,((int(x.max())+1+margin+alignment-1)//alignment)*alignment)
    bottom=min(h,((int(y.max())+1+margin+alignment-1)//alignment)*alignment)
    return left,top,right,bottom

class ActorNativeEvaluator:
    """Render only frozen semantic native pixels; excluded RGB never contributes.

    The zero-filled remainder of the metric crop is computational storage only.
    Full-footprint metric support excludes it. The stored RGB target is unchanged;
    no semantic opacity, background RGB replacement, or HD upsampling is used.
    """
    def __init__(self,manifest):
        path=Path(manifest);payload=json.loads(path.read_text())
        self.rows=payload.get('native_eval',[])
        if len(self.rows)!=3 or {r['camera']for r in self.rows}!={'DC','HC','KA'}:
            raise ValueError('Three sealed native actor eval rows required')
        for r in self.rows:
            for name in ['rgb','mask']:
                if file_sha(r[name])!=r[name+'_sha256']:raise ValueError('Native actor eval source/hash mismatch')
            c=r['camera_row']
            if (c['w'],c['h'])!=(8192,4320) or any(float(c.get(k,0))!=0 for k in ['k1','k2','p1','p2']):
                raise ValueError('Genuine current-calibration undistorted native eval required')
        self.manifest_sha256=file_sha(path)

    def render(self,model,index,*,chunk=512):
        import torch
        from PIL import Image
        from nerfstudio.cameras.cameras import Cameras,CameraType
        from marshawn_actor_metrics import actor_image_metrics
        row=self.rows[index];c=row['camera_row']
        if Path(row['mask']).suffix.lower()=='.png':
            with Image.open(row['mask'])as image:a=np.asarray(image).copy()
            if a.ndim!=2 or not np.isin(a,[0,255]).all():raise ValueError('Native semantic PNG must be exact binary 0/255')
            mask=a==255
        elif Path(row['mask']).suffix=='.npy':mask=np.load(row['mask'],allow_pickle=False)
        else:
            with np.load(row['mask'],allow_pickle=False)as z:mask=z['actor_mask'].copy()
        if mask.dtype!=np.bool_ or mask.shape!=(c['h'],c['w']):raise ValueError('Exact boolean native semantic mask required')
        x0,y0,x1,y1=actor_crop_rectangle(mask);local_mask=mask[y0:y1,x0:x1]
        with Image.open(row['rgb'])as image:target=np.asarray(image.convert('RGB').crop((x0,y0,x1,y1))).copy()
        device=model.device
        camera=Cameras(camera_to_worlds=torch.tensor(c['transform_matrix'],dtype=torch.float32,device=device)[None,:3,:4],
            fx=c['fl_x'],fy=c['fl_y'],cx=c['cx'],cy=c['cy'],width=c['w'],height=c['h'],camera_type=CameraType.PERSPECTIVE).to(device)
        yy,xx=np.where(local_mask);pred=torch.zeros((y1-y0,x1-x0,3),device=device)
        opacity=torch.zeros((y1-y0,x1-x0,1),device=device)
        with torch.inference_mode():
            for start in range(0,len(yy),chunk):
                ys=torch.from_numpy(yy[start:start+chunk]).to(device);xs=torch.from_numpy(xx[start:start+chunk]).to(device)
                coords=torch.stack([ys+y0+.5,xs+x0+.5],-1)
                rays=camera.generate_rays(camera_indices=0,coords=coords)
                outputs=model(rays)
                pred[ys,xs]=outputs['rgb'];opacity[ys,xs]=outputs['accumulation'].reshape(-1,1)
            gt=torch.from_numpy(target).to(device).float()/255
            m=torch.from_numpy(local_mask).to(device)[None,None]
            lpips=model.lpips.net
            result=actor_image_metrics(pred.permute(2,0,1)[None],gt.permute(2,0,1)[None],m,lpips)
            if any(result[k] is None or not math.isfinite(result[k]) for k in ['psnr','ssim','lpips']):
                raise ValueError('All five native actor LPIPS levels and finite actor metrics required')
        details=dict(result,camera=row['camera'],crop_native_xyxy=[x0,y0,x1,y1],rendered_actor_rays=len(yy),
            GT_sha256=row['rgb_sha256'],semantic_domain_sha256=row['mask_sha256'],mask_is_not_alpha_GT=True)
        return details,{'img':torch.cat([gt,pred],dim=1),'actor_prediction':pred,'actor_opacity':opacity}

def __getattr__(name):
    if name not in ['ActorProposalPixelSampler','ActorConfidenceDataManager','ActorConfidenceModel','ActorConfidencePipeline']:
        raise AttributeError(name)
    _install_native_classes()
    return globals()[name]

def _install_native_classes():
    import torch
    from nerfstudio.data.pixel_samplers import PixelSampler
    from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
    from nerfstudio.models.instant_ngp import NGPModel
    from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline
    from nerfstudio.engine.callbacks import TrainingCallback,TrainingCallbackLocation
    from marshawn_actor_confidence import weighted_rgb_mse
    from marshawn_packed_distortion_candidate import PackedDistortionCapture,load_frozen_formula,normalized_spacing

    class ActorProposalPixelSampler(PixelSampler):
        """Cache the frozen mask once; sample pixels globally with replacement."""
        def sample(self,image_batch):
            if isinstance(image_batch['image'],list):raise ValueError('Actor proposal requires fixed equal rasters')
            # CacheDataloader shuffles even when it caches every split image.
            # Canonicalize the entire batch once, never just the mask/indices:
            # native PixelSampler later translates local indices via image_idx.
            if image_batch is not getattr(self,'_actor_source_batch',None):
                ids=image_batch['image_idx'].detach().cpu().long()
                expected_ids=torch.arange(len(self.actor_expected_proposal))
                if ids.ndim!=1 or not torch.equal(ids.sort().values,expected_ids):
                    raise ValueError('Actor proposal must include ALL unique split cameras')
                if torch.equal(ids,expected_ids):canonical=image_batch
                else:
                    order=ids.argsort();canonical={}
                    for key,value in image_batch.items():
                        if torch.is_tensor(value) and value.ndim and value.shape[0]==len(ids):
                            canonical[key]=value.index_select(0,order.to(value.device))
                        else:canonical[key]=value
                self._actor_source_batch=image_batch
                self._actor_canonical_batch=canonical
            return super().sample(self._actor_canonical_batch)
        def sample_method(self,batch_size,num_images,image_height,image_width,mask=None,device='cpu'):
            if mask is None or self.config.ignore_mask:raise ValueError('Explicit common proposal mask required')
            if not hasattr(self,'_actor_flat'):
                expected=self.actor_expected_proposal.to(mask.device)
                if not torch.equal(mask.reshape_as(expected).bool(),expected):raise ValueError('Dataparser mask is not the common proposal')
                self._actor_flat=torch.nonzero(expected.reshape(-1),as_tuple=False).reshape(-1)
                self._actor_shape=(num_images,image_height,image_width)
            if self._actor_shape!=(num_images,image_height,image_width):raise ValueError('Frozen proposal shape changed')
            return draw_uniform_proposal(self._actor_flat,batch_size,image_height,image_width)

    class ActorConfidenceDataManager(VanillaDataManager):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.config.train_num_images_to_sample_from!=float('inf')or self.config.cpu_fas_prefetch:
                raise ValueError('Global all-camera proposal/no alternate prefetch sampler required')
            table=ActorConfidenceStore(self.config.actor_manifest,self.config.actor_policy)
            object.__setattr__(self,'actor_table',table)
            for split in ['train','eval']:
                ds=getattr(self,split+'_dataset');rows=table.for_dataset(ds.image_filenames,split)
                values=torch.from_numpy(np.stack([r['weights']for r in rows]));proposal=torch.from_numpy(np.stack([r['proposal']for r in rows]))
                object.__setattr__(self,'_actor_weights_'+split,values)
                sampler=getattr(self,split+'_pixel_sampler')
                if sampler is not None:
                    if type(sampler) is not ActorProposalPixelSampler:raise ValueError('Actor uniform-proposal sampler required')
                    sampler.actor_expected_proposal=proposal
        def _attach(self,result,split):
            rays,batch=result;index=batch['indices'].detach().cpu().long();values=getattr(self,'_actor_weights_'+split)
            batch['actor_confidence']=values[index[:,0],index[:,1],index[:,2]].to(batch['image'].device)
            batch['actor_proposal_mean_weight']=self.actor_table.normalizers[split][self.config.actor_policy]
            return rays,batch
        def next_train(self,step):return self._attach(super().next_train(step),'train')
        def next_eval(self,step):return self._attach(super().next_eval(step),'eval')

    class ActorConfidenceModel(NGPModel):
        """Native neural parameters; confidence changes only reconstruction loss."""
        def get_loss_dict(self,outputs,batch,metrics_dict=None):
            if self.renderer_rgb.background_color!='black':raise ValueError('Explicit reviewed fixedblack baseline required')
            target=batch['image'].to(self.device)
            if target.shape[-1]!=3:raise ValueError('Unmodified RGB3; confidence is not alpha GT')
            return {'rgb_loss':weighted_rgb_mse(outputs['rgb'],target,batch['actor_confidence'],
                proposal_mean_weight=batch['actor_proposal_mean_weight'])}
        def get_metrics_dict(self,outputs,batch):
            mse=weighted_rgb_mse(outputs['rgb'],batch['image'].to(self.device),batch['actor_confidence'],
                proposal_mean_weight=batch['actor_proposal_mean_weight'])
            return dict(psnr=-10*torch.log10(mse.clamp_min(1e-12)),num_samples_per_batch=outputs['num_samples_per_ray'].sum(),
                actor_effective_batch_weight=batch['actor_confidence'].sum())
        def get_image_metrics_and_images(self,*args,**kwargs):
            raise RuntimeError('Use frozen actor-only native foreground evaluator; never score whole background')

    class ActorConfidencePipeline(DynamicBatchPipeline):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.model.field.appearance_embedding_dim or self.model.renderer_rgb.background_color!='black':
                raise ValueError('NoApp0 and reviewed fixedblack baseline required')
            if type(self.datamanager) is not ActorConfidenceDataManager:raise ValueError('Explicit actor confidence data manager required')
            object.__setattr__(self,'_actor_formula',load_frozen_formula(self.config.actor_distortion_source,self.config.actor_distortion_source_sha256))
            object.__setattr__(self,'_actor_evaluator',ActorNativeEvaluator(self.config.actor_manifest))
            for name in ['_actor_actual_core','_actor_actual_threshold','_actor_last_core_step']:
                object.__setattr__(self,name,None)
            if getattr(self.config,'actor_geometry_manifest',None) is not None:
                from marshawn_actor_geometry import ActorObservedGeometry
                geometry=ActorObservedGeometry(self.config.actor_geometry_manifest,self.config.actor_geometry_variant,
                    self.model.field.aabb.detach().cpu().numpy(),self.config.actor_manifest,
                    Path(self.config.datamanager.dataparser.data)/'transforms.json',self.config.actor_preparation,
                    device=self.device,free_weight=self.config.actor_free_weight,band_weight=self.config.actor_band_weight)
                object.__setattr__(self,'_actor_geometry',geometry)
            else:object.__setattr__(self,'_actor_geometry',None)
        def get_train_loss_dict(self,step):
            # Fetch once, so sampled confidence and actual graph packet share ordering.
            rays,batch=self.datamanager.next_train(step)
            with PackedDistortionCapture(self.model)as capture:
                outputs=self._model(rays);metrics=self.model.get_metrics_dict(outputs,batch);losses=self.model.get_loss_dict(outputs,batch,metrics)
                if capture.calls!=1 or capture.packet is None:raise ValueError('Expected one graph-connected main packet')
                p=capture.packet;s,e=normalized_spacing(p['starts'],p['ends'],p['ray_indices'],p['near'],p['far'],'linear')
                per_ray=self._actor_formula(s,e,p['weights'],p['ray_indices'],p['num_rays'])
                losses['actor_packed_distortion']=float(self.config.actor_distortion_weight)*weighted_scalar_mean(
                    per_ray,batch['actor_confidence'],batch['actor_proposal_mean_weight'])
            if self._actor_geometry is not None:
                geometry_losses,geometry_metrics=self._actor_geometry(self.model.field,int(step))
                losses.update(geometry_losses);metrics.update(geometry_metrics)
            self._update_dynamic_num_rays_per_batch(int(metrics['num_samples_per_batch']));self._update_pixel_samplers()
            metrics['num_rays_per_batch']=torch.tensor(self.datamanager.train_pixel_sampler.num_rays_per_batch)
            return outputs,losses,metrics
        @torch.no_grad()
        def cache_after_native_core(self,step):
            if step%16:raise ValueError('Only fresh native16 core updates')
            grid=self.model.occupancy_grid
            threshold=torch.clamp(grid.occs[grid.occs>=0].mean(),max=.01)
            actual=(grid.occs>threshold).reshape_as(grid.binaries)
            if not torch.equal(actual,grid.binaries):raise ValueError('Native core callback order changed')
            object.__setattr__(self,'_actor_actual_core',grid.binaries.clone())
            object.__setattr__(self,'_actor_actual_threshold',float(threshold))
            object.__setattr__(self,'_actor_last_core_step',int(step))
        def get_training_callbacks(self,training_callback_attributes):
            callbacks=super().get_training_callbacks(training_callback_attributes)
            if len([c for c in callbacks if getattr(c.func,'__name__','')=='update_occupancy_grid'])!=1:
                raise ValueError('One native occupancy update required')
            callbacks.append(TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=16,func=self.cache_after_native_core))
            return callbacks
        def export_grid_for_checkpoint(self,checkpoint,step):
            checkpoint=Path(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)['pipeline']
            def value(suffix):
                arrays=[v for k,v in saved.items()if k.endswith(suffix)]
                if not arrays or not all(torch.equal(arrays[0],v)for v in arrays[1:]):raise ValueError('Native shared grid aliases mismatch')
                return arrays[0]
            active=value('occupancy_grid.binaries');ema=value('occupancy_grid.occs')
            if self._actor_actual_core is None or not torch.equal(active,self._actor_actual_core.cpu()):
                raise ValueError('Checkpoint must match actual cached GPU core; no CPU-mean authority')
            folder=Path(self.config.actor_grid_artifact_dir);folder.mkdir(parents=True,exist_ok=True)
            archive=folder/(checkpoint.stem+'_grid.npz');bounds=value('occupancy_grid.aabbs')
            np.savez_compressed(archive,core=active.numpy(),active=active.numpy(),ema=ema.numpy(),
                world_aabb=bounds[0].reshape(2,3).numpy(),cascade_aabbs=bounds.numpy(),resolution=value('occupancy_grid.resolution').numpy())
            receipt=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),checkpoint_step=int(step),
                core_update_step=self._actor_last_core_step,gpu_native_threshold=self._actor_actual_threshold,
                grid_archive=str(archive),grid_archive_sha256=file_sha(archive),radius=0,
                policy='Unchanged native EMA core; no dilation or semantic carving',
                actor_manifest_sha256=self.datamanager.actor_table.sha256,actor_policy=self.config.datamanager.actor_policy,
                source_sha256=file_sha(__file__),no_extra_nn_state=True)
            (folder/(checkpoint.stem+'_grid.json')).write_text(json.dumps(receipt,indent=2)+'\n')
            return receipt
        def _actor_eval(self,index):
            return self._actor_evaluator.render(self.model,index,chunk=self.model.config.eval_num_rays_per_chunk)
        def get_eval_image_metrics_and_images(self,step):
            previous=self.training;self.eval()
            try:
                details,images=self._actor_eval((int(step)//max(1,int(self.config.actor_eval_interval)))%3)
                return {k:details[k]for k in ['psnr','ssim','lpips']},images
            finally:self.train(previous)
        def get_average_eval_image_metrics(self,step=None,output_path=None,get_std=False):
            from PIL import Image
            previous=self.training;self.eval();rows=[]
            try:
                for i in range(3):
                    details,images=self._actor_eval(i);rows.append(details)
                    if output_path is not None:
                        out=Path(output_path);out.mkdir(parents=True,exist_ok=True)
                        for name,value in images.items():
                            a=(value.detach().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
                            if a.shape[-1]==1:a=a[...,0]
                            Image.fromarray(a).save(out/(details['camera']+'_'+name+'.png'))
                if output_path is not None:
                    (Path(output_path)/'actor_metric_domain.json').write_text(json.dumps({'rows':rows,
                        'domain_manifest_sha256':self._actor_evaluator.manifest_sha256},indent=2)+'\n')
                metrics={k:float(np.mean([r[k]for r in rows]))for k in ['psnr','ssim','lpips']}
                if get_std:metrics.update({k+'_std':float(np.std([r[k]for r in rows]))for k in ['psnr','ssim','lpips']})
                return metrics
            finally:self.train(previous)

    for cls in [ActorProposalPixelSampler,ActorConfidenceDataManager,ActorConfidenceModel,ActorConfidencePipeline]:
        cls.__qualname__=cls.__name__;cls.__module__=__name__;globals()[cls.__name__]=cls
