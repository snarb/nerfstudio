"""Experimental opacity controls and hard first-intersection LookCloser."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Literal, Type
import torch
import torch.nn.functional as F
from nerfstudio.fields.lookcloser_field import LookCloserField
from nerfstudio.fields.lookcloser_surface_field import LookCloserSurfaceField
from nerfstudio.model_components.lookcloser_opacity import first_crossing, sdf_weights, opacity_profile_loss, depth_interval_loss
from nerfstudio.models.lookcloser import LookCloserModel, LookCloserModelConfig


@dataclass
class LookCloserOpacityModelConfig(LookCloserModelConfig):
    _target: Type = field(default_factory=lambda: LookCloserOpacityModel)
    representation: Literal['volume','opaque_surface','semantic_surface'] = 'volume'
    opacity_regularization: float = 0.
    opacity_ramp_steps: int = 4096
    depth_prior_strength: float = 0.
    surface_num_samples: int = 256
    surface_refine_steps: int = 8
    surface_inverse_width: float = 400.
    eikonal_loss_mult: float = .1
    semantic_grid_path: str | None = None
    semantic_transparency_mult: float = .01


class LookCloserOpacityModel(LookCloserModel):
    config: LookCloserOpacityModelConfig

    def populate_modules(self):
        if self.config.representation not in {'volume','opaque_surface','semantic_surface'}:
            raise ValueError('Unknown LookCloser representation')
        controls=[self.config.opacity_regularization,self.config.depth_prior_strength,self.config.eikonal_loss_mult]
        if any(not math.isfinite(value) or value<0 for value in controls):
            raise ValueError('Regularization must be nonnegative')
        if self.config.surface_num_samples < 2 or self.config.surface_refine_steps < 0 or self.config.surface_inverse_width <= 0:
            raise ValueError('Invalid surface integrator configuration')
        if self.config.representation!='volume' and (
            self.config.appearance_embedding_dim!=0 or self.config.background_color not in {'black','white'}
            or self.config.rgb_output_parameterization!='sigmoid' or self.config.reconstruction_loss_type!='charbonnier'):
            raise ValueError('Surface experiments require sigmoid RGB/Charbonnier, black or white background, and no appearance embedding')
        self.field_class = LookCloserField if self.config.representation == 'volume' else LookCloserSurfaceField
        super().populate_modules()
        self.freq_grid.skip_invalid_depth_updates = True
        self.register_buffer('opacity_representation',torch.tensor(
            ['volume','opaque_surface','semantic_surface'].index(self.config.representation),dtype=torch.int64))
        self.register_buffer('semantic_permissions',torch.zeros(1,1,1,dtype=torch.bool))
        if self.config.representation == 'semantic_surface':
            if self.config.semantic_grid_path is None:
                raise ValueError('semantic_surface requires reviewed train-only spatial permissions')
            payload = torch.load(Path(self.config.semantic_grid_path),map_location='cpu',weights_only=True)
            if payload.get('train_only') is not True or payload.get('reviewed') is not True:
                raise ValueError('Semantic permissions must be train-only and reviewed')
            if not torch.allclose(payload['aabb'].float(),self.scene_box.aabb.cpu(),atol=1e-6):
                raise ValueError('Semantic grid AABB differs from the field')
            self.semantic_permissions = payload['permissions'].bool()
            if self.semantic_permissions.ndim!=3 or min(self.semantic_permissions.shape)<1:
                raise ValueError('Semantic permissions must be a nonempty 3D grid')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        key = prefix+'opacity_representation'
        if key in state_dict and int(state_dict[key]) != int(self.opacity_representation):
            error_msgs.append('Checkpoint surface/volume representation is incompatible with config')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def get_training_callbacks(self,attributes):
        if self.config.representation != 'volume':
            return []  # Density occupancy updates are not valid for an SDF.
        return super().get_training_callbacks(attributes)

    def _semantic_allowed(self,positions):
        aabb = self.scene_box.aabb.to(positions)
        uv = (positions-aabb[0])/(aabb[1]-aabb[0])
        shape = torch.tensor(self.semantic_permissions.shape,device=positions.device)
        ijk = (uv*shape).long()
        inside = ((uv>=0)&(uv<1)).all(-1)
        ijk = torch.minimum(ijk.clamp_min(0),shape-1)
        return inside & self.semantic_permissions[ijk[:,0],ijk[:,1],ijk[:,2]]

    def _surface_outputs(self,ray_bundle):
        o,d = ray_bundle.origins,ray_bundle.directions
        n = len(o)
        count = self.config.surface_num_samples
        near,far = ray_bundle.nears,ray_bundle.fars
        edges = torch.linspace(0,1,count+1,device=o.device)
        t = near+(far-near)*edges[None,:]
        t = self._guided_edges(ray_bundle,t)
        count=t.shape[1]-1
        points = o[:,None]+d[:,None]*t[...,None]
        sdf = self.field.query_surface(points.reshape(-1,3)).reshape(n,count+1)
        first,hit = first_crossing(sdf.detach())
        crossing = (sdf.detach()[:,:-1]*sdf.detach()[:,1:]<=0) & (
            (sdf.detach()[:,:-1]!=0)|(sdf.detach()[:,1:]!=0))
        if self.config.representation == 'opaque_surface':
            ray_ids = torch.arange(n,device=o.device)
            brackets = first
            valid = hit
        else:
            # Spatially permitted transparency may reveal subsequent surfaces.
            ray_ids,brackets = torch.nonzero(crossing,as_tuple=True)
            valid = torch.ones_like(ray_ids,dtype=torch.bool)
        with torch.no_grad():
            lo,hi = t[ray_ids,brackets],t[ray_ids,brackets+1]
            s_lo = sdf.detach()[ray_ids,brackets]
            for _ in range(self.config.surface_refine_steps):
                mid = (lo+hi)*.5
                s_mid = self.field.query_surface(o[ray_ids]+d[ray_ids]*mid[:,None]).flatten()
                same = s_mid*s_lo>0
                lo = torch.where(same,mid,lo)
                hi = torch.where(same,hi,mid)
                s_lo = torch.where(same,s_mid,s_lo)
            root = (lo+hi)*.5
            root_points = o[ray_ids]+d[ray_ids]*root[:,None]
        _,color,alpha = self.field.query_surface(root_points,d[ray_ids])
        if self.config.representation == 'opaque_surface':
            hard_rgb = color*valid[:,None]
            hard_acc = valid[:,None].float()
            hard_depth = root[:,None]*valid[:,None]
            transparency = color.sum()*0
        else:
            allowed = self._semantic_allowed(root_points)
            # Nerfacc scans require FP32 even under mixed-precision training.
            # Accumulate colors in the same dtype; casting retains head grads.
            alpha = alpha.float()
            alpha = torch.where(allowed[:,None],alpha,torch.ones_like(alpha))
            # Packed ray IDs are sorted; opaque crossings set all later weights to zero.
            import nerfacc
            packing = nerfacc.pack_info(ray_ids,n)
            w,_ = nerfacc.render_weight_from_alpha(alpha[:,0],packed_info=packing)
            hard_rgb = w.new_zeros(n,3).index_add(0,ray_ids,w[:,None]*color.float())
            hard_acc = w.new_zeros(n,1).index_add(0,ray_ids,w[:,None])
            expected_depth = w.new_zeros(n,1).index_add(0,ray_ids,w[:,None]*root[:,None])/hard_acc.clamp_min(1e-6)
            # Surface geometry/frequency correspondence uses an actual surface,
            # never the mean location between hair and background layers.
            has_crossing=packing[:,1]>0
            hard_depth=w.new_zeros(n,1)
            hard_depth[has_crossing]=root[packing[has_crossing,0],None]
            front_allowed=torch.zeros(n,dtype=torch.bool,device=o.device)
            front_allowed[has_crossing]=allowed[packing[has_crossing,0]]
            # RGB is averaged over the whole ray batch. Use that same domain
            # so a tiny permission region is not overwhelmed by its penalty.
            transparency = ((1-alpha[allowed]).sum()/max(n,1) if allowed.any() else color.sum()*0)
        if self.config.background_color == 'white':
            hard_rgb = hard_rgb+1-hard_acc
        output = dict(rgb=hard_rgb,depth=hard_depth,accumulation=hard_acc,
                      num_samples_per_ray=torch.full((n,1),count+self.config.surface_refine_steps+1,device=o.device),
                      surface_hit=hit[:,None].float(),semantic_transparency=transparency)
        if self.config.representation == 'semantic_surface':
            output['expected_depth']=expected_depth
            output['prior_opaque']=~front_allowed[:,None]
        if not self.training and self.config.representation == 'opaque_surface':
            gradients = self.field.finite_gradients(root_points)
            output['normals'] = F.normalize(gradients,dim=-1)*valid[:,None]
        if self.training or getattr(self,'_surrogate_eval',False):
            mids = (t[:,:-1]+t[:,1:])*.5
            sample_points = o[:,None]+d[:,None]*mids[...,None]
            _,rgb,_ = self.field.query_surface(sample_points.reshape(-1,3),d[:,None].expand(-1,count,-1).reshape(-1,3))
            inverse_width = self.config.surface_inverse_width*min(1.,.1+self.current_train_step/4096.)
            weights = sdf_weights(sdf,inverse_width)
            soft_rgb = (weights[...,None]*rgb.reshape(n,count,3)).sum(1)
            soft_acc = weights.sum(-1,keepdim=True)
            soft_depth = (weights*mids).sum(-1,keepdim=True)/soft_acc.clamp_min(1e-6)
            if self.config.background_color == 'white':
                soft_rgb = soft_rgb+1-soft_acc
            if self.training:
                # Hard forward, surrogate backward, including rays without a hit.
                # Appearance/opacity learn at actual intersections. Only geometry
                # uses the surrogate, so semantic alpha receives RGB gradients.
                geometry_rgb = (weights[...,None]*rgb.detach().reshape(n,count,3)).sum(1)
                if self.config.background_color == 'white':
                    geometry_rgb = geometry_rgb+1-soft_acc
                output['rgb'] = hard_rgb+geometry_rgb-geometry_rgb.detach()
                output['depth'] = hard_depth.detach()+soft_depth-soft_depth.detach()
            output.update(prior_weights=weights.reshape(-1,1),prior_mids=mids.reshape(-1,1),
                          prior_indices=torch.arange(n,device=o.device).repeat_interleave(count),
                          soft_rgb=soft_rgb,soft_accumulation=soft_acc,
                          surrogate_rgb_gap=(hard_rgb-soft_rgb).abs().mean())
        if self.training and self.config.eikonal_loss_mult>0:
            # A failed bracket is not a surface. Semantic back surfaces hidden
            # by a fully opaque first hit should not change this regularizer.
            visible_roots = valid if self.config.representation == 'opaque_surface' else w.detach()>1e-4
            surface_points = root_points.detach()[visible_roots]
            sample = surface_points[::max(1,len(surface_points)//256)]
            aabb = self.scene_box.aabb.to(o)
            uniform = aabb[0]+torch.rand(256,3,device=o.device)*(aabb[1]-aabb[0])
            sample = torch.cat([sample,uniform])
            if len(sample):
                gradients = self.field.finite_gradients(sample)
                output['eikonal'] = (gradients.norm(dim=-1)-1).square().mean()
        metadata=ray_bundle.metadata or {}
        if self.training and self.config.depth_prior_strength>0 and 'depth_prior' in metadata:
            gt=metadata['depth_prior']
            output['prior_sdf']=self.field.query_surface(o+d*gt)
            fractions=torch.tensor([.25,.5,.75,.95],device=o.device)
            probes=o[:,None]+d[:,None]*(gt*fractions[None,:])[...,None]
            output['prior_free_sdf']=self.field.query_surface(probes.reshape(-1,3)).reshape(n,4)
        return output

    def _guided_edges(self,ray_bundle,t):
        metadata=ray_bundle.metadata or {}
        if self.config.depth_prior_strength<=0 or not self.training or 'depth_prior' not in metadata:
            return t
        valid=metadata['depth_valid'].reshape(-1)
        if not valid.any():
            return t
        depth=metadata['depth_prior']
        tolerance=torch.maximum(2*metadata['depth_uncertainty'],.01*depth/self.config.depth_prior_strength)
        offsets=torch.linspace(-2,2,65,device=t.device)
        fine=depth+tolerance*offsets[None,:]
        fine=torch.minimum(torch.maximum(fine,ray_bundle.nears),ray_bundle.fars)
        fine=torch.where(valid[:,None],fine,ray_bundle.nears.expand_as(fine))
        return torch.sort(torch.cat([t,fine],-1),-1).values

    def _guided_volume_outputs(self,ray_bundle):
        """Partition, rather than overlap, intervals around trusted depths.

        The auxiliary samples query the actual learned field. No GT depth/RGB
        is substituted into the render, and eval uses the normal volume sampler.
        """
        o,d=ray_bundle.origins,ray_bundle.directions
        n=len(o)
        near,far=ray_bundle.nears,ray_bundle.fars
        edges=torch.linspace(0,1,self.config.fixed_num_samples_per_ray+1,device=o.device)
        t=self._guided_edges(ray_bundle,near+(far-near)*edges[None,:])
        starts,ends=t[:,:-1],t[:,1:]
        mids=(starts+ends)*.5
        count=mids.shape[1]
        positions=o[:,None]+d[:,None]*mids[...,None]
        sigma,rgb=self.field.query_points(positions.reshape(-1,3),d[:,None].expand(-1,count,-1).reshape(-1,3))
        sigma=sigma.float().reshape(n,count)
        alpha=1-torch.exp(-sigma*(ends-starts))
        trans=torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),1-alpha+1e-10],-1),-1)[:,:-1]
        weights=alpha*trans
        acc=weights.sum(-1,keepdim=True)
        color=(weights[...,None]*rgb.reshape(n,count,3)).sum(1)
        if self.config.background_color=='white':
            color=color+1-acc
        spacing_starts=(starts-near)/(far-near).clamp_min(1e-6)
        spacing_ends=(ends-near)/(far-near).clamp_min(1e-6)
        return dict(rgb=color,accumulation=acc,depth=(weights*mids).sum(-1,keepdim=True)/acc.clamp_min(1e-6),
                    num_samples_per_ray=torch.full((n,),count,device=o.device),
                    packed_weights=weights.reshape(-1,1),packed_spacing_starts=spacing_starts.reshape(-1,1),
                    packed_spacing_ends=spacing_ends.reshape(-1,1),
                    packed_ray_indices=torch.arange(n,device=o.device).repeat_interleave(count))

    def _volume_outputs(self,ray_bundle):
        metadata=ray_bundle.metadata or {}
        if self.config.depth_prior_strength<=0 or not self.training or 'depth_valid' not in metadata or not metadata['depth_valid'].any():
            return super().get_outputs(ray_bundle)
        trusted=metadata['depth_valid'].reshape(-1)
        groups=[]
        for selected,guided in [(trusted,True),(~trusted,False)]:
            original=torch.nonzero(selected,as_tuple=True)[0]
            if not len(original):
                continue
            local=ray_bundle[original]
            result=self._guided_volume_outputs(local) if guided else super().get_outputs(local)
            if 'packed_weights' not in result:
                count=result['loss_weights'].shape[1]
                result.update(packed_weights=result['loss_weights'].reshape(-1,1),
                              packed_spacing_starts=result['loss_ray_samples'].spacing_starts.reshape(-1,1),
                              packed_spacing_ends=result['loss_ray_samples'].spacing_ends.reshape(-1,1),
                              packed_ray_indices=torch.arange(len(original),device=original.device).repeat_interleave(count))
            result['packed_ray_indices']=original[result['packed_ray_indices']]
            groups.append((original,result))
        output={}
        for key in ['rgb','depth','accumulation']:
            value=groups[0][1][key]
            merged=value.new_zeros(len(ray_bundle),value.shape[-1])
            for original,result in groups:
                merged=merged.index_copy(0,original,result[key])
            output[key]=merged
        for key in ['packed_weights','packed_spacing_starts','packed_spacing_ends','packed_ray_indices']:
            output[key]=torch.cat([result[key] for _,result in groups])
        order=torch.argsort(output['packed_ray_indices'],stable=True)
        for key in ['packed_weights','packed_spacing_starts','packed_spacing_ends','packed_ray_indices']:
            output[key]=output[key][order]
        output['num_samples_per_ray']=torch.bincount(output['packed_ray_indices'],minlength=len(ray_bundle))
        return output

    def get_outputs(self,ray_bundle):
        if self.config.representation != 'volume':
            return self._surface_outputs(ray_bundle)
        outputs = self._volume_outputs(ray_bundle)
        if 'packed_weights' in outputs:
            weights,indices = outputs['packed_weights'],outputs['packed_ray_indices']
            starts,ends = outputs['packed_spacing_starts'],outputs['packed_spacing_ends']
        else:
            weights = outputs['loss_weights'].reshape(-1,1)
            starts = outputs['loss_ray_samples'].spacing_starts.reshape(-1,1)
            ends = outputs['loss_ray_samples'].spacing_ends.reshape(-1,1)
            indices = torch.arange(len(ray_bundle),device=weights.device).repeat_interleave(outputs['loss_weights'].shape[1])
        if self.training and self.config.opacity_regularization>0:
            outputs['opacity_profile'] = opacity_profile_loss(weights,starts,ends,indices,len(ray_bundle))
        if self.training and self.config.depth_prior_strength>0:
            near = ray_bundle.nears[indices]
            span = ray_bundle.fars[indices]-near
            outputs.update(prior_weights=weights,prior_mids=near+span*(starts+ends)*.5,prior_indices=indices)
        return outputs

    def get_loss_dict(self,outputs,batch,metrics_dict=None):
        if self.config.representation == 'volume':
            losses = super().get_loss_dict(outputs,batch,metrics_dict)
        else:
            image = batch['image'].to(self.device)
            losses = {'rgb_loss':torch.sqrt((outputs['rgb']-image).square()+1e-4).mean()}
            if 'eikonal' in outputs:
                losses['eikonal_loss'] = self.config.eikonal_loss_mult*outputs['eikonal']
            if self.config.representation == 'semantic_surface':
                losses['semantic_transparency_loss'] = self.config.semantic_transparency_mult*outputs['semantic_transparency']
        if self.config.opacity_regularization>0 and 'opacity_profile' in outputs:
            ramp = min(1.,max(0.,(self.current_train_step-self.config.adaptive_warmup_steps)/max(self.config.opacity_ramp_steps,1)))
            deficit = -torch.log(outputs['accumulation'].clamp_min(1e-6)).mean()
            losses['opacity_loss'] = self.config.opacity_regularization*ramp*(outputs['opacity_profile']+deficit)
        if self.config.depth_prior_strength>0 and 'depth_prior' in batch:
            strength = self.config.depth_prior_strength
            depth = batch['depth_prior'].to(self.device)
            uncertainty = batch['depth_uncertainty'].to(self.device)
            valid = batch['depth_valid'].to(self.device)
            if 'prior_opaque' in outputs:
                valid=valid&outputs['prior_opaque']
            tolerance = torch.maximum(2*uncertainty,.01*depth/strength)
            prior = depth_interval_loss(outputs['prior_weights'],outputs['prior_mids'],outputs['prior_indices'],
                                        len(depth),depth,tolerance,valid)
            if self.config.representation != 'volume' and valid.any():
                distance = ((outputs['depth']-depth).abs()-tolerance).clamp_min(0)/tolerance.clamp_min(1e-6)
                prior = prior+F.smooth_l1_loss(distance[valid],torch.zeros_like(distance[valid]))
                signed=outputs['prior_sdf'][valid]/tolerance[valid].clamp_min(1e-6)
                prior = prior+F.smooth_l1_loss(signed,torch.zeros_like(signed))
                free=(tolerance-outputs['prior_free_sdf']).clamp_min(0)/tolerance.clamp_min(1e-6)
                free=free[valid.reshape(-1)]
                prior = prior+F.smooth_l1_loss(free,torch.zeros_like(free))
            losses['trusted_depth_loss'] = .01*strength*prior
        return losses
