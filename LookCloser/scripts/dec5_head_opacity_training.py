"""Opt-in soft-matte supervision of a bounded, independently learned head density."""
from pathlib import Path
import json
import numpy as np
import torch
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng
from run_dec5_hair_campaign import atomic_json,sha


def head_ray_intervals(origins,directions,transform):
    unit=(origins.float()[:,None,:]*transform[None,:,:3]).sum(-1)+transform[:,3]
    direction=(directions.float()[:,None,:]*transform[None,:,:3]).sum(-1)
    parallel=direction.abs()<1e-10;safe=torch.where(parallel,torch.ones_like(direction),direction)
    t0=-unit/safe;t1=(1-unit)/safe
    low=torch.where(parallel,torch.full_like(t0,-float('inf')),torch.minimum(t0,t1))
    high=torch.where(parallel,torch.full_like(t0,float('inf')),torch.maximum(t0,t1))
    near=low.max(-1).values.clamp_min(0);far=high.min(-1).values
    valid=((~parallel)|((unit>=0)&(unit<=1))).all(-1)&(far>near)
    return near,far,valid


def head_opacity_nll(thickness,alpha):
    return -alpha*torch.log(-torch.expm1(-thickness)+1e-12)+(1-alpha)*thickness


class HeadOpacityGuidance:
    def __init__(self,trainer,targets,semantics,output,weight=.05):
        self.pipe=trainer.pipeline;self.model=self.pipe.model;self.weight=weight;self.samples=2048
        self.transform=self.model.field.density_geometry_roi_transform
        if self.transform is None or not (getattr(trainer,'dec5_density_only',False) or getattr(trainer,'dec5_joint_head',False)):
            raise ValueError('Head opacity needs an explicit scoped density continuation mode')
        payload=json.loads((self.pipe.datamanager.config.data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in payload['frames'] if f['file_path'] in payload['train_filenames']}
        receipt=json.loads((targets/'complete.json').read_text());dataset_sha=sha(self.pipe.datamanager.config.data/'transforms.json')
        if receipt['dataset_sha256']!=dataset_sha or receipt.get('uses_validation_rgb') is not False or receipt.get('uses_test_rgb') is not False:raise ValueError('Train-only foreground targets required')
        records={r['camera']:r for r in receipt['records']};self.cameras=self.pipe.datamanager.train_dataset.cameras.to(self.pipe.device)
        self.alpha=[];self.fit=[];self.validation=[];self.width=[];rows=[];rng=np.random.default_rng(42)
        for index,path in enumerate(self.pipe.datamanager.train_dataset.image_filenames):
            f=by[Path(path).name];name=f['physical_camera'];target=targets/(name+'.npz')
            if sha(target)!=records[name]['sha256']:raise ValueError('Changed alpha training target')
            data=np.load(target);up=np.zeros((f['w'],f['h']),np.float32);known=np.zeros_like(up)
            x0,y0,x1,y1=data['portrait_crop'].astype(int);up[y0:y1,x0:x1]=data['alpha'].astype(np.float32)/65535;known[y0:y1,x0:x1]=1
            alpha=np.ascontiguousarray(np.rot90(up,-1));known=np.ascontiguousarray(np.rot90(known,-1))>0
            sem=np.load(semantics/(name+'.npz'))['probability'].astype(np.float32)/65535
            import cv2
            roi=(cv2.dilate((sem[...,0]>.1).astype(np.uint8),np.ones((31,31),np.uint8))>0)&(sem[...,1]<.1)&known
            groups=[roi&(alpha>.01)&(alpha<.99),roi&(alpha>=.99),roi&(alpha<=.01)]
            fit=[];val=[]
            for group in groups:
                flat=np.flatnonzero(group);withheld=(flat*1103515245+(index+1)*12345)%20==0
                fit.append(flat[~withheld]);hold=flat[withheld]
                val.append(rng.choice(hold,min(32,len(hold)),replace=False))
            self.alpha.append(alpha.reshape(-1));self.width.append(f['w']);self.fit.append(fit);self.validation.append(val)
            rows.append(dict(camera=name,fit_per_class=[len(v) for v in fit],withheld_monitor_per_class=[len(v) for v in val]))
        if len(rows)!=59 or set(records)!={r['camera'] for r in rows}:raise ValueError('Train59 alpha inventory mismatch')
        atomic_json(output/'head_opacity_guidance.json',dict(records=rows,weight=weight,samples_per_ray=self.samples,rays_per_step=128,
            samples='2048 stratified density points inside oriented head box; all additional queries counted',
            targets_receipt_sha256=sha(targets/'complete.json'),dataset_sha256=dataset_sha,
            alpha='person matte on known hair-adjacent head crops; face/neck excluded by train semantic probability',
            withheld_alpha_labels='5 percent by deterministic pixel hash, excluded from opacity fitting; RGB remains training RGB',
            original_appearance_and_outside_density=('joint RGB-supervised updates; spatial head-density scope is retained' if getattr(trainer,'dec5_joint_head',False) else
                                                     'frozen by separate branch and spatial field gate'),
            uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))

    def rays(self,index,flat):
        width=self.width[index];uv=np.stack((flat//width+.5,flat%width+.5),-1)
        camera=self.cameras[index:index+1]
        rays=camera.generate_rays(0,coords=torch.as_tensor(uv,device=self.pipe.device,dtype=torch.float32),keep_shape=False)
        target=torch.as_tensor(self.alpha[index][flat],device=self.pipe.device)
        near,far,valid=head_ray_intervals(rays.origins,rays.directions,self.transform)
        return rays[valid],near[valid],far[valid],target[valid]

    def thickness(self,rays,near,far,jitter):
        count=len(near)
        if count==0:return near
        offset=torch.rand((count,1),device=self.pipe.device) if jitter else .5
        fractions=(torch.arange(self.samples,device=self.pipe.device)[None]+offset)/self.samples
        distance=near[:,None]+(far-near)[:,None]*fractions
        xyz=rays.origins[:,None]+rays.directions[:,None]*distance[...,None]
        flat=xyz.reshape(-1,3);density=torch.cat([self.model.field.density_fn(part).reshape(-1) for part in flat.split(65536)])
        return density.reshape(count,self.samples).clamp_min(0).mean(-1)*(far-near)

    def install(self):
        original=self.pipe.get_train_loss_dict
        def loss(step):
            outputs,losses,metrics=original(step)
            with fork_seeded_rng(self.pipe.config.training_seed,'head_opacity',step,self.pipe.device):
                index=int(torch.randint(len(self.fit),(1,)).item());selected=[]
                for values,n in zip(self.fit[index],[64,32,32]):
                    if len(values):selected.extend(values[torch.randint(len(values),(n,)).numpy()].tolist())
                if not selected:return outputs,losses,metrics
                rays,near,far,target=self.rays(index,np.asarray(selected,dtype=np.int64))
                if not len(near):return outputs,losses,metrics
                thickness=self.thickness(rays,near,far,True)
            losses['head_opacity_loss']=self.weight*head_opacity_nll(thickness,target).mean()
            self.pipe.cumulative_point_samples.add_(len(near)*self.samples)
            metrics['cumulative_point_samples']=self.pipe.cumulative_point_samples.clone()
            metrics['head_opacity_rays']=torch.tensor(len(near),device=self.pipe.device)
            return outputs,losses,metrics
        self.pipe.get_train_loss_dict=loss

    def evaluate(self):
        self.pipe.eval();errors=[]
        try:
            with torch.no_grad():
                for index,groups in enumerate(self.validation):
                    flat=np.concatenate(groups)
                    if not len(flat):continue
                    rays,near,far,target=self.rays(index,flat)
                    if not len(near):continue
                    alpha=-torch.expm1(-self.thickness(rays,near,far,False))
                    errors.append((alpha-target).abs().cpu().numpy())
            values=np.concatenate(errors)
            return dict(eval_head_opacity_mae=float(values.mean()),eval_head_opacity_p90=float(np.quantile(values,.9)),eval_head_opacity_monitor_rays=len(values))
        finally:self.pipe.train()
