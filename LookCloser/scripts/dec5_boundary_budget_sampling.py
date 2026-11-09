"""Spend part of whole-image quota on hair contour, preserving body/FAS draws."""
from pathlib import Path
import cv2,numpy as np,torch
from dec5_leader_person_sampling import LeaderPersonSampler
from nerfstudio.utils.lookcloser_rng import stream_seed


class BoundaryBudgetSampler(LeaderPersonSampler):
    def __init__(self,*args,boundary_fraction=.1,boundary_frequency=False,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<boundary_fraction<=1-self.fraction+1e-9:raise ValueError('Boundary budget must come from whole-image quota')
        self.boundary_fraction=boundary_fraction;self.boundary_frequency=boundary_frequency
        self.boundary_pools=[];self.boundary_weights=[];self.boundary_draws=0
        dataset=args[1];semantics=Path(args[2])
        for index,path in enumerate(dataset.image_filenames):
            name=Path(path).stem[len('frame_train_'):]
            hair=np.load(semantics/(name+'.npz'))['probability'][...,0]>.5*65535
            kernel=np.ones((11,11),np.uint8)
            band=cv2.dilate(hair.astype(np.uint8),kernel)!=cv2.erode(hair.astype(np.uint8),kernel)
            pool=np.flatnonzero(band);self.boundary_pools.append(pool)
            frequency=torch.load(Path(path).parent.parent/'lookcloser_frequencies'/(Path(path).stem+'.pt'),map_location='cpu',weights_only=True).numpy()
            # Same level1..3 weighting, evaluated at semantic boundary pixels.
            frequency=cv2.resize(frequency,(hair.shape[1],hair.shape[0]),interpolation=cv2.INTER_NEAREST)
            levels=np.clip(np.log(np.maximum(frequency,16)/16)/np.log(8192/16),0,1)
            weight=(1+2*levels).ravel()[pool].astype(np.float64)
            self.boundary_weights.append(np.cumsum(weight/weight.sum()) if len(weight) else weight)

    def sample(self,image_batch,**kwargs):
        batch=super().sample(image_batch,**kwargs)
        rng=np.random.RandomState(stream_seed(424242,'pixel',self.step))
        indices=batch['indices'].numpy();classes=batch['subject_sampling_class'].numpy()
        selected=(self.last_source==2)&(rng.random(len(indices))<self.boundary_fraction/(1-self.fraction))
        old=np.bincount(classes,minlength=4)
        for c in np.unique(indices[selected,0]):
            take=np.flatnonzero(selected&(indices[:,0]==c));pool=self.boundary_pools[c]
            if not len(pool):continue
            if self.boundary_frequency:pick=pool[np.searchsorted(self.boundary_weights[c],rng.random(len(take))).clip(0,len(pool)-1)]
            else:pick=pool[rng.randint(len(pool),size=len(take))]
            h,w,sh,sw=self.shapes[c]
            indices[take,1]=((pick//sw+rng.random(len(take)))*h/sh).astype(np.int64)
            indices[take,2]=((pick%sw+rng.random(len(take)))*w/sw).astype(np.int64)
            yy=np.minimum(((indices[take,1]+.5)*sh/h).astype(int),sh-1)
            xx=np.minimum(((indices[take,2]+.5)*sw/w).astype(int),sw-1)
            classes[take]=self.labels[c][yy,xx];self.boundary_draws+=len(take)
        available=image_batch['image_idx'].tolist();lookup={int(c):i for i,c in enumerate(available)}
        local=torch.tensor([lookup[int(c)] for c in indices[:,0]])
        batch['image']=image_batch['image'][local,batch['indices'][:,1],batch['indices'][:,2]]
        new=np.bincount(classes,minlength=4)
        for key,change in zip(['hair','person_nonhair','background','uncertain'],new-old):self.audit[key]+=int(change)
        self.last_batch=new.tolist()
        return batch

    def save(self,path):
        from run_dec5_sequence import read,write
        super().save(path);record=read(path)
        record['policy'].update(boundary_fraction=self.boundary_fraction,boundary_frequency=self.boundary_frequency,
            boundary_origin='whole-image quota only; original body and body-FAS indices preserved',boundary_radius_semantic_pixels=5)
        record['boundary_draws']=self.boundary_draws;write(path,record)
