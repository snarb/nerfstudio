"""Luster body80/FAS50 recipe adapted to calibrated native DEC5 RGB.

80% is sampling support (core plus protective outer band); the remaining20%
samples the whole image. It is not a hard 80/20 person/background pixel split.
Masks never edit RGB. Native frequency cells retain their original camera IDs.
"""
from pathlib import Path
import cv2,numpy as np,torch
from scipy.ndimage import binary_fill_holes
from dec5_subject_sampling import SubjectSampler
from nerfstudio.utils.lookcloser_rng import stream_seed
from run_dec5_hair_campaign import atomic_json,sha


class LeaderPersonSampler(SubjectSampler):
    def __init__(self,sampler,dataset,semantics,fraction=.8,fas_strength=.5,hair_core_mix=0.,hair_mix_start=0):
        if len(dataset.image_filenames)!=64:raise ValueError('Declared 64-camera train split required')
        if not 0<=hair_core_mix<=1:raise ValueError('Invalid hair-core mixture')
        if hair_mix_start<0:raise ValueError('Negative hair-mixture switch')
        self.sampler=sampler;self.fraction=fraction;self.fas_strength=fas_strength;self.step=0
        self.hair_core_mix=hair_core_mix;self.core_hair=[];self.hair_cells=[];self.focused_hits=0
        self.hair_mix_start=hair_mix_start
        self.shapes=[];self.core=[];self.band=[];self.cells=[];self.reliable=[];self.labels=[];self.records=[]
        self.audit=dict(batches=0,hair=0,person_nonhair=0,background=0,uncertain=0,rays=0)
        self.source_hits=np.zeros(4,np.int64);self.camera_hits=np.zeros(64,np.int64)
        self.audit_start_sample_count=int(getattr(sampler,'sample_count',0))
        kernel=lambda r:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
        for index,path in enumerate(dataset.image_filenames):
            name=Path(path).stem[len('frame_train_'):];source=semantics/(name+'.npz')
            value=np.load(source)['probability'].astype(np.float32)/65535
            original=(value[...,2]<.5)|(value[...,0]>.05)
            core=binary_fill_holes(cv2.morphologyEx(original.astype(np.uint8),cv2.MORPH_CLOSE,kernel(8))|original)
            # DEC5 is stored sideways; the leader's wider head/upper-body band
            # is defined in portrait space and then mapped back to source pixels.
            portrait=np.ascontiguousarray(np.rot90(core));y,x=np.nonzero(portrait)
            outer=cv2.dilate(portrait.astype(np.uint8),kernel(16)).astype(bool)
            if len(y):
                wide=cv2.dilate(portrait.astype(np.uint8),kernel(32)).astype(bool)
                cut=int(y.min()+.75*(y.max()-y.min()+1));outer[:cut]|=wide[:cut]
            band=np.ascontiguousarray(np.rot90(outer&~portrait,-1))
            height=int(dataset.cameras.height[index]);width=int(dataset.cameras.width[index])
            self.shapes.append((height,width,*core.shape))
            self.core.append(np.flatnonzero(core));self.band.append(np.flatnonzero(band))
            self.core_hair.append(np.flatnonzero(value[...,0]>.05))
            path=Path(path);frequency_path=path.parent.parent/'lookcloser_frequencies'/(path.stem+'.pt')
            frequency=torch.load(frequency_path,map_location='cpu',weights_only=True).numpy()
            levels=np.rint(np.log(np.maximum(frequency,16)/16)/np.log((8192/16)**(1/15))).clip(0,15).astype(np.uint8)
            coverage=cv2.resize(original.astype(np.float32),(frequency.shape[1],frequency.shape[0]),interpolation=cv2.INTER_AREA)
            reliable=coverage>=.9
            self.cells.append([np.flatnonzero(reliable&(levels==level)) for level in range(16)])
            hair_coverage=cv2.resize((value[...,0]>.5).astype(np.float32),(frequency.shape[1],frequency.shape[0]),interpolation=cv2.INTER_AREA)
            self.hair_cells.append([np.flatnonzero((hair_coverage>=.9)&(levels==level)) for level in range(16)])
            self.reliable.append((np.flatnonzero(reliable),frequency.shape[1]))
            self.labels.append(np.where(value[...,0]>.5,0,np.where(value[...,2]<.2,1,np.where(value[...,2]>.8,2,3))).astype(np.uint8))
            self.records.append(dict(camera=name,semantic_sha256=sha(source),frequency_sha256=sha(frequency_path),
                native_shape=[height,width],semantic_shape=list(core.shape),core_pixels=int(core.sum()),
                band_pixels=int(band.sum()),reliable_frequency_cells=int(reliable.sum())))

    def sample(self,image_batch,commit_sample_count=True,**kwargs):
        images=image_batch['image']
        if not isinstance(images,torch.Tensor) or images.device.type!='cpu' or set(image_batch)-{'image','image_idx'}:
            raise ValueError('Leader support sampler requires homogeneous CPU RGB without masks/auxiliary targets')
        rng=np.random.RandomState(stream_seed(42,'pixel',self.step))
        n=self.sampler.num_rays_per_batch;available=image_batch['image_idx'].detach().cpu().numpy()
        camera=available[rng.randint(len(available),size=n)]
        shapes=np.asarray(self.shapes)[camera];height,width=shapes[:,:2].T
        y=(rng.random(n)*height).astype(np.int64);x=(rng.random(n)*width).astype(np.int64)
        u=rng.random(n);source=np.where(u<self.fraction*5/6,0,np.where(u<self.fraction,1,2))
        strength=self.fas_strength*min(1.,max(0.,(self.step-4096)/4096))
        use_fas=(source==0)&(rng.random(n)<strength)
        active_mix=self.hair_core_mix if self.step>=self.hair_mix_start else 0.
        focused=(source==0)&(rng.random(n)<active_mix) if active_mix else np.zeros(n,bool)
        for c in np.unique(camera):
            h,w,sh,sw=self.shapes[c]
            for kind,pool in [(0,self.core[c]),(1,self.band[c])]:
                selected=np.flatnonzero((camera==c)&(source==kind))
                if not len(pool):
                    pool=self.core[c]
                    if not len(pool):source[selected]=2;continue
                pick=pool[rng.randint(len(pool),size=len(selected))]
                y[selected]=((pick//sw+rng.random(len(selected)))*h/sh).astype(np.int64)
                x[selected]=((pick%sw+rng.random(len(selected)))*w/sw).astype(np.int64)
            selected=np.flatnonzero((camera==c)&focused)
            if len(selected) and len(self.core_hair[c]):
                pool=self.core_hair[c];pick=pool[rng.randint(len(pool),size=len(selected))]
                y[selected]=((pick//sw+rng.random(len(selected)))*h/sh).astype(np.int64)
                x[selected]=((pick%sw+rng.random(len(selected)))*w/sw).astype(np.int64)
            selected=np.flatnonzero((camera==c)&use_fas)
            groups=[(selected,self.cells[c])] if not active_mix else [
                (selected[~focused[selected]],self.cells[c]),(selected[focused[selected]],self.hair_cells[c])]
            for chosen,cells in groups:
                weights=np.asarray([len(pool)>0 for pool in cells])*np.linspace(1,3,16)
                if not weights.sum():cells=self.cells[c];weights=np.asarray([len(pool)>0 for pool in cells])*np.linspace(1,3,16)
                if len(chosen) and weights.sum():
                    levels=np.searchsorted(np.cumsum(weights/weights.sum()),rng.random(len(chosen))).clip(0,15)
                    for level in np.unique(levels):
                        take=chosen[levels==level];pool=cells[level];cell=pool[rng.randint(len(pool),size=len(take))]
                        columns=self.reliable[c][1];y[take]=(cell//columns)*8+rng.randint(8,size=len(take));x[take]=(cell%columns)*8+rng.randint(8,size=len(take))
                    source[chosen]=3
        self.last_source=source.copy()
        indices=np.stack((camera,y,x),-1);classes=np.empty(n,np.uint8)
        for c in np.unique(camera):
            take=camera==c;h,w,sh,sw=self.shapes[c]
            yy=np.minimum(((y[take]+.5)*sh/h).astype(int),sh-1);xx=np.minimum(((x[take]+.5)*sw/w).astype(int),sw-1)
            classes[take]=self.labels[c][yy,xx]
        lookup={int(c):i for i,c in enumerate(available)};local=torch.tensor([lookup[int(c)] for c in camera])
        counts=np.bincount(classes,minlength=4)
        for key,value in zip(['hair','person_nonhair','background','uncertain'],counts.tolist()):self.audit[key]+=value
        self.audit['batches']+=1;self.audit['rays']+=n;self.last_batch=counts.tolist()
        self.source_hits+=np.bincount(source,minlength=4);self.camera_hits+=np.bincount(camera,minlength=64)
        self.focused_hits+=int(focused.sum())
        if commit_sample_count:self.sampler.sample_count+=1
        return dict(indices=torch.from_numpy(indices),image=images[local,torch.from_numpy(y),torch.from_numpy(x)],
                    subject_sampling_class=torch.from_numpy(classes))

    def frequency_pixels(self,camera_indices):
        rng=np.random.RandomState(stream_seed(42,'frequency_grid',self.step))
        cameras=np.asarray(camera_indices);y=np.zeros(len(cameras),np.int64);x=y.copy();valid=np.zeros(len(cameras),bool)
        for c in np.unique(cameras):
            take=cameras==c;pool,columns=self.reliable[c]
            if len(pool):
                cells=pool[rng.randint(len(pool),size=int(take.sum()))]
                y[take]=(cells//columns)*8+rng.randint(8,size=int(take.sum()))
                x[take]=(cells%columns)*8+rng.randint(8,size=int(take.sum()));valid[take]=True
        return torch.from_numpy(y),torch.from_numpy(x),torch.from_numpy(valid)

    def save(self,path):
        atomic_json(path,dict(policy=dict(person_support_fraction=self.fraction,whole_image_fraction=1-self.fraction,
            experimental_hair_core_mix=self.hair_core_mix,
            hair_mix_start_step=self.hair_mix_start,
            protective_band_share_of_support=1/6,body_fas_strength=self.fas_strength,fas_warmup=4096,fas_ramp=4096,
            frequency_class_weights='1 to3 over nonempty levels; uniform within each selected level',
            rgb_masked=False),audit=self.audit,source_hits=self.source_hits.tolist(),
            hair_focused_draws=self.focused_hits,
            source_names=['core_uniform','protective_band','whole_image','body_FAS'],camera_hits=self.camera_hits.tolist(),
            field_queries_by_class=self.query_counts.cpu().tolist() if getattr(self,'query_counts',None) is not None else None,
            records=self.records,uses_validation_rgb=False,implementation_sha256=sha(__file__)))
