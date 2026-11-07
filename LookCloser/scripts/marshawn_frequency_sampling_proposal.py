"""CPU-only prototype for an actor-focused frequency side-channel; never installed in training.

The original pipeline still draws camera IDs/y/x. prepare() shadows those draws on
an isolated CPU Torch generator, so hook(indices) can retain the exact original
uniform 20% despite the existing hook exposing only camera IDs. The separate
NumPy generator owns all quota/actor-pool draws and is resumable.
"""
import copy
import numpy as np
import torch

class FrequencyProposal:
    def __init__(self, shapes, pools, seed=4201024):
        self.shapes=np.asarray(shapes,dtype=np.int64)
        self.pools=pools
        if self.shapes.shape!=(len(pools),2) or (self.shapes<=0).any():raise ValueError('Invalid camera shapes')
        self.rng=np.random.default_rng(seed);self.updates=0;self.pending=None
        for i,row in enumerate(pools):
            for name in ('core','band','valid'):
                x=np.asarray(row[name]);limit=int(np.prod(self.shapes[i]))
                if x.ndim!=1 or not np.issubdtype(x.dtype,np.integer) or (x<0).any() or (x>=limit).any():raise ValueError('Invalid pool')
            if not len(row['valid']):raise ValueError('No valid fallback')
    def prepare(self, available, count, global_cpu_state):
        if self.pending is not None:raise ValueError('Previous frequency hook not consumed')
        available=torch.as_tensor(available,dtype=torch.long)
        if count<=0 or len(available)!=len(self.pools) or sorted(available.tolist())!=list(range(len(self.pools))):raise ValueError('Exact global TRAIN camera IDs required')
        generator=torch.Generator(device='cpu');generator.set_state(global_cpu_state)
        ids=available[torch.randint(len(available),(count,),generator=generator)]
        shapes=torch.from_numpy(self.shapes)[ids]
        y=(torch.rand(count,generator=generator)*shapes[:,0].float()).long()
        x=(torch.rand(count,generator=generator)*shapes[:,1].float()).long()
        self.pending=(ids,y,x,generator.get_state())
        return ids.clone()
    def hook(self,indices):
        if self.pending is None:raise ValueError('No shadowed historical draws')
        ids,y,x,expected_rng=self.pending
        if not torch.equal(indices.cpu().reshape(-1),ids):raise ValueError('Camera order differs from historical draws')
        if not torch.equal(torch.get_rng_state(),expected_rng):raise ValueError('Historical CPU RNG trajectory changed before hook')
        n=len(ids);order=self.rng.permutation(n);nc=round(.6*n);nb=round(.2*n)
        labels=np.full(n,2,np.int8);labels[order[:nc]]=0;labels[order[nc:nc+nb]]=1
        y=y.clone();x=x.clone();fallbacks=0
        for camera in np.unique(ids.numpy()):
            for label,name in [(0,'core'),(1,'band')]:
                chosen=np.flatnonzero((ids.numpy()==camera)&(labels==label));pool=self.pools[camera][name]
                if not len(pool):
                    fallbacks+=len(chosen);pool=self.pools[camera]['core']
                    if not len(pool):pool=self.pools[camera]['valid']
                flat=np.asarray(pool)[self.rng.integers(len(pool),size=len(chosen))]
                y[chosen]=torch.from_numpy(flat//self.shapes[camera,1]);x[chosen]=torch.from_numpy(flat%self.shapes[camera,1])
        self.pending=None;self.updates+=1
        self.last=dict(labels=labels,quotas=[int((labels==i).sum())for i in range(3)],fallbacks=fallbacks)
        # Legacy projection ignores support_valid: no new opacity/validity policy.
        return y,x,torch.ones(n,dtype=torch.bool)
    def state_dict(self):
        if self.pending is not None:raise ValueError('Cannot checkpoint during frequency update')
        return dict(schema=1,updates=self.updates,rng=copy.deepcopy(self.rng.bit_generator.state))
    def load_state_dict(self,state):
        if state.get('schema')!=1 or self.pending is not None:raise ValueError('Invalid private state')
        self.rng.bit_generator.state=copy.deepcopy(state['rng']);self.updates=int(state['updates'])
