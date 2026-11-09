"""Explicit frozen sampling prior; no renderer, selector or occupancy mutation."""
from pathlib import Path
import hashlib
import numpy as np
import torch


class FrozenBelowFloorPrior:
    """80% volume-weighted frozen active cells,20% whole below-floor band.

    This mixture defines a focused scene prior, not a uniform-volume integral.
    Both treatment and lambda0 control draw/query exactly the same1024points.
    All random draws use a separately checkpointed CPU generator.
    """
    def __init__(self, table, *, seed=42, count=1024, focused_probability=.8, dt=.0015625):
        table=Path(table);self.table_sha256=hashlib.sha256(table.read_bytes()).hexdigest()
        with np.load(table,allow_pickle=False)as z:
            self.lo=torch.from_numpy(z['cell_lo'].copy()).double()
            self.hi=torch.from_numpy(z['cell_hi'].copy()).double()
            self.band=torch.from_numpy(z['band'].copy()).double()
            volume=np.prod(z['cell_hi']-z['cell_lo'],axis=1,dtype=np.float64)
            self.source_grid_sha256=str(z['original_grid_sha256'])
        if self.source_grid_sha256!='7db15fdda89f0661e767e53dd97ab34f3346550aa319d4363bdd6332b3cbac0f':
            raise ValueError('Exact original native source active grid required')
        if self.band.shape!=(2,3) or len(volume)==0 or np.any(volume<=0):raise ValueError('Positive clipped source cell volumes required')
        self.cdf=torch.from_numpy(np.cumsum(volume,dtype=np.float64)/volume.sum());self.cdf[-1]=1.
        self.seed=seed;self.count=count;self.focused_probability=focused_probability;self.dt=dt
        if (seed,count,focused_probability,dt)!=(42,1024,.8,.0015625):raise ValueError('Exact declared prior configuration required')
        self.generator=torch.Generator(device='cpu');self.generator.manual_seed(seed)
        self.draws=0;self.last_packet=None

    def draw(self, device):
        # Fixed five scalar uniforms per point, including unused cell uniforms
        # for whole-band samples; no rejection or data-dependent RNG advance.
        u=torch.rand((self.count,5),generator=self.generator,dtype=torch.float64,device='cpu')
        focused=u[:,0]<self.focused_probability
        cell=torch.searchsorted(self.cdf,u[:,1].contiguous(),right=True)
        lo=torch.where(focused[:,None],self.lo[cell],self.band[0])
        hi=torch.where(focused[:,None],self.hi[cell],self.band[1])
        points=(lo+(hi-lo)*u[:,2:]).float()
        # Preserve the strict z<-.05 domain after float32 endpoint rounding.
        upper=self.band[1,2].float()
        points[:,2].clamp_(max=torch.nextafter(upper,torch.tensor(float('-inf'))))
        if not torch.all(points[:,2]<upper):raise ValueError('Strict below-plane domain violated')
        self.draws+=1
        self.last_packet=dict(draw=self.draws,points_sha256=hashlib.sha256(points.numpy().tobytes()).hexdigest(),
            focused_count=int(focused.sum()),whole_band_count=int((~focused).sum()))
        return points.to(device=device),dict(self.last_packet)

    def objective(self, sigma, coefficient):
        if coefficient not in [0.,1e-4,1e-3]:raise ValueError('Declared zero or prior weights only')
        sigma=sigma.float()
        if sigma.numel()!=self.count or not torch.isfinite(sigma).all() or (sigma<0).any():
            raise FloatingPointError('Finite nonnegative native sigma required even for lambda0')
        auxiliary=torch.log1p(sigma*self.dt).mean()
        if not torch.isfinite(auxiliary):raise FloatingPointError('Finite native log1p density objective required')
        # Keep the lambda0 field query and autograd path for the matched control.
        return coefficient*auxiliary,auxiliary

    def state_dict(self):
        return dict(table_sha256=self.table_sha256,source_grid_sha256=self.source_grid_sha256,seed=self.seed,count=self.count,
            focused_probability=self.focused_probability,dt=self.dt,draws=self.draws,
            generator_state=self.generator.get_state().clone(),last_packet=self.last_packet)

    def load_state_dict(self,state):
        expected=self.state_dict()
        for key in ['table_sha256','source_grid_sha256','seed','count','focused_probability','dt']:
            if state[key]!=expected[key]:raise ValueError('Private prior checkpoint contract differs:'+key)
        self.generator.set_state(state['generator_state'].cpu());self.draws=int(state['draws']);self.last_packet=state['last_packet']
