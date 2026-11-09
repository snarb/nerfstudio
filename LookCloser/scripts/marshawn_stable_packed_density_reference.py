"""FP64 segmented exclusive optical-depth reference; no cross-ray subtraction.

Diagnostic reference only. Never substitutes field outputs or native weights.
"""
import torch


def stable_packed_density_weights(starts,ends,sigma,ids,ray_count):
    n=len(ids)
    if any(value.shape!=(n,)for value in [starts,ends,sigma])or ids.dtype not in [torch.int32,torch.int64]:
        raise ValueError('Matching flat intervals/density and integer ray IDs required')
    if (ids<0).any()or(ids>=ray_count).any()or(ids[1:]<ids[:-1]).any():
        raise ValueError('Packed IDs must be sorted and in range')
    if any(not torch.isfinite(value).all()for value in [starts,ends,sigma])or(ends<starts).any()or(sigma<0).any():
        raise ValueError('Finite nonnegative optical depth required')
    tau=sigma.detach().double()*(ends.detach().double()-starts.detach().double())
    if not n:return tau
    counts=torch.bincount(ids.long(),minlength=ray_count)
    # Integer offsets are exact; no floating global prefix or subtraction of tau.
    offsets=torch.cumsum(counts,0)-counts
    local=torch.arange(n,device=ids.device)-offsets[ids.long()]
    padded=tau.new_zeros((ray_count,int(counts.max())))
    padded[ids.long(),local]=tau
    exclusive=torch.zeros_like(padded)
    exclusive[:,1:]=torch.cumsum(padded[:,:-1],dim=1)
    before=exclusive[ids.long(),local]
    return torch.exp(-before)*-torch.expm1(-tau)
