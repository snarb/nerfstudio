"""First detected view-independent density level set, refined in world space."""
import torch


@torch.no_grad()
def first_sigma_surface(density_fn,rays,starts,ends,density,indices,threshold,refine_steps=8,assume_sorted=False):
 if threshold<=0 or not 0<=refine_steps<=16:raise ValueError('Positive density threshold and bounded refinement required')
 starts,ends,density=[x.reshape(-1).float() for x in [starts,ends,density]];indices=indices.reshape(-1).long();near=rays.nears.reshape(-1);far=rays.fars.reshape(-1);n=len(near);result=near.new_zeros(n)
 if not len(indices):return result
 if not(len(starts)==len(ends)==len(density)==len(indices)):raise ValueError('Packed density/interval binding mismatch')
 if not assume_sorted:
  order=torch.argsort(starts,stable=True);order=order[torch.argsort(indices[order],stable=True)];starts,ends,density,indices=[x[order] for x in [starts,ends,density,indices]]
 midpoint=near[indices]+.5*(starts+ends)*(far[indices]-near[indices]);qualifying=torch.nonzero(density>=threshold).reshape(-1);first=torch.full((n,),len(indices),device=indices.device,dtype=torch.long)
 first.scatter_reduce_(0,indices[qualifying],qualifying,reduce='amin',include_self=True);hit=torch.nonzero(first<len(indices)).reshape(-1)
 if not len(hit):return result
 current=first[hit];previous=(current-1).clamp_min(0);has_previous=(current>0)&(indices[previous]==hit);lo=torch.where(has_previous,midpoint[previous],near[hit]);hi=midpoint[current];origins=rays.origins[hit];directions=rays.directions[hit]
 low_density=density_fn(origins+directions*lo[:,None]).reshape(-1);hi=torch.where(low_density>=threshold,lo,hi)
 for _ in range(refine_steps):
  mid=(lo+hi)*.5;value=density_fn(origins+directions*mid[:,None]).reshape(-1);inside=value>=threshold;hi=torch.where(inside,mid,hi);lo=torch.where(inside,lo,mid)
 result[hit]=hi;return result
