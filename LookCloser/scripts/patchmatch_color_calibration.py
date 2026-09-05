"""Train-camera radiometric calibration primitives; no RGB source averaging."""
from __future__ import annotations
import numpy as np


def decode_exposed_linear(rgb):
    """Invert the ingest's known sRGB + per-channel Reinhard display curve."""
    rgb=np.asarray(rgb,dtype=np.float64)
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    return linear/np.maximum(1-linear,1e-6)


def encode_exposed_linear(exposed):
    exposed=np.maximum(np.asarray(exposed,dtype=np.float64),0)
    linear=exposed/(1+exposed)
    return np.clip(np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055),0,1)


def apply_camera_gain(rgb,gain,log_gain_grid=None):
    """CHW Torch RGB in [0,1]; diagonal gain in exposed-linear ingest domain."""
    import torch
    gain=torch.as_tensor(gain,device=rgb.device,dtype=rgb.dtype).reshape(3,1,1)
    if not bool(torch.isfinite(gain).all()) or not bool((gain>0).all()):
        raise ValueError('Camera gains must be positive finite RGB values')
    if log_gain_grid is not None:
        grid=torch.as_tensor(log_gain_grid,device=rgb.device,dtype=rgb.dtype)
        if grid.ndim!=2 or not bool(torch.isfinite(grid).all()):
            raise ValueError('Spatial exposure grid must be finite and 2D')
        field=torch.nn.functional.interpolate(grid[None,None],size=rgb.shape[-2:],mode='bilinear',align_corners=True)[0]
        gain=gain*torch.exp(field)
    linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
    exposed=linear/(1-linear).clamp_min(1e-6)*gain
    mapped=exposed/(1+exposed)
    return torch.where(mapped<=.0031308,mapped*12.92,1.055*mapped.pow(1/2.4)-.055).clamp(0,1)


def solve_relative_gains(count,pairs,deltas,weights):
    """Robust connected camera-graph fit, with geometric-mean gain one per channel."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs=np.asarray(pairs,dtype=int);deltas=np.asarray(deltas,dtype=float)
    if deltas.ndim==1:deltas=deltas[:,None]
    weights=np.asarray(weights,dtype=float)
    if len(pairs)!=len(deltas) or len(pairs)!=len(weights) or not len(pairs):
        raise ValueError('Invalid camera graph observations')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(count,count))
    if connected_components(graph,directed=False)[0]!=1:
        raise ValueError('Radiometric graph is disconnected; cannot choose a common gauge')
    design=np.zeros((len(pairs),count));design[np.arange(len(pairs)),pairs[:,0]]=1;design[np.arange(len(pairs)),pairs[:,1]]=-1
    result=np.zeros((count,deltas.shape[1]))
    for channel in range(deltas.shape[1]):
        w=weights.copy()
        for _ in range(5):
            lhs=np.vstack((design*w[:,None],np.ones((1,count))*max(w.sum(),1)))
            rhs=np.r_[deltas[:,channel]*w,0]
            result[:,channel]=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
            error=design@result[:,channel]-deltas[:,channel]
            sigma=max(float(np.median(np.abs(error)))*1.4826,.005)
            w=weights*np.sqrt(np.minimum(1,1.5*sigma/np.maximum(np.abs(error),1e-8)))
    return np.exp(result-result.mean(0))


def grid_basis(uv,width,height,grid_width,grid_height):
    """Four bilinear low-frequency gain-grid coefficients at image coordinates."""
    gx=np.clip(uv[:,0]/(width-1)*(grid_width-1),0,grid_width-1)
    gy=np.clip(uv[:,1]/(height-1)*(grid_height-1),0,grid_height-1)
    x=np.minimum(gx.astype(int),grid_width-2);y=np.minimum(gy.astype(int),grid_height-2)
    dx=gx-x;dy=gy-y
    ids=np.stack((y*grid_width+x,y*grid_width+x+1,(y+1)*grid_width+x,(y+1)*grid_width+x+1),-1)
    weights=np.stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy),-1)
    return ids,weights


def fit_spatial_exposure(frames,uv,valid,held,log_lum,base_gain,pairs,grid_width=8,grid_height=5,
                         smoothness_weight=10.,max_multiplier=1.25):
    """Smooth per-camera achromatic field, trained only on training surface blocks."""
    from scipy.sparse import coo_matrix,vstack,eye
    from scipy.sparse.linalg import lsqr
    if grid_width<2 or grid_height<2 or smoothness_weight<=0 or max_multiplier<1:
        raise ValueError('Invalid spatial exposure regularization')
    count=len(frames);nodes=grid_width*grid_height
    basis=[grid_basis(uv[i],f['w'],f['h'],grid_width,grid_height) for i,f in enumerate(frames)]
    rr=[];cc=[];vv=[];target=[];offset=0;rng=np.random.default_rng(19)
    adjusted=log_lum+np.log(base_gain[:,0])[:,None]
    for i,j in pairs:
        candidates=np.flatnonzero(valid[i]&valid[j]&~held)
        chosen=rng.choice(candidates,min(len(candidates),512),replace=False)
        n=len(chosen);row=np.repeat(np.arange(offset,offset+n),4)
        for camera,sign in [(i,1),(j,-1)]:
            ids,weights=basis[camera]
            rr.append(row);cc.append((camera*nodes+ids[chosen]).ravel());vv.append((sign*weights[chosen]).ravel())
        target.extend((adjusted[j,chosen]-adjusted[i,chosen]).tolist());offset+=n
    design=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(offset,count*nodes)).tocsr()
    target=np.asarray(target)
    # Neighbor regularization is independent of image content or semantic labels.
    a=[];b=[]
    for camera in range(count):
        ids=np.arange(nodes).reshape(grid_height,grid_width)+camera*nodes
        a.extend(ids[:,:-1].ravel());b.extend(ids[:,1:].ravel())
        a.extend(ids[:-1].ravel());b.extend(ids[1:].ravel())
    edges=len(a)
    smooth=coo_matrix((np.r_[np.full(edges,smoothness_weight),np.full(edges,-smoothness_weight)],
                       (np.r_[np.arange(edges),np.arange(edges)],np.r_[a,b])),shape=(edges,count*nodes)).tocsr()
    prior=eye(count*nodes,format='csr')*.5
    weights=np.sqrt(np.minimum(1,.15/np.maximum(np.abs(target),1e-8)))
    solution=np.zeros(count*nodes)
    for _ in range(2):
        lhs=vstack((design.multiply(weights[:,None]),smooth,prior),format='csr')
        rhs=np.r_[target*weights,np.zeros(edges+count*nodes)]
        solution=lsqr(lhs,rhs,atol=1e-6,btol=1e-6,iter_lim=300)[0]
        residual=design@solution-target
        weights=np.sqrt(np.minimum(1,.1/np.maximum(np.abs(residual),1e-8)))
    # Small monotonic exposure changes; never blur or warp source detail.
    solution=np.clip(solution,-np.log(max_multiplier),np.log(max_multiplier)).reshape(count,grid_height,grid_width)
    values=[]
    for i,(ids,w) in enumerate(basis):values.append((solution[i].ravel()[ids]*w).sum(-1))
    return solution,np.asarray(values),{'grid':[grid_width,grid_height],'training_equations':offset,
              'smoothness_weight':smoothness_weight,'zero_field_weight':.5,'max_multiplier':max_multiplier,'uses_eval_rgb':False}


def correct_projected_exposure(warped,valid_masks,grid_width=16,grid_height=9):
    """Train-source overlap fit in target coordinates; no target image is read.

    This diagnostic is view-dependent and must pass camera-path checks separately.
    RGB detail remains from one source; only a smooth scalar gain is applied.
    """
    count=len(warped);_,height,width=warped[0].shape
    yy,xx=np.mgrid[0:height:4,0:width:4]
    coords=np.column_stack((xx.ravel(),yy.ravel()))
    uv=np.broadcast_to(coords,(count,*coords.shape))
    rgb=np.stack([x[:,::4,::4].detach().cpu().numpy().reshape(3,-1).T for x in warped])
    valid=np.stack([x[::4,::4].detach().cpu().numpy().ravel() for x in valid_masks])
    valid&=(rgb>.1).all(-1)&(rgb<.9).all(-1)
    held=((coords[:,0]//32)*73856093^(coords[:,1]//32)*19349663)%5==0
    pairs=[(i,j) for i in range(count) for j in range(i+1,count) if (valid[i]&valid[j]&~held).sum()>=100]
    if not pairs:return warped,{'enabled':False,'reason':'no_visible_unclipped_train_overlap'}
    exposed=decode_exposed_linear(rgb);log_lum=np.log((exposed@np.array([.2126,.7152,.0722])).clip(1e-7))
    grid,values,stats=fit_spatial_exposure([dict(w=width,h=height)]*count,uv,valid,held,log_lum,
                                         np.ones((count,3)),pairs,grid_width,grid_height,1.,2.)
    corrected=encode_exposed_linear(exposed*np.exp(values[...,None]))
    before=[];after=[]
    for i,j in pairs:
        check=valid[i]&valid[j]&held
        before.extend(np.abs(rgb[i,check]-rgb[j,check]).mean(-1).tolist())
        after.extend(np.abs(corrected[i,check]-corrected[j,check]).mean(-1).tolist())
    stats.update(enabled=True,uses_eval_rgb=False,source_averaging=False,view_dependent=True,
                 gain_grids=grid.tolist(),validation_samples=len(before),
                 validation_pair_l1_before=float(np.median(before)) if before else None,
                 validation_pair_l1_after=float(np.median(after)) if after else None)
    return [apply_camera_gain(rgb,[1,1,1],field) for rgb,field in zip(warped,grid)],stats
