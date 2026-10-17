"""Read-only alternative surface statistics from the exact frozen ray weights.

This appends diagnostic depths; existing geometry, RGB, grids and calibration
are never overwritten. No photographs or evaluation masks enter this renderer.
"""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from luster_hybrid_common import DIGEST,geometry,log,native_rows,read,sha,write
from render_luster_hybrid import load,state_hashes,packed_quantiles


def packed_mode(outputs,rays):
    """Highest termination-PDF bin, using physical bin widths rather than mass."""
    indices=outputs['packed_ray_indices'].long();weights=outputs['packed_weights'].flatten().double()
    result=torch.full((len(rays),),torch.nan,device=weights.device)
    if not len(weights):return result
    starts=outputs['packed_spacing_starts'].flatten().double();ends=outputs['packed_spacing_ends'].flatten().double()
    span=(rays.fars-rays.nears).flatten().double()
    widths=(ends-starts)*span[indices]
    counts=torch.bincount(indices,minlength=len(rays));offsets=counts.cumsum(0)-counts
    cumulative=weights.cumsum(0);prefix=torch.cat([weights.new_zeros(1),cumulative])
    before=prefix[:-1]-prefix[offsets[indices]]
    transmission=(1-before).clamp_min(1e-12)
    optical=-torch.log1p(-(weights/transmission).clamp(0,1-1e-12))
    density_at_bin_start=transmission*optical/widths.clamp_min(1e-12)
    density_at_bin_start=torch.where(widths>0,density_at_bin_start,0)
    maximum=weights.new_full((len(rays),),-torch.inf)
    maximum.scatter_reduce_(0,indices,density_at_bin_start,reduce='amax',include_self=True)
    positions=torch.arange(len(weights),device=weights.device)
    selected=torch.where(density_at_bin_start==maximum[indices],positions,len(weights))
    argmax=torch.full((len(rays),),len(weights),dtype=torch.long,device=weights.device)
    argmax.scatter_reduce_(0,indices,selected,reduce='amin',include_self=True)
    total=weights.new_zeros(len(rays));total.scatter_add_(0,indices,weights)
    valid=(total>=.95)&(argmax<len(weights))
    # Piecewise-constant density yields an exponentially decreasing PDF inside
    # each interval, so its maximum is at the interval start.
    result[valid]=(rays.nears.flatten()[valid]+starts[argmax[valid]]*span[valid]).float()
    return result


def run(a):
    cfg,model=load(a.root,a.chunk)
    from render_luster_quality import camera_for
    state=state_hashes(model);rows=native_rows()
    for cid in a.cameras:
        parent=a.root/'geometry'/f'cam{cid:03d}'/'receipt.json';pr=read(parent)
        assert pr['state_hash']==state and pr['checkpoint_sha256']==DIGEST
        out=a.root/'surface_probe'/f'cam{cid:03d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'receipt.json').exists():
            r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
            assert all(sha(out/f'{key}.npy')==value for key,value in r['hashes'].items())
            continue
        row=rows[cid];h,w=row['h'],row['w'];camera=camera_for(row,model.device)
        arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=np.float32,shape=(h,w)) for k in ['q25','q75','mode']}
        original=geometry(a.root,cid);start_time=time.monotonic();max_median_error=0.
        with torch.inference_mode():
            for start in range(0,h*w,a.chunk):
                ids=torch.arange(start,min(start+a.chunk,h*w),device=model.device)
                coords=torch.stack([ids//w,ids%w],-1).float()+.5
                rays=model.collider(camera.generate_rays(camera_indices=0,coords=coords));outputs=model.get_outputs(rays)
                q=packed_quantiles(outputs,rays,(.25,.5,.75));mode=packed_mode(outputs,rays)
                z_scale=-(rays.directions@camera.camera_to_worlds[0,:3,:3])[:,2]
                q*=z_scale[:,None];mode*=z_scale
                q=q.cpu().numpy();mode=mode.cpu().numpy();stop=start+len(ids)
                expected=np.asarray(original['depth_z_q50']).reshape(-1)[start:stop]
                np.testing.assert_array_equal(np.isnan(expected),np.isnan(q[:,1]))
                good=np.isfinite(expected)
                if good.any():max_median_error=max(max_median_error,float(np.max(np.abs(expected[good]-q[good,1]))))
                for key,value in [('q25',q[:,0]),('q75',q[:,2]),('mode',mode)]:arrays[key].reshape(-1)[start:stop]=value
        assert max_median_error<=1e-6,'Frozen median reproduction changed'
        assert state_hashes(model)==state,'Model state changed'
        for v in arrays.values():v.flush()
        write(out/'receipt.json',dict(camera=cid,checkpoint_sha256=DIGEST,state_hash=state,code_sha256=sha(Path(__file__)),parent_geometry_sha256=sha(parent),max_median_error=max_median_error,hashes={k:sha(out/f'{k}.npy') for k in arrays},definitions={'q25':'conditional termination camera-z quantile .25','q75':'conditional termination camera-z quantile .75','mode':'camera-z at maximum piecewise-constant-density termination PDF bin start'},seconds=time.monotonic()-start_time))
        log(a.root,'surface_probe_complete',camera=cid,seconds=time.monotonic()-start_time,max_median_error=max_median_error)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cameras',nargs='+',type=int,required=True);p.add_argument('--chunk',type=int,default=4096);a=p.parse_args();run(a)
