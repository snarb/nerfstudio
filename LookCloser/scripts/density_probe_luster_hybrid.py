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


def packed_density_surface(outputs,rays):
    """Intrinsic density peak and density-squared centroid inside q10--q90.

    Optical depth is recovered from frozen termination weights / transmittance.
    The central termination interval excludes unseen density behind the surface.
    No RGB enters this readout; physical interval widths avoid sample-count bias.
    """
    indices=outputs['packed_ray_indices'].long();weights=outputs['packed_weights'].flatten().double()
    result=torch.full((len(rays),2),torch.nan,device=weights.device)
    if not len(weights):return result
    starts=outputs['packed_spacing_starts'].flatten().double();ends=outputs['packed_spacing_ends'].flatten().double()
    span=(rays.fars-rays.nears).flatten().double();near=rays.nears.flatten().double()
    left=near[indices]+starts*span[indices];right=near[indices]+ends*span[indices]
    widths=right-left
    counts=torch.bincount(indices,minlength=len(rays));offsets=counts.cumsum(0)-counts
    prefix=torch.cat([weights.new_zeros(1),weights.cumsum(0)])
    before=prefix[:-1]-prefix[offsets[indices]]
    transmission=(1-before).clamp_min(1e-12)
    optical=-torch.log1p(-(weights/transmission).clamp(0,1-1e-12))
    density=optical/widths.clamp_min(1e-12)
    q=packed_quantiles(outputs,rays,(.1,.9)).double()
    left=torch.maximum(left,q[indices,0]);right=torch.minimum(right,q[indices,1])
    overlap=(right-left).clamp_min(0)
    eligible=(overlap>0)&torch.isfinite(density)&(density>0)
    peak=torch.where(eligible,density,-torch.inf)
    maximum=weights.new_full((len(rays),),-torch.inf)
    maximum.scatter_reduce_(0,indices,peak,reduce='amax',include_self=True)
    positions=torch.arange(len(weights),device=weights.device)
    chosen=torch.where(eligible&(peak==maximum[indices]),positions,len(weights))
    argmax=torch.full((len(rays),),len(weights),dtype=torch.long,device=weights.device)
    argmax.scatter_reduce_(0,indices,chosen,reduce='amin',include_self=True)
    valid=(argmax<len(weights))&torch.isfinite(q).all(-1)
    midpoint=(left+right)*.5
    result[valid,0]=midpoint[argmax[valid]].float()
    # Scale by the ray maximum before squaring, avoiding overflow. This is an
    # intrinsic density statistic, not a termination-mass or texture fit.
    mass=torch.where(eligible,(density/maximum[indices].clamp_min(1e-12)).square()*overlap,0.)
    denominator=weights.new_zeros(len(rays));numerator=weights.new_zeros(len(rays))
    denominator.scatter_add_(0,indices,mass)
    numerator.scatter_add_(0,indices,torch.where(eligible,mass*midpoint,0.))
    result[valid,1]=(numerator[valid]/denominator[valid]).float()
    return result


def run(a):
    cfg,model=load(a.root,a.chunk)
    from render_luster_quality import camera_for
    state=state_hashes(model);rows=native_rows()
    for cid in a.cameras:
        parent=a.root/'geometry'/f'cam{cid:03d}'/'receipt.json';pr=read(parent)
        assert pr['state_hash']==state and pr['checkpoint_sha256']==DIGEST
        out=a.root/'density_probe'/f'cam{cid:03d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'receipt.json').exists():
            r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
            assert all(sha(out/f'{key}.npy')==value for key,value in r['hashes'].items())
            continue
        row=rows[cid];h,w=row['h'],row['w'];camera=camera_for(row,model.device)
        arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=np.float32,shape=(h,w)) for k in ['density_peak','density_center']}
        original=geometry(a.root,cid);start_time=time.monotonic();max_median_error=0.
        with torch.inference_mode():
            for start in range(0,h*w,a.chunk):
                ids=torch.arange(start,min(start+a.chunk,h*w),device=model.device)
                coords=torch.stack([ids//w,ids%w],-1).float()+.5
                rays=model.collider(camera.generate_rays(camera_indices=0,coords=coords));outputs=model.get_outputs(rays)
                q=packed_quantiles(outputs,rays,(.25,.5,.75));surface=packed_density_surface(outputs,rays)
                z_scale=-(rays.directions@camera.camera_to_worlds[0,:3,:3])[:,2]
                q*=z_scale[:,None];surface*=z_scale[:,None]
                q=q.cpu().numpy();surface=surface.cpu().numpy();stop=start+len(ids)
                expected=np.asarray(original['depth_z_q50']).reshape(-1)[start:stop]
                np.testing.assert_array_equal(np.isnan(expected),np.isnan(q[:,1]))
                good=np.isfinite(expected)
                if good.any():max_median_error=max(max_median_error,float(np.max(np.abs(expected[good]-q[good,1]))))
                for key,value in [('density_peak',surface[:,0]),('density_center',surface[:,1])]:arrays[key].reshape(-1)[start:stop]=value
        assert max_median_error<=1e-6,'Frozen median reproduction changed'
        assert state_hashes(model)==state,'Model state changed'
        for v in arrays.values():v.flush()
        write(out/'receipt.json',dict(camera=cid,checkpoint_sha256=DIGEST,state_hash=state,code_sha256=sha(Path(__file__)),parent_geometry_sha256=sha(parent),max_median_error=max_median_error,hashes={k:sha(out/f'{k}.npy') for k in arrays},definitions={'density_peak':'camera-z midpoint of maximum intrinsic-density bin clipped to termination q10--q90','density_center':'camera-z density-squared physical-width centroid within termination q10--q90'},seconds=time.monotonic()-start_time))
        log(a.root,'density_probe_complete',camera=cid,seconds=time.monotonic()-start_time,max_median_error=max_median_error)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cameras',nargs='+',type=int,required=True);p.add_argument('--chunk',type=int,default=4096);a=p.parse_args();run(a)
