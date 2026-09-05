"""One-sided harmonic gain leveling of hard-selected train texture patches.

Boundary gains use ONE lower-rank selected source at a mutually visible point.
They are not an average of source RGB. Harmonic extension modifies only exposure/
chromatic response; source detail and the categorical source map remain intact.
"""
from __future__ import annotations
import math
import torch
from patchmatch_color_calibration import apply_camera_gain
from surface_color_field import solve_surface_field


def exposed_log(rgb):
    linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
    return (linear/(1-linear).clamp_min(1e-6)).clamp_min(1e-7).log()


def level_hard_source_seams(prediction,selection,warped,valid_masks,depth):
    """A lower-rank source defines the color gauge; genuine depth edges are barred."""
    if prediction.ndim!=3 or prediction.shape[0]!=3 or selection.shape!=depth.shape:
        raise ValueError('Expected CHW RGB and HW source/depth maps')
    output=prediction.clone();gain_map=torch.zeros_like(prediction);rows=[]
    logz=depth.clamp_min(1e-7).log();height,width=depth.shape
    for rank in range(1,len(warped)):
        region=(selection==rank)&(depth>0)
        if not bool(region.any()):continue
        # Padding guarantees an actual neighbor, never wrapped image coordinates.
        data=torch.zeros_like(prediction);chosen=torch.full_like(selection,len(warped))
        source_log=exposed_log(warped[rank]);reference_log=exposed_log(output)
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            y0,y1=max(0,-dy),min(height,height-dy);x0,x1=max(0,-dx),min(width,width-dx)
            a=(slice(y0,y1),slice(x0,x1));b=(slice(y0+dy,y1+dy),slice(x0+dx,x1+dx))
            lower=selection[b]
            accept=region[a]&(lower>=0)&(lower<rank)&(lower<chosen[a])&valid_masks[rank][b]
            accept&=(depth[b]>0)&((logz[a]-logz[b]).abs()<.0075)
            # Compare RGB at the SAME neighbor location; preserve real image edges.
            delta=(reference_log[(slice(None),*b)]-source_log[(slice(None),*b)]).clamp(-math.log(2),math.log(2))
            chosen[a]=torch.where(accept,lower,chosen[a])
            data[(slice(None),*a)]=torch.where(accept[None],delta,data[(slice(None),*a)])
        seeds=chosen<len(warped)
        if not bool(seeds.any()):
            rows.append({'rank':rank,'pixels':int(region.sum()),'seeds':0,'corrected':False});continue
        yy,xx=torch.where(region)
        y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
        local_region=region[y0:y1,x0:x1]
        local_depth=torch.where(local_region,depth[y0:y1,x0:x1],0)
        # Boundary seeds dominate ||b||. A 1e-4 relative residual can leave a
        # broad interior almost uncorrected while declaring convergence. Solve
        # this opt-in leveling system in float64 with a strict true residual.
        field,solver=solve_surface_field(data[None,:,y0:y1,x0:x1].double(),
                  seeds[None,None,y0:y1,x0:x1].double()*128,local_depth.double(),
                  smoothness=1.,ridge=1e-6,max_iterations=4096,tolerance=1e-9)
        solver.update(dtype='float64',required_true_relative_residual=5e-9,
                      converged=solver['max_relative_residual']<5e-9)
        if not solver['converged']:raise RuntimeError(f'Seam gain solve failed for source {rank}: {solver}')
        field=field[0].to(prediction.dtype).clamp(-math.log(2),math.log(2))
        corrected=apply_camera_gain(warped[rank][:,y0:y1,x0:x1],[1,1,1],field)
        output[:,y0:y1,x0:x1]=torch.where(local_region[None],corrected,output[:,y0:y1,x0:x1])
        gain_map[:,y0:y1,x0:x1]=torch.where(local_region[None],field,gain_map[:,y0:y1,x0:x1])
        rows.append({'rank':rank,'pixels':int(region.sum()),'seeds':int(seeds.sum()),'corrected':True,'solver':solver,
                     'gain_min':float(field.exp()[local_region[None].expand_as(field)].min()),
                     'gain_max':float(field.exp()[local_region[None].expand_as(field)].max())})
    return output,gain_map,{'enabled':True,'uses_eval_rgb':False,'uses_semantic_masks':False,'source_averaging':False,
                           'source_labels_unchanged':True,'primary_unchanged':True,'view_dependent':True,
                           'method':'one_sided_harmonic_exposed_linear_log_gain','patches':rows}
