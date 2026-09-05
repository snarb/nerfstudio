"""Local train-camera bandwidth graphs for hard source selection only.

Each edge measures a relative blur variance. Withholding whole camera-pair edges
checks cycle consistency; this is not independent spatial/scene validation.
Only the source cost field is smoothed. RGB, geometry and visibility are untouched.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
import numpy as np

from source_bandwidth_prior import bandwidth_observations, bandwidth_source_costs


def fit_camera_graph(observations, *, minimum_samples=3, maximum_median_error=.35,
                     maximum_p90_error=.75):
    """Fit variance[j]-variance[i], with a leave-one-camera-pair-out gate."""
    grouped = defaultdict(list)
    for r in observations:
        i,j = int(r['primary_rank']),int(r['source_rank'])
        value = float(r['relative_blur_variance'])
        if i < 0 or i >= j or not np.isfinite(value):
            raise ValueError('Invalid ordered camera-pair observation')
        grouped[i,j].append(value)
    edges = [(i,j,float(np.median(v))) for (i,j),v in sorted(grouped.items()) if len(v)>=minimum_samples]
    nodes = sorted({n for i,j,_ in edges for n in (i,j)})
    empty = {'qualified':False,'nodes':nodes,'edges':len(edges),'held_edges':0,'relative_variances':{}}
    if len(nodes)<3 or len(edges)<len(nodes)+1:
        return dict(empty,reason='insufficient_redundant_edges')
    index = {n:i for i,n in enumerate(nodes)}
    design = np.zeros((len(edges),len(nodes)),np.float64)
    for k,(i,j,_) in enumerate(edges):design[k,index[i]]=-1;design[k,index[j]]=1
    values = np.array([v for _,_,v in edges])
    if np.linalg.matrix_rank(design)<len(nodes)-1:
        return dict(empty,reason='disconnected_graph')
    checks = []
    for k,(i,j,v) in enumerate(edges):
        use = np.arange(len(edges))!=k
        if np.linalg.matrix_rank(design[use])<len(nodes)-1:
            return dict(empty,reason='unvalidated_bridge')
        fit = np.linalg.lstsq(design[use],values[use],rcond=None)[0]
        prediction = float(design[k]@fit)
        checks.append({'pair':[i,j],'observed':v,'predicted':prediction,'absolute_error':abs(prediction-v)})
    errors = [r['absolute_error'] for r in checks]
    median,p90 = np.quantile(errors,[.5,.9])
    qualified = median<=maximum_median_error and p90<=maximum_p90_error
    fit = np.linalg.lstsq(design,values,rcond=None)[0]
    return {'qualified':bool(qualified),'nodes':nodes,'edges':len(edges),'held_edges':len(checks),
            'median_absolute_error':float(median),'p90_absolute_error':float(p90),
            'relative_variances':{n:float(v) for n,v in zip(nodes,fit)},'checks':checks,
            'reason':'qualified' if qualified else 'inconsistent_camera_pair_cycles'}


def quality_field_from_graphs(models, shape, depth, baseline_relative, *, device='cpu'):
    """Smooth scalar camera costs on depth-connected pixels, never source RGB."""
    import torch
    from surface_color_field import solve_surface_field
    count,height,width = shape
    if depth.shape!=(height,width) or baseline_relative.shape!=(count,):
        raise ValueError('Invalid source-quality field dimensions')
    if not np.isfinite(depth).all() or not np.isfinite(baseline_relative).all() or (depth<0).any():
        raise ValueError('Nonfinite/negative source-quality input')
    target = np.broadcast_to(baseline_relative[:,None,None],shape).astype(np.float32).copy()
    weights = np.full(shape,.1,np.float32)
    layer = np.floor(np.log(np.maximum(depth.astype(np.float64),1e-9))/.01).astype(np.int32)
    assigned = 0
    for model in models:
        if not model['qualified']:continue
        bx,by,dz = model['cell'];x0,y0=bx*128,by*128;x1,y1=min(x0+128,width),min(y0+128,height)
        if min(x0,y0)<0 or x0>=width or y0>=height:raise ValueError('Model cell outside raster')
        local = (depth[y0:y1,x0:x1]>0)&(layer[y0:y1,x0:x1]==dz)
        nodes = model['nodes'];values=np.array([model['relative_variances'][n] for n in nodes])
        if 0 in nodes:
            offset = -model['relative_variances'][0]
        else:
            # An unobserved camera retains its original global prior. Align the
            # local relative gauge using only represented cameras, without RGB.
            offset = float(np.median(baseline_relative[nodes]-values))
        for n,v in zip(nodes,values):
            target[n,y0:y1,x0:x1][local]=np.clip(v+offset,-6.25,6.25)
            weights[n,y0:y1,x0:x1][local]=1.
        assigned += int(local.sum())
    if assigned==0:return target,{'enabled':False,'reason':'no_qualified_local_support'}
    field,solver=solve_surface_field(torch.as_tensor(target[:,None],device=device,dtype=torch.float64),
                 torch.as_tensor(weights[:,None],device=device,dtype=torch.float64),
                 torch.as_tensor(depth,device=device,dtype=torch.float64),
                 smoothness=64.,ridge=1e-6,max_iterations=1536,tolerance=1e-7)
    if solver['max_relative_residual']>5e-7:
        raise RuntimeError(f'Local bandwidth field failed its true-residual gate: {solver}')
    return field[:,0].float().cpu().numpy(),dict(solver,enabled=True,assigned_pixels=assigned,dtype='float64')


def local_bandwidth_source_costs(rgb, valid, depth, penalty, *, device='cpu'):
    if not math.isfinite(penalty) or penalty<=0:
        raise ValueError('Need a positive finite local bandwidth penalty')
    if rgb.ndim!=4 or valid.shape!=rgb.shape[:-1] or depth.shape!=valid.shape[1:]:
        raise ValueError('Invalid RGB/visibility/depth shapes')
    global_costs,global_audit=bandwidth_source_costs(rgb,valid,penalty,allow_primary_penalty=True)
    baseline=global_costs[:,0,0]/penalty
    baseline-=baseline[0]
    gray=np.ascontiguousarray(rgb@np.array([.2126,.7152,.0722],np.float32),dtype=np.float32)
    groups=defaultdict(list)
    for i,j in combinations(range(len(rgb)),2):
        observations=bandwidth_observations(gray[i],gray[j],valid[i],valid[j],stride=8,patch_size=24)
        for row in observations:
            x,y=row['x'],row['y'];bx,by=x//128,y//128
            if ((x-20)//128!=bx or (x+19)//128!=bx or (y-20)//128!=by or (y+19)//128!=by):continue
            window=depth[y-20:y+20,x-20:x+20]
            if not np.isfinite(window).all() or (window<=0).any() or np.log(window.max()/window.min())>.0075:continue
            dz=int(np.floor(np.log(float(depth[y,x]))/.01))
            groups[bx,by,dz].append({'primary_rank':i,'source_rank':j,'relative_blur_variance':row['relative_blur_variance']})
    models=[dict(fit_camera_graph(rows),cell=list(key),patch_observations=len(rows)) for key,rows in sorted(groups.items())]
    relative,solver=quality_field_from_graphs(models,valid.shape,depth,baseline,device=device)
    costs=np.maximum(relative-relative.min(0,keepdims=True),0)*penalty
    return costs.astype(np.float32),{'enabled':True,'uses_eval_rgb':False,'uses_semantic_masks':False,
            'modifies_rgb':False,'source_averaging':False,'modifies_visibility':False,
            'penalty_per_pixel_variance':penalty,'patch_size':24,'search_radius':8,'stride':8,
            'spatial_block':128,'log_depth_bin':.01,'single_layer_maximum_log_range':.0075,
            'validation':'leave_one_camera_pair_out_cycle_consistency_not_independent_scene_validation',
            'qualification':{'minimum_edge_patches':3,'median_error_maximum':.35,'p90_error_maximum':.75},
            'qualified_cells':sum(m['qualified'] for m in models),'cells':models,'solver':solver,
            'global_fallback':global_audit,'interpretation':__doc__}
