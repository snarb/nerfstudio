"""Native termination-depth quantiles from saved traces; no GPU or renormalization."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def quantile_depth(start,end,sigma,weight,transmittance,q):
    """Unconditional ray quantile, exact within a constant-density interval."""
    arrays=[np.asarray(x,dtype=np.float64).reshape(-1)for x in [start,end,sigma,weight,transmittance]]
    start,end,sigma,weight,transmittance=arrays
    if not 0<q<1 or any(not np.isfinite(x).all()for x in arrays)or (weight<0).any():raise ValueError('Invalid finite ray distribution')
    cumulative=np.cumsum(weight)
    if not len(cumulative)or cumulative[-1]<q:return None
    index=int(np.searchsorted(cumulative,q,side='left'));previous=cumulative[index-1]if index else 0.
    if sigma[index]<=0 or transmittance[index]<=0:raise ValueError('Positive target mass in zero-density interval')
    fraction=(q-previous)/transmittance[index]
    distance=-np.log1p(-np.clip(fraction,0.,1.-np.finfo(float).eps))/sigma[index]
    result=start[index]+distance
    if result>end[index]+1e-5:raise ValueError('Stored weights/transmittance do not match interval density')
    return float(min(result,end[index]))


def stats(values):
    a=np.array([x for x in values if x is not None],float)
    return dict(valid=len(a),unknown=len(values)-len(a),mean=float(a.mean())if len(a)else None,
                median=float(np.median(a))if len(a)else None,p10=float(np.quantile(a,.1))if len(a)else None,
                p90=float(np.quantile(a,.9))if len(a)else None,max_abs=float(abs(a).max())if len(a)else None)


def main(args):
    comparison=json.loads(args.comparison.read_text());output={}
    for name,model in comparison['models'].items():
        folder=Path(model['source'])/'clothing103_trace/rays';audit_path=folder/'audit.json';audit=json.loads(audit_path.read_text())
        if hashlib.sha256(audit_path.read_bytes()).hexdigest()!=comparison['files'][str(audit_path)]:raise ValueError('Saved trace audit changed')
        rows=[]
        for record in audit['rows']:
            pixel=record['pixel'];point=np.array(pixel['witness_xyz']);origin=np.array(record['origin']);direction=np.array(record['direction']);depth=float((point-origin)@direction)
            path=folder/record['profiles']['native']['trace']
            with np.load(path,allow_pickle=False)as z:
                start,end,sigma,weight,trans=[z[k].reshape(-1)for k in ['t_start','t_end','sigma','weights','transmittance']]
            quantiles={str(q):quantile_depth(start,end,sigma,weight,trans,q)for q in [.1,.5,.9]}
            residual=None if quantiles['0.5']is None else quantiles['0.5']-depth
            spread=None if quantiles['0.1']is None or quantiles['0.9']is None else quantiles['0.9']-quantiles['0.1']
            rows.append(dict(point=pixel['witness_id'],camera=pixel['camera_name'],actor=pixel['actor'],region=pixel['region'],group=pixel['diagnostic_group'],
                witness_depth=depth,native_quantiles=quantiles,signed_median_residual=residual,central10_90_width=spread,
                transmission=max(0.,1.-float(np.sum(weight,dtype=np.float64))),point_to_pixel_ray_distance=pixel['point_to_pixel_ray_distance'],
                declared_positive_band_halfwidth=max(.005,.01*depth),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        groups={}
        for group in sorted({r['group']for r in rows}):
            subset=[r for r in rows if r['group']==group]
            groups[group]=dict(rays=len(subset),signed_median_depth_residual=stats([r['signed_median_residual']for r in subset]),
                absolute_median_depth_residual=stats([abs(r['signed_median_residual'])if r['signed_median_residual']is not None else None for r in subset]),
                central10_90_width=stats([r['central10_90_width']for r in subset]),
                band_halfwidth=stats([r['declared_positive_band_halfwidth']for r in subset]))
        output[name]=dict(rows=rows,groups=groups)
    result=dict(kind='EXACT_NATIVE_UNCONDITIONAL_DEPTH_QUANTILES',models=output,comparison_sha256=hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
        quantile_definition='10/50/90% of original ray probability, exact exponential inversion within stored intervals; no opacity renormalization; missing quantiles unknown',
        units='Prepared-world units, not certified meters',signed_residual='Native t50 minus observed witness projection; negative is in front',
        caveats=['Median and10–90 width include all foreground/fog layers, not only a selected surface peak.','Band containment does not imply thin cloth or zero depth bias.','SfM witness, calibration and pixel-ray residual remain uncertain.'],optimizer_updates=0,GPU_used=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(args.output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
