"""Post-hoc DC failure rays: no depth truth, no independent selection metric."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from summarize_marshawn_depth_residuals import quantile_depth


def compare(c,out):
    query=c/'independent_occupancy/DC_error12_15188/queries.json';q=json.loads(query.read_text());models={};files={}
    for selected in json.loads((out.parent/'selected_models.json').read_text()):
        name=selected['name'];root=Path(selected['archive']);folder=root/'DC12_trace/rays';audit=json.loads((folder/'audit.json').read_text());chain=json.loads((root/'DC12_trace/coordinate_chain.json').read_text());health=json.loads((root/'DC12_producer_health.json').read_text())
        if audit['step']!=selected['step'] or not audit['frozen_state_unchanged']or not health['audit_environment_healthy']:raise ValueError('Immutable selected and healthy producer required')
        if chain['queries_sha256']!=hashlib.sha256(query.read_bytes()).hexdigest()or chain['feature_reweighting_strength']!=(1. if selected['step']<=75940 else .3) or chain['density_activation']!='trunc_exp'or not chain['correct_sh_directions']:raise ValueError('Actual field or query changed')
        assert [r['pixel']for r in audit['rows']]==q['pixels'];rows=[]
        for r in audit['rows']:
            p=r['pixel'];gt=np.array(p['gt_rgb']);profiles={};convergence=r['uniform_convergence'];converged=convergence['opacity_abs_difference']<=.005 and convergence['foreground_rgb_max_abs_difference']<=1/255
            for profile in ['native','halo1','all_cells','uniform','uniform2x']:
                v=r['profiles'][profile];rgb=np.array(v['integrated_foreground_rgb']);profiles[profile]=dict(opacity=v['opacity'],rgb=rgb.tolist(),max_abs_RGB_error=float(abs(rgb-gt).max()),source_trace=v['trace'])
            n=profiles['native']
            if True:
                from PIL import Image
                png=np.array(Image.open(root/'renders/DC.png').convert('RGB'))[p['y'],p['x']]/255.
            else:png=np.array(p['saved_png_rgb'][name])
            png_error=float(abs(np.array(n['rgb'])-png).max())
            if png_error>1.5/255:raise ValueError('Native diagnostic replay differs from saved selected RGB raster')
            with np.load(folder/r['profiles']['native']['trace'])as z:
                a=[z[k].reshape(-1)for k in ['t_start','t_end','sigma','weights','transmittance']]
                quantiles={str(v):quantile_depth(*a,v)for v in [.1,.5,.9]}
            rows.append(dict(pixel=p,profiles=profiles,uniform_converged=converged,native_matches_saved_png_max_abs=png_error,native_unconditional_depth_quantiles=quantiles,
                native_halo_max_RGB_change=float(np.max(abs(np.array(n['rgb'])-profiles['halo1']['rgb']))),depth_truth_available=False))
        models[name]=dict(rows=rows,step=selected['step'],checkpoint_sha256=audit['checkpoint_sha256'],coordinate_chain=chain,producer_health=health,actual_parameter_binding=json.loads((root/'DC12_trace/actual_parameter_binding.json').read_text()))
        for p in [folder/'audit.json',root/'DC12_trace/coordinate_chain.json',root/'DC12_producer_health.json']:files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    out.mkdir(exist_ok=True,parents=True);result=dict(kind='POSTHOC_DC12_NATIVE_RGB_OCCUPANCY_DIAGNOSTIC',models=models,query_sha256=hashlib.sha256(query.read_bytes()).hexdigest(),files=files,
        source='Selected from GT clothing and actor15188 RGB failure; diagnostic only, never training or independent model ranking.',depth_ground_truth_available=False,training_allowed=False,model_promotion_authorized=False,
        caveats=['Opacity near1 means a wrong-colored opaque rendering, not an empty transparent ray.','Halo/dense RGB changes probe the learned field, not certified geometry in skipped cells.','No independently triangulated depth truth exists for these selected eval pixels.'])
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Post-hoc DC failure-ray audit','', 'Twelve eval rays, never TRAIN input. No depth ground truth or independent ranking metric.','', '| Pixel | Model | Opacity | Max RGB error | Native→halo RGB change |','|---|---|---:|---:|---:|']
    for i,p in enumerate(q['pixels']):
        for name,m in models.items():
            r=m['rows'][i];n=r['profiles']['native'];lines.append(f'| {p["x"]},{p["y"]} {p["region"]} | {name} | {n["opacity"]:.6f} | {n["max_abs_RGB_error"]:.5f} | {r["native_halo_max_RGB_change"]:.5f} |')
    (out/'report.md').write_text('\n'.join(lines)+'\n');(out/'complete.json').write_text(json.dumps(dict(complete=True,all_selected_rays_converged=all(r['uniform_converged']for m in models.values()for r in m['rows']),all_native_match_saved_png=True,production_mutated=False),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();compare(a.campaign,a.output)
