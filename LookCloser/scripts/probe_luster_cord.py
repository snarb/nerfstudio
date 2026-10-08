#!/usr/bin/env python3
"""Read-only cord crops, PatchMatch support and native ray termination audit."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from export_luster_opacity import load
from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import score, sha, write


def probe(selection, output, compare_sampling=False, inspect_layers=False):
    if output.exists():
        raise FileExistsError('Preserve existing cord audits')
    selected = json.loads(selection.read_text())
    selected = selected.get('selected', selected)
    trainer, request = load(selected)
    model = trainer.pipeline.model
    before = field_fingerprint(model)
    root = Path(request['data']).parent
    manifests = {n: json.loads((root/n/'manifest.json').read_text())
                 for n in ['depth_confidence', 'depth_confidence_min6']}
    if any(m['transforms_sha256'] != sha(root/'data/transforms.json') or not m['train_only']
           for m in manifests.values()):
        raise ValueError('Require matching training-only calibrated depth')
    output.mkdir(parents=True)
    rows, panels = [], []
    with torch.no_grad():
        for cid, box in [(95, (850, 1200, 925, 1406)), (97, (810, 1150, 900, 1406))]:
            ds = (trainer.pipeline.datamanager.eval_dataset if cid == 95 else
                  trainer.pipeline.datamanager.train_dataset)
            index = next(i for i, p in enumerate(ds.image_filenames) if int(p.stem.split('_')[1]) == cid)
            name = ds.image_filenames[index].name
            x0, y0, x1, y1 = box
            yy, xx = torch.meshgrid(torch.arange(y0, y1, device=model.device),
                                    torch.arange(x0, x1, device=model.device), indexing='ij')
            camera = ds.cameras[index:index+1].to(model.device)
            rays = camera.generate_rays(0, coords=torch.stack([yy, xx], -1).float()+.5)
            gt = ds[index]['image'][y0:y1, x0:x1].to(model.device).float()
            images = [('gt', gt)]
            metrics = {}
            modes = [('native', {})]
            if compare_sampling:
                modes.append(('dense', dict(adaptive_coarse_step_size=.001, max_steps_per_ray=4096)))
            for label, overrides in modes:
                original = {k: getattr(model.config, k) for k in overrides}
                for k, v in overrides.items():
                    setattr(model.config, k, v)
                result = model.get_outputs_for_camera_ray_bundle(rays)
                rgb = result['rgb'].float()
                metrics[label] = score(model, rgb, gt)
                images.append((label, rgb))
                for k, v in original.items():
                    setattr(model.config, k, v)
            layer_summary = None
            native_layer_rays = []
            if inspect_layers:
                # Opaque dominant-contributor counterfactual, with native
                # geometry/colors frozen. This is a diagnostic, not a renderer
                # or an independently established cord-surface estimate.
                flat = rays.flatten()
                winner_colors = np.zeros((len(flat), 3), dtype=np.float32)
                winner_masses, cell_alphas, widths = [], [], []
                parity = 0.
                annotated = ([(879,1260),(880,1280),(881,1300),(882,1320),
                              (883,1340),(884,1360)] if cid==95 else
                             [(838,1170),(840,1200),(841,1240),(842,1270),
                              (844,1295),(844,1340),(846,1360)])
                annotated_ids = {(y-y0)*(x1-x0)+x-x0:(x,y) for x,y in annotated}
                cell_length = float(((model.occupancy_grid.aabbs[0][3:]-model.occupancy_grid.aabbs[0][:3])/
                                     model.occupancy_grid.resolution.to(model.occupancy_grid.aabbs)).norm())
                for start in range(0, len(flat), 512):
                    end = min(start+512, len(flat)); br = flat[start:end]
                    captured = {}
                    def capture_native_field(module, args, values):
                        from nerfstudio.field_components.field_heads import FieldHeadNames
                        frustums = args[0].frustums
                        captured['t'] = ((frustums.starts+frustums.ends)/2).flatten()
                        captured['sigma'] = values[FieldHeadNames.DENSITY].flatten().float()
                        captured['color'] = values[FieldHeadNames.RGB].float()
                    hook = model.field.register_forward_hook(capture_native_field)
                    try:
                        out = model(br)
                    finally:
                        hook.remove()
                    ids = out['packed_ray_indices'].flatten().long()
                    w = out['packed_weights'].flatten().float()
                    t,sigma,color = [captured[k] for k in ['t','sigma','color']]
                    if len(t)!=len(w):
                        raise ValueError('Native field hook/sample count differs')
                    reconstructed = torch.zeros(end-start,3,device=model.device).index_add_(0,ids,w[:,None]*color)
                    parity = max(parity,float((reconstructed-out['rgb']).abs().max()))
                    ni,nw,nt,ns,nc=[v.cpu().numpy() for v in [ids,w,t,sigma,color]]
                    if len(ni)>1 and (np.diff(ni)<0).any():
                        raise ValueError('Expected ray-grouped packed samples')
                    bounds=np.r_[0,np.cumsum(np.bincount(ni,minlength=end-start))]
                    for i in range(end-start):
                        left,right=bounds[i:i+2]
                        if left==right:
                            continue
                        ww = nw[left:right];best=left+ww.argmax()
                        winner_colors[start+i] = nc[best]
                        winner_masses.append(float(nw[best]))
                        cell_alphas.append(float(-np.expm1(-ns[best]*cell_length)))
                        cdf=np.cumsum(ww)/max(float(ww.sum()),1e-8)
                        quantiles=np.searchsorted(cdf,[.05,.95]).clip(0,right-left-1)
                        widths.append(float(nt[left+quantiles[1]]-nt[left+quantiles[0]]))
                        if start+i in annotated_ids:
                            active=np.flatnonzero(ww>1e-4)+left
                            groups=np.split(active,np.flatnonzero(np.diff(nt[active])>.03)+1)
                            clusters=[]
                            for group in groups:
                                if not len(group):continue
                                mass=float(nw[group].sum())
                                clusters.append(dict(t_min=float(nt[group].min()),t_max=float(nt[group].max()),
                                    mass=mass,conditional_rgb=((nw[group,None]*nc[group]).sum(0)/max(mass,1e-8)).tolist()))
                            native_layer_rays.append(dict(xy=list(annotated_ids[start+i]),
                                dominant_t=float(nt[best]),dominant_weight=float(nw[best]),
                                dominant_rgb=nc[best].tolist(),termination_width=widths[-1],clusters=clusters))
                if parity>2e-4:
                    raise ValueError(f'Native color reconstruction mismatch {parity}')
                winner = torch.from_numpy(winner_colors).to(model.device).reshape(y1-y0,x1-x0,3)
                images.append(('opaque_dominant_sample',winner))
                metrics['opaque_dominant_sample'] = score(model,winner,gt)
                layer_summary = dict(native_rgb_reconstruction_max_error=parity,
                    dominant_weight_quantiles=np.quantile(winner_masses,[.1,.5,.9]).tolist(),
                    dominant_cell_opacity_quantiles=np.quantile(cell_alphas,[.1,.5,.9]).tolist(),
                    termination_90pct_width_quantiles=np.quantile(widths,[.1,.5,.9]).tolist(),
                    scope='Whole narrow crop, including neighboring fabric; dominant sample is not certified cord depth')
            panel = Image.new('RGB', ((x1-x0)*len(images), y1-y0))
            for i, (label, value) in enumerate(images):
                im = Image.fromarray(np.rint(value.cpu().numpy().clip(0, 1)*255).astype('uint8'))
                im.save(output/f'cam{cid:03d}_{label}.png')
                panel.paste(im, ((x1-x0)*i, 0))
            panels.append((cid, panel))
            row = dict(camera=cid, box=box, metrics=metrics)
            if layer_summary is not None:
                row['layer_summary'] = layer_summary
                row['native_layer_rays'] = native_layer_rays
            if cid == 97:
                maps = {}
                for n, m in manifests.items():
                    receipt = m['maps'][name]; path = root/n/receipt['path']
                    if sha(path) != receipt['sha256']:
                        raise ValueError('Depth archive changed')
                    with np.load(path) as a:
                        maps[n] = {k: a[k].copy() for k in a.files}
                # Manually inspected cord centreline; diagnostic annotation only.
                points = [(838,1170),(840,1200),(841,1240),(842,1270),
                          (844,1295),(844,1340),(846,1360)]
                coords = torch.tensor([[y,x] for x,y in points], device=model.device).float()+.5
                pr = camera.generate_rays(0, coords=coords)
                out = model(pr)
                ids = out['packed_ray_indices'].flatten().cpu().numpy()
                weights = out['packed_weights'].flatten().float().cpu().numpy()
                lo, hi = [out[k].flatten().cpu().numpy() for k in ['packed_spacing_starts','packed_spacing_ends']]
                near, far = [getattr(pr,k).flatten().cpu().numpy() for k in ['nears','fars']]
                t = near[ids]+(far-near)[ids]*(lo+hi)/2
                norm = pr.metadata['directions_norm'].flatten().cpu().numpy()
                a, b = maps.values()
                measurements = []
                for i,(x,y) in enumerate(points):
                    depth = float(a['depth'][y,x])*norm[i]
                    eps = max(2*float(a['uncertainty'][y,x])*norm[i], .003)
                    w, ti = weights[ids==i], t[ids==i]
                    measured = np.isfinite(depth) and depth > 0
                    measurements.append(dict(xy=[x,y], depth=float(depth), tolerance=float(eps),
                        valid3=bool(a['valid'][y,x]), valid6=bool(b['valid'][y,x]),
                        witnesses=int(a['witnesses'][y,x]), accumulation=float(w.sum()),
                        front_mass=float(w[ti<depth-eps].sum()) if measured else None,
                        band_mass=float(w[abs(ti-depth)<=eps].sum()) if measured else None,
                        back_mass=float(w[ti>depth+eps].sum()) if measured else None))
                row['centreline'] = measurements
                # Narrow strips follow the manually reviewed physical braid.
                strip = np.zeros(a['valid'].shape, bool)
                for (xa,ya),(xb,yb) in zip(points,points[1:]):
                    for y in range(ya,yb+1):
                        x = int(round(xa+(xb-xa)*(y-ya)/(yb-ya)))
                        strip[y,x-3:x+4] = True
                row['centre_strip'] = dict(pixels=int(strip.sum()),
                    measured_fraction=float((a['depth'][strip]>0).mean()),
                    valid3_fraction=float(a['valid'][strip].mean()),
                    valid6_fraction=float(b['valid'][strip].mean()),
                    witnesses_quantiles=np.quantile(a['witnesses'][strip],[0,.5,1]).tolist())
            rows.append(row)
    if field_fingerprint(model) != before:
        raise ValueError('Diagnostic changed field')
    sheet = Image.new('RGB',(max(p.width for _,p in panels),sum(p.height+30 for _,p in panels)), '#222')
    d=ImageDraw.Draw(sheet);top=0
    for cid,p in panels:
        d.text((2,top+4),f'cam{cid}: GT | native'+(' | dense' if compare_sampling else '')+
               (' | opaque dominant' if inspect_layers else ''),fill='white')
        sheet.paste(p,(0,top+25));top+=p.height+30
    sheet.resize((sheet.width*2,sheet.height*2)).save(output/'comparison.png')
    write(output/'metrics.json',dict(selection=str(selection),checkpoint=selected['checkpoint'],
        step=selected['step'],read_only=True,masks_used=False,field_sha256=before,
        probe_sha256=sha(Path(__file__)),
        depth_scope='Training cam097 only; cam095 is held out and has no PatchMatch map',
        band_scope='Raw measured depth audit; unqualified depths are not ground truth',rows=rows))
    print(json.dumps([dict(camera=r['camera'],metrics=r['metrics'],
                           centre_strip=r.get('centre_strip')) for r in rows]),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-sampling',action='store_true')
    p.add_argument('--inspect-layers',action='store_true')
    a=p.parse_args();probe(a.selection,a.output,a.compare_sampling,a.inspect_layers)
