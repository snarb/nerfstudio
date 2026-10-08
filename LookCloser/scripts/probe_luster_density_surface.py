#!/usr/bin/env python3
"""Read-only hard density-level face crops from an existing volume field."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from export_luster_opacity import load
from run_luster_opacity import score, sha, write


def hard_hits(field, rays, packed, threshold, refinements=12, color_field=None):
    indices=packed['packed_ray_indices'].reshape(-1).long()
    near=rays.nears.flatten(); far=rays.fars.flatten()
    starts=packed['packed_spacing_starts'].flatten()
    ends=packed['packed_spacing_ends'].flatten()
    mids=near[indices]+(far-near)[indices]*(starts+ends)*.5
    count=len(rays.origins)
    colors=rays.origins.new_zeros(count,3)
    depths=rays.origins.new_zeros(count)
    if not len(indices):
        return colors, depths, dict(hit_fraction=0.,bracket_fraction=0.,root_residual_p90=None)
    positions=rays.origins[indices]+rays.directions[indices]*mids[:,None]
    density,_=field.query_points(positions,rays.directions[indices])
    density=density.flatten().float()
    sample_indices=torch.arange(len(indices),device=indices.device)
    first=torch.full((count,),len(indices),dtype=torch.long,device=indices.device)
    first.scatter_reduce_(0,indices,torch.where(density>=threshold,sample_indices,len(indices)),reduce='amin')
    hit=first<len(indices)
    ray_ids=torch.nonzero(hit).flatten()
    first=first[hit]
    previous=(first-1).clamp_min(0)
    same_ray=(first>0)&(indices[previous]==ray_ids)
    low=torch.where(same_ray,mids[previous],near[ray_ids])
    high=mids[first]
    origins=rays.origins[ray_ids]; directions=rays.directions[ray_ids]
    before,_=field.query_points(origins+directions*low[:,None],directions)
    bracket=before.flatten()<threshold
    # An already occupied near bound stays a hard boundary; it is separately
    # reported rather than silently called a converged level-set root.
    high=torch.where(bracket,high,low)
    for _ in range(refinements):
        mid=(low+high)*.5
        values,_=field.query_points(origins+directions*mid[:,None],directions)
        inside=values.flatten()>=threshold
        high=torch.where(inside,mid,high)
        low=torch.where(inside,low,mid)
    root=(low+high)*.5
    values,rgb=field.query_points(origins+directions*root[:,None],directions)
    if color_field is not None:
        _,rgb=color_field.query_points(origins+directions*root[:,None],directions)
    colors[ray_ids]=rgb.float(); depths[ray_ids]=root
    residual=(values.flatten().float()-threshold).abs()/threshold
    return colors,depths,dict(hit_fraction=float(hit.float().mean()),
        bracket_fraction=float(bracket.float().mean()) if len(bracket) else 0.,
        root_residual_p90=float(torch.quantile(residual,.9)) if len(residual) else None,
        root_residual_below_1pct_fraction=float((residual<=.01).float().mean()) if len(residual) else 0.)


def probe(selection, output, thresholds=(100.,300.,1000.), chunk=1024, field_override=None, color_override=None):
    if output.exists():
        raise FileExistsError('Preserve previous diagnostic renders')
    if chunk<1 or any(not np.isfinite(v) or v<=0 for v in thresholds):
        raise ValueError('Positive chunk size and finite positive density levels required')
    selected=json.loads(selection.read_text());selected=selected.get('selected',selected)
    trainer,request=load(selected);model=trainer.pipeline.model
    if getattr(model.config,'representation','volume')!='volume':
        raise ValueError('Require a learned volume field')
    if model.config.appearance_embedding_dim!=0:
        raise ValueError('Point-color diagnostic requires no appearance embedding')
    if field_override is not None:
        model.field.load_state_dict(torch.load(field_override,map_location=model.device,weights_only=True),strict=True)
    color_field=None
    if color_override is not None:
        if field_override is not None:
            raise ValueError('Choose either a complete field override or an independent color field')
        color_trainer,_=load(selected)
        color_field=color_trainer.pipeline.model.field
        color_field.load_state_dict(torch.load(color_override,map_location=model.device,weights_only=True),strict=True)
    output.mkdir(parents=True)
    sheet=Image.new('RGB',(250*(2+len(thresholds)),225*6),'#222222');draw=ImageDraw.Draw(sheet)
    records=[];row=0
    with torch.no_grad():
        for split,dataset in [('eval',trainer.pipeline.datamanager.eval_dataset),('train',trainer.pipeline.datamanager.train_dataset)]:
            wanted=list(range(len(dataset))) if split=='eval' else [i for i,p in enumerate(dataset.image_filenames) if int(p.stem.split('_')[1]) in [11,97,151]]
            for index in wanted:
                name=dataset.image_filenames[index].name
                x0,y0,x1,y1=request['roi_boxes'][name]['face']
                gt=dataset[index]['image'][y0:y1,x0:x1].to(model.device)
                height,width=gt.shape[:2]
                yy,xx=torch.meshgrid(torch.arange(y0,y1,device=model.device),torch.arange(x0,x1,device=model.device),indexing='ij')
                camera=dataset.cameras[index:index+1].to(model.device)
                rays=camera.generate_rays(0,coords=torch.stack([yy,xx],-1).reshape(-1,2).float()+.5)
                native=[];results={level:[] for level in thresholds};diagnostics={level:[] for level in thresholds}
                for offset in range(0,len(rays.origins),chunk):
                    batch=rays[offset:offset+chunk]
                    packed=model(batch)
                    native.append(packed['rgb'].float())
                    for level in thresholds:
                        rgb,_,stats=hard_hits(model.field,batch,packed,level,color_field=color_field)
                        results[level].append(rgb)
                        diagnostics[level].append(dict(rays=len(batch.origins),**stats))
                images=[gt,torch.cat(native).reshape(height,width,3)]
                entries=[dict(kind='adapted_volume_diagnostic' if field_override else 'volume',
                              **score(model,images[-1],gt))]
                for level in thresholds:
                    rgb=torch.cat(results[level]).reshape(height,width,3);images.append(rgb)
                    stats=diagnostics[level];total=sum(s['rays'] for s in stats)
                    entry=dict(kind='hard_density_level',threshold=level,**score(model,rgb,gt),
                               hit_fraction=sum(s['rays']*s['hit_fraction'] for s in stats)/total,
                               bracket_fraction_weighted_by_batch=sum(s['rays']*s['bracket_fraction'] for s in stats)/total,
                               batches=stats)
                    entries.append(entry)
                    panel=torch.cat([gt,rgb],1).cpu().numpy()
                    Image.fromarray(np.rint(panel.clip(0,1)*255).astype('uint8')).save(output/f'{split}_{index:03d}_face_tau_{level:g}.png')
                for column,img in enumerate(images):
                    im=Image.fromarray(np.rint(img.cpu().numpy().clip(0,1)*255).astype('uint8'));im.thumbnail((245,195))
                    sheet.paste(im,(column*250+3,row*225+25))
                label='adapted volume' if field_override else 'volume'
                draw.text((3,row*225+4),name+f' GT | {label} | '+ ' | '.join(f'level{v:g}' for v in thresholds),fill='white')
                records.append(dict(split=split,index=index,image=name,box=[x0,y0,x1,y1],results=entries))
                row+=1;sheet.save(output/'face.jpg');print(json.dumps(dict(image=name,results=[{k:v for k,v in e.items() if k!='batches'} for e in entries])),flush=True)
    write(output/'manifest.json',dict(selected=selected,checkpoint_sha256=sha(selected['checkpoint']),
        transforms_sha256=sha(Path(request['data'])/'transforms.json'),thresholds=list(thresholds),
        feature_reweighting_strength=model.field.feature_reweighting_strength,source_sha256=sha(Path(__file__)),
        field_override=str(field_override) if field_override else None,
        field_override_sha256=sha(field_override) if field_override else None,
        color_override=str(color_override) if color_override else None,
        color_override_sha256=sha(color_override) if color_override else None,
        masks_used=False,depth_prior_used=False,read_only=True,visual_review='pending',records=records,
        scope='Native face crops only; first sampled density-level boundaries with bisection. Frequency-cell discontinuities and missed narrow peaks are not certified surfaces; not a trained opaque model.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--thresholds',type=float,nargs='+',default=[100.,300.,1000.])
    parser.add_argument('--chunk',type=int,default=1024)
    parser.add_argument('--field-override',type=Path)
    parser.add_argument('--color-override',type=Path)
    args=parser.parse_args();probe(args.selection,args.output,args.thresholds,args.chunk,args.field_override,args.color_override)
