#!/usr/bin/env python3
"""Short RGB-only diagnostic on fixed hard density boundaries, never a volume export."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from export_luster_opacity import load
from probe_luster_density_surface import hard_hits, probe
from run_luster_opacity import sha, write


def geometry_hash(field):
    digest=hashlib.sha256()
    for name,value in field.state_dict().items():
        if name.startswith('mlp_color.'):
            continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit(selection,output,threshold=1000.,steps=1000,batch=1024,train_encoding=False):
    if output.exists():
        raise FileExistsError('Use a new diagnostic output')
    if steps<1 or batch<1 or threshold<=0:
        raise ValueError('Positive steps, batch and threshold required')
    selected=json.loads(selection.read_text());selected=selected.get('selected',selected)
    trainer,request=load(selected);model=trainer.pipeline.model;field=model.field
    color_field=field
    if train_encoding:
        color_trainer,_=load(selected)
        color_field=color_trainer.pipeline.model.field
    if getattr(model.config,'representation','volume')!='volume':
        raise ValueError('Only a volume teacher can initialize this diagnostic')
    output.mkdir(parents=True)
    write(output/'request.json',dict(selection=str(selection),selected=selected,
        threshold=threshold,steps=steps,batch=batch,train_encoding=train_encoding,source_sha256=sha(Path(__file__)),
        scope='RGB-only hard-boundary diagnostic; fresh color Adam; geometry frozen; all RGB targets retained'))
    source_hash=geometry_hash(field)
    for name,p in field.named_parameters():
        p.requires_grad_(not train_encoding and name.startswith('mlp_color.'))
    if train_encoding:
        for p in color_field.parameters():p.requires_grad_(True)
    # Separate RGB optimization is intentional and recorded: this tests whether
    # copied color can fit the fixed opaque boundary, not a full recipe run.
    parameters=color_field.parameters() if train_encoding else field.mlp_color.parameters()
    optimizer=torch.optim.Adam(parameters,lr=.001)
    scaler=torch.amp.GradScaler('cuda')
    manager=trainer.pipeline.datamanager
    manager.config.train_num_rays_per_batch=batch
    if hasattr(manager,'train_pixel_sampler'):
        manager.train_pixel_sampler.set_num_rays_per_batch(batch)
    start=time.time()
    for step in range(steps):
        rays,target=manager.next_train(selected['step']+step+1)
        target_rgb=target['image'].to(model.device)
        model.eval()
        with torch.no_grad():
            packed=model(rays)
            _,depth,_=hard_hits(field,rays,packed,threshold)
            hit=depth>0
            positions=rays.origins[hit]+rays.directions[hit]*depth[hit,None]
        if hit.any():
            _,rgb=color_field.query_points(positions,rays.directions[hit])
            prediction=torch.zeros_like(target_rgb).index_copy(0,torch.nonzero(hit).flatten(),rgb.float())
            objective=(prediction-target_rgb).square().mean()
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite RGB adaptation')
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(objective).backward();scaler.step(optimizer);scaler.update()
        if step%100==0 or step==steps-1:
            row=dict(step=step,elapsed=time.time()-start,hit_fraction=float(hit.float().mean()),
                     scaler=float(scaler.get_scale()),gpu_bytes=torch.cuda.memory_allocated())
            write(output/'progress.json',row);print(json.dumps(row),flush=True)
    final_hash=geometry_hash(field)
    if final_hash!=source_hash:
        raise ValueError('RGB-only adaptation changed geometry or frequency-grid state')
    archive=output/'rgb_only_field.pt'
    torch.save(color_field.state_dict(),archive)
    write(output/'manifest.json',dict(selected=selected,source_checkpoint_sha256=sha(selected['checkpoint']),
        field_file=str(archive),field_sha256=sha(archive),threshold=threshold,steps=steps,batch=batch,
        geometry_sha256_before=source_hash,geometry_sha256_after=final_hash,
        feature_reweighting_strength=field.feature_reweighting_strength,
        train_encoding=train_encoding,
        optimizer='Fresh Adam on independent color field, lr.001' if train_encoding else 'Fresh Adam on mlp_color only, lr.001',source_sha256=sha(Path(__file__)),
        scope='Fixed-geometry RGB adaptation diagnostic. Render only with the hard-density probe; not a standard volume or trained geometry checkpoint.',
        masks_used=False,depth_prior_used=False))
    # The probe explicitly restores this field over the source teacher and
    # evaluates the hard renderer. It must not silently evaluate volume RGB.
    probe(selection,output/'review',[threshold],chunk=1024,
          field_override=None if train_encoding else archive,color_override=archive if train_encoding else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=1000.)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch',type=int,default=1024)
    p.add_argument('--train-encoding',action='store_true',help='Fit a separate complete color field; geometry teacher stays frozen')
    a=p.parse_args();fit(a.selection,a.output,a.threshold,a.steps,a.batch,a.train_encoding)
