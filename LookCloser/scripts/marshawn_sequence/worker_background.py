"""Declared fresh ablation: weak empty-ray prior on conservative TRAIN background.

The sealed worker/runtime and every RGB/foreground/head label stay unchanged.
This additional source file is explicitly included in the immutable run binding.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from common import read,atomic,sha
import worker


def trusted_background(semantic,unknown):
    actor=semantic['black']|semantic['grey']
    protected=maximum_filter(actor,size=129,mode='constant',cval=0)
    mask=semantic['valid']&~protected&~unknown&(semantic['soft_alpha']==0)
    mask[:64]=False;mask[-64:]=False;mask[:,:64]=False;mask[:,-64:]=False
    return mask


def install(pipeline,spec):
    original_install(pipeline,spec)
    strength=spec.get('background_opacity_prior')
    if strength not in (.001,.01,.1):raise ValueError('Explicit background prior strength required')
    root=Path(spec['data']);folder=Path(spec['output'])/'background_prior';folder.mkdir(exist_ok=True)
    names=[Path(p).stem for p in pipeline.datamanager.train_dataset.image_filenames]
    ready=read(root/'ready.json');expected={r['camera']:r for r in ready['records']}
    masks=[];widths=[];sizes=[];records=[]
    for name in names:
        semantic=root/'semantics'/(name+'.npz');part=root/'parts'/(name+'.npz')
        if sha(semantic)!=expected[name]['semantic_sha256'] or sha(part)!=read(root/'parts'/(name+'.json'))['archive_sha256']:raise ValueError('Background prior source changed')
        with np.load(semantic) as s,np.load(part) as p:
            background=trusted_background(s,p['unknown_people']|p['crew_exclusion'])
            if np.any(background & (s['black']|s['grey'])):raise ValueError('Background prior intersects foreground')
        masks.append(background.reshape(-1));widths.append(background.shape[1]);sizes.append(background.size)
        records.append(dict(camera=name,semantic_sha256=sha(semantic),parts_sha256=sha(part),background_pixels=int(background.sum())))
    masks=np.concatenate(masks);offsets=np.r_[0,np.cumsum(sizes)[:-1]];widths=np.asarray(widths)
    atomic(folder/'manifest.json',dict(strength=strength,train_cameras=names,ready_sha256=sha(root/'ready.json'),records=records,rule='valid, production soft alpha exactly0, no foreground within64 native pixels in either axis, no unknown/crew person,64px image-edge exclusion',not_physical_alpha_ground_truth=True,warmup=4096,ramp_end=12288))
    model=pipeline.model;original=model.get_loss_dict
    def loss(outputs,batch,metrics_dict=None):
        result=original(outputs,batch,metrics_dict)
        if model.training:
            c,y,x=batch['indices'].detach().cpu().long().numpy().T
            flags=torch.from_numpy(masks[offsets[c]+y*widths[c]+x]).to(outputs['accumulation'].device)
            opacity=outputs['accumulation'].float().reshape(-1)
            ramp=min(1.,max(0.,(pipeline.marshawn_sampling_adapter.step-4096)/8192))
            result['clear_background_opacity_loss']=strength*ramp*torch.where(flags,opacity,0).square().sum()/flags.sum().clamp_min(1)
        return result
    model.get_loss_dict=loss


original_install=worker.install_core

if __name__=='__main__':
    worker.install_core=install
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--technical-steps',type=int);a=p.parse_args();worker.train(a.spec,a.technical_steps)
