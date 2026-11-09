"""Serving adapter for a sequence field plus its frozen background and FR phase.

Bare parameter loading omits both external frozen background and non-tensor
feature reweighting. This adapter restores them without training or GPU optimizer
allocation, and permits moving a complete frame directory to another host.
"""
from pathlib import Path
import torch,yaml
from run_dec5_sequence import read
from dec5_sequence_worker import sha
from dec5_checkpoint_runtime import restore
from dec5_background_separation import attach_inference


def load(training,data=None,checkpoint=None,device='cuda',eval_chunk=8192):
    training=Path(training);request=read(training/'request.json');best=read(training/'best.json')
    checkpoint=Path(checkpoint or best['path'])
    data=Path(data or training.parent/'data')
    if sha(data/'transforms.json')!=request['data_sha256']:raise ValueError('Serving dataset differs from training')
    digest=sha(checkpoint)
    if best and checkpoint==Path(best['path']) and digest!=best['sha256']:raise ValueError('Selected checkpoint checksum mismatch')
    configs=list(training.rglob('config.yml'))
    if len(configs)!=1:raise ValueError('One native training config required')
    cfg=yaml.load(configs[0].read_text(),Loader=yaml.Loader)
    cfg.pipeline.datamanager.data=data;cfg.pipeline.datamanager.dataparser.data=data
    cfg.pipeline.datamanager.cache_images_type='uint8';cfg.pipeline.model.eval_num_rays_per_chunk=eval_chunk
    pipe=cfg.pipeline.setup(device=device,test_mode='val')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    pipe.load_pipeline(state['pipeline'],state['step']);pipe.model.current_train_step=state['step'];pipe.eval()
    strength=restore(pipe.model,state,request)
    dependency=read(training/'background_dependency.json')
    candidates=[Path(dependency['path']),data.parent/'ray_background/background.pt']
    background=next((p for p in candidates if p.is_file() and sha(p)==dependency['sha256']),None)
    if background is None:raise ValueError('Missing hash-matched frozen background')
    volume=request.get('recipe',{}).get('background_volume')
    provider=attach_inference(pipe,background,volume)
    provenance=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,step=state['step'],
        data_sha256=request['data_sha256'],background_sha256=dependency['sha256'],
        feature_reweighting_strength=strength,eval_rays_per_chunk=eval_chunk,optimizer_restored=False,
        background_volume=volume)
    del state
    return pipe,provider,provenance
