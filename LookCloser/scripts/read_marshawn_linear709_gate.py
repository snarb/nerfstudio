"""Exact immutable-checkpoint linear709 display reader, with live state proof.

Reuses the frozen generic reader; this wrapper never launches/resumes training.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(a):
    import torch
    from nerfstudio.utils import eval_utils
    from probe_marshawn_occupancy import digest_model
    import render_marshawn_previews as preview
    if sha(a.checkpoint)!=a.checkpoint_sha256:
        raise ValueError('Immutable checkpoint SHA mismatch')
    if sha(preview.__file__)!=a.generic_sha256:
        raise ValueError('Frozen generic reader changed')
    original=eval_utils.eval_setup;captured={}
    def setup(*args,**kwargs):
        result=original(*args,**kwargs);pipe=result[1];model=pipe.model
        if model.renderer_rgb.clamp_output or type(model).__name__!='BoundedLinear709Model':
            raise ValueError('Expected unclipped physical linear709 native model')
        bounds=model.scene_aabb.reshape(2,3)
        for b in [model.field.aabb,model.occupancy_grid.aabbs[0].reshape(2,3),
                  pipe.datamanager.train_dataparser_outputs.scene_box.aabb,
                  pipe.datamanager.eval_dataset._dataparser_outputs.scene_box.aabb]:
            torch.testing.assert_close(b,bounds.to(b),rtol=0,atol=1e-6)
        captured.update(pipe=pipe,model_hash=digest_model(model),
                        grid={k:v.detach().cpu().clone()for k,v in model.occupancy_grid.state_dict().items()})
        return result
    eval_utils.eval_setup=setup
    try:
        preview.run(SimpleNamespace(config=a.config,checkpoint=a.checkpoint,output=a.output,
            code_root=a.code_root,data=None,max_width=a.max_width,chunk_size=a.chunk_size,
            camera=None,bbox=None,skip_strict_bounds=True,display_transform='linear-reinhard-srgb'))
    finally:
        eval_utils.eval_setup=original
    model=captured['pipe'].model;after=digest_model(model)
    equal_grid=all(torch.equal(v.detach().cpu(),captured['grid'][k])for k,v in model.occupancy_grid.state_dict().items())
    if after!=captured['model_hash']or not equal_grid or sha(a.checkpoint)!=a.checkpoint_sha256:
        raise ValueError('Read-only render changed live model/grid or checkpoint')
    receipt=dict(checkpoint_sha256=a.checkpoint_sha256,model_state_sha256_before=captured['model_hash'],
        model_state_sha256_after=after,field_and_all_persistent_model_state_unchanged=True,
        grid_EMA_active_AABB_resolution_unchanged=True,checkpoint_file_unchanged=True,
        display_transform='Physical linear709 -> Reinhard -> sRGB ONCE, no extra exposure',
        raw_buffers='Physical linear709 RGB+GT, unchanged floats',source_sha256=sha(__file__),
        generic_sha256=a.generic_sha256,optimizer_updates=0,training_callbacks=0,no_CONT_sent=True)
    (a.output/'live_state_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','checkpoint','output','code-root']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--generic-sha256',required=True)
    p.add_argument('--max-width',type=int,default=1920);p.add_argument('--chunk-size',type=int,default=512)
    run(p.parse_args())
