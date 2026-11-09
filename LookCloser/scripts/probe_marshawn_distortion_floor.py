#!/usr/bin/env python3
"""No-opt coordinate-mode gradients on four vetted TRAIN floor pixels only."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for d in [ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',
          ROOT/'final_pair/code',ROOT/'final_pair/release_tools_v3',Path(__file__).parent]:
    sys.path.insert(0,str(d))
import numpy as np
import torch
from nerfstudio.utils.eval_utils import eval_setup
from probe_marshawn_occupancy import digest_model
from probe_marshawn_packed_distortion import grad_summary
from marshawn_packed_distortion_candidate import PackedDistortionCapture,load_frozen_formula,packet_terms,file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    run=ROOT/'final_pair/radius0/marshawn_001660_final_radius0/instant-ngp-bounded/cold_seed42';ck=run/'nerfstudio_models/step-000008000.ckpt'
    def cfg(c):c.eval_checkpoint=ck;return c
    _,pipe,loaded,step=eval_setup(run/'config.yml',test_mode='val',update_config_callback=cfg)
    assert loaded.resolve()==ck.resolve()and step==8000
    model=pipe.model;ds=pipe.datamanager.train_dataset
    by={Path(f).stem.split('_')[-1]:i for i,f in enumerate(ds.image_filenames)}
    selection=ROOT/'floor_ray_diagnosis/selection.json';queries=[q for q in json.loads(selection.read_text())['queries']if q['split']=='train']
    formula_source=ROOT/'code_sfm/nerfstudio/models/lookcloser.py';formula=load_frozen_formula(formula_source,file_sha(formula_source))
    named=[(n,p)for n,p in model.named_parameters()if p.requires_grad]
    before=digest_model(model);pyrng=random.getstate();nprng=np.random.get_state();trng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all();rows=[]
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(42);pipe.train()
            for q in queries:
                i=by[q['camera']];camera=ds.cameras[i:i+1].to(model.device)
                ray=camera.generate_rays(0,coords=torch.tensor([[q['y']+.5,q['x']+.5]],device=model.device),keep_shape=False)
                rgb=ds.get_data(i,image_type='float32')['image'][q['y'],q['x']][None].to(model.device)
                with PackedDistortionCapture(model)as capture:
                    with torch.autocast(device_type='cuda',enabled=True):
                        outputs=model(ray);loss=model.get_loss_dict(outputs,{'image':rgb})['rgb_loss']
                    terms=packet_terms(capture.packet,formula,.01)
                # One ray has ~2048x larger mean-loss derivative than the global
                # packet. Use the equivalent per-ray scale, not the batch scale.
                reference,grads=grad_summary(loss,named,32.)
                summaries={'RGB':reference}
                for mode in ['linear','inverse_depth']:
                    summaries[mode],_=grad_summary(terms[mode],named,32.,grads)
                row=dict(camera=q['camera'],x=q['x'],y=q['y'],split='train',inside_approx_support_hull=q['inside_15point_XY_hull'],
                    packed_samples=int(capture.packet['weights'].numel()),gradient_norms=summaries,
                    optimizer_updates=0,plane_used_as_loss_target=False,gradient_unscale=32.,
                    scale_note='65536/2048: equivalent per-ray AMP scale for this isolated single-ray diagnostic; production setting unchanged')
                rows.append(row);print(json.dumps(row),flush=True)
                del grads,terms,outputs,loss,capture
            pipe.eval()
    finally:
        random.setstate(pyrng);np.random.set_state(nprng)
    assert random.getstate()==pyrng
    n=np.random.get_state();assert n[0]==nprng[0]and np.array_equal(n[1],nprng[1])and n[2:]==nprng[2:]
    assert torch.equal(trng,torch.get_rng_state())and all(torch.equal(x,y)for x,y in zip(crng,torch.cuda.get_rng_state_all()))
    assert digest_model(model)==before
    r=dict(passed=True,checkpoint=str(ck),checkpoint_sha256=file_sha(ck),source_sha256=file_sha(__file__),
        floor_selection_sha256=file_sha(selection),queries=len(rows),train_only=True,no_plane_supervision=True,
        Python_NumPy_TorchCPU_CUDA_RNG_restored=True,field_grid_unchanged=True,optimizer_updates=0,rows=rows,
        interpretation='Four selected floor-ray gradients are diagnostics, not unbiased full-image averages or evidence of quality improvement.')
    (a.output/'actual_floor_probe.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(passed=True,queries=len(rows))),flush=True)


if __name__=='__main__':main()
