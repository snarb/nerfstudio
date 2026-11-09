"""Actual native floor crops at two inference steps, same field/candidate guard."""
from pathlib import Path
import argparse,json,sys,copy
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def main(a):
    for p in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',ROOT/'native_png_linear_distortion_STAGED/code']:
        sys.path.insert(0,str(p))
    import torch
    from PIL import Image,ImageDraw
    from nerfstudio.utils.eval_utils import eval_setup
    from marshawn_release_common import authoritative_grid
    from build_marshawn_release_grid import witness_guard
    from probe_marshawn_occupancy import file_sha,digest_model
    source=json.loads((a.root/'DISTORTION/audit.json').read_text());checkpoint=Path(source['checkpoint']);config=ROOT/'native_png_linear_distortion_STAGED/run/marshawn_001660_png_distortion_linear001/instant-ngp-bounded/cold_seed42/config.yml'
    torch.set_num_threads(2)
    def configure(cfg):cfg.eval_checkpoint=checkpoint;return cfg
    cfg,pipe,loaded,step=eval_setup(config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);model=pipe.model;pipe.eval();before=digest_model(model)
    saved=model.occupancy_grid.binaries.detach().clone();savedstep=model.config.render_step_size;folder=ROOT/'native_png_linear_distortion_STAGED/grids';stem=f'step-{step:09d}_grid'
    core,active,ema,boxes,policy=authoritative_grid(folder/(stem+'.npz'),folder/(stem+'.json'),checkpoint);points=np.load(ROOT/'geometry/train_witnesses.npz')['xyz'];_,_,candidate,_=witness_guard(core,boxes,points)
    z=np.load(a.root/'DISTORTION/witness_union_halo1_candidate/grid.npz')
    if not np.array_equal(candidate,z['active'])or not np.array_equal(saved.cpu().numpy(),active):raise ValueError('Candidate/source changed')
    a.output.mkdir(exist_ok=False);rows=[];bbox=[850,650,1450,980];x0,y0,x1,y1=bbox
    def im(value):return Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8))
    try:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.as_tensor(candidate,device=model.device))
        fixed=digest_model(model)
        ds=pipe.datamanager.eval_dataset
        for name in ['HC','DC','KA']:
            ci=[Path(p).stem.split('_')[-1]for p in ds.image_filenames].index(name);cam=copy.deepcopy(ds.cameras[ci:ci+1]).to(model.device);cam.cx-=x0;cam.cy-=y0;cam.width.fill_(x1-x0);cam.height.fill_(y1-y0)
            gt=ds.get_image_float32(ci).cpu().numpy()[y0:y1,x0:x1,:3];outputs={}
            for dt in [.003,.000375]:
                model.config.render_step_size=dt
                with torch.inference_mode():out=model.get_outputs_for_camera(cam)
                arrays={k:v.detach().float().cpu().numpy()for k,v in out.items()if torch.is_tensor(v)};outputs[str(dt)]=arrays
                prefix=name+'_dt'+str(dt).replace('.','p');im(arrays['rgb']).save(a.output/(prefix+'.png'));np.savez_compressed(a.output/(prefix+'_buffers.npz'),**arrays)
            old=outputs['0.003']['rgb'];new=outputs['0.000375']['rgb'];delta=abs(new-old);panel=Image.new('RGB',(4*(x1-x0),y1-y0+25),'black');draw=ImageDraw.Draw(panel)
            for j,(value,label)in enumerate([(gt,'GT fixedscalarPNG'),(old,'candidate dt.003'),(new,'candidate dt.000375'),(delta*16,'16x absoluteRGBdifference')]):panel.paste(im(value),(j*(x1-x0),25));draw.text((j*(x1-x0)+5,5),name+' '+label,fill='white')
            panel.save(a.output/(name+'_comparison.jpg'),quality=95)
            row=dict(camera=name,bbox=bbox,max_RGB_delta=float(delta.max()),mean_RGB_delta=float(delta.mean()),p99_RGB_delta=float(np.quantile(delta,.99)),fraction_channels_changed_over1code=float((delta>1/255).mean()),coarse_mean_abs_GT_error=float(abs(old-gt).mean()),fine_mean_abs_GT_error=float(abs(new-gt).mean()))
            rows.append(row);(a.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=3))+'\n');print(json.dumps(row),flush=True)
        if digest_model(model)!=fixed:raise ValueError('Modelchanged duringrender')
    finally:
        model.config.render_step_size=savedstep
        with torch.no_grad():model.occupancy_grid.binaries.copy_(saved)
        if digest_model(model)!=before or model.config.render_step_size!=savedstep:raise ValueError('Originalstate/configrestorationfailed')
    result=dict(checkpoint_sha256=file_sha(checkpoint),source_selection_sha256=source['selection_sha256'],candidate_sha256=file_sha(a.root/'DISTORTION/witness_union_halo1_candidate/grid.npz'),script_sha256=file_sha(__file__),native_API='model.get_outputs_for_camera',steps=[.003,.000375],rows=rows,no_GT_compositing=True,original_state_and_config_restored=True,optimizer_updates=0)
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
