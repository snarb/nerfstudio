"""Actual train-camera floor GT/render contacts and descriptive color residuals.

No fitting, training, GT compositing or geometry edits. Low-pass residual metrics
are descriptive; three projected views do not determine a physical BRDF.
"""
from pathlib import Path
import argparse,json,sys,copy
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def main(a):
    for p in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',ROOT/'native_png_linear_distortion_STAGED/code']:
        sys.path.insert(0,str(p))
    import torch,cv2
    from PIL import Image,ImageDraw
    from nerfstudio.utils.eval_utils import eval_setup
    from marshawn_release_common import authoritative_grid
    from build_marshawn_release_grid import witness_guard
    from probe_marshawn_occupancy import file_sha,digest_model
    stage=ROOT/'native_png_linear_distortion_STAGED';finish=json.loads((stage/'finished_receipt.json').read_text());checkpoint=Path(finish['checkpoint']);config=checkpoint.parent.parent/'config.yml';selpath=ROOT/'floor_ray_diagnosis/selection.json';selection=json.loads(selpath.read_text());chosen=[q for q in selection['queries']if q['id']in [0,2,3,6,8,9]]
    torch.set_num_threads(2)
    def configure(cfg):cfg.eval_checkpoint=checkpoint;return cfg
    cfg,pipe,loaded,step=eval_setup(config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);model=pipe.model;pipe.eval();saved=model.occupancy_grid.binaries.detach().clone();before=digest_model(model)
    folder=stage/'grids';stem=f'step-{step:09d}_grid';core,active,ema,boxes,policy=authoritative_grid(folder/(stem+'.npz'),folder/(stem+'.json'),checkpoint);points=np.load(ROOT/'geometry/train_witnesses.npz')['xyz'];_,_,mask,_=witness_guard(core,boxes,points)
    if model.config.render_step_size!=.003:raise ValueError('MatchselectedguardedHDstep.003')
    a.output.mkdir(exist_ok=False);rows=[];ds=pipe.datamanager.train_dataset
    def rgb_im(x):return Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8))
    try:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.as_tensor(mask,device=model.device))
        fixed=digest_model(model)
        for q in chosen:
            valid=[s for s in q['train_sources']if s['camera']in q['manual_floor_source_cameras']];contact=Image.new('RGB',(3*256,len(valid)*290),'black');draw=ImageDraw.Draw(contact)
            for row,s in enumerate(valid):
                name=s['camera'];ci=[Path(p).stem.split('_')[-1]for p in ds.image_filenames].index(name);u,v=s['uv'];x0=max(0,int(round(u))-64);y0=max(0,int(round(v))-64);x1=min(1920,int(round(u))+64);y1=min(1012,int(round(v))+64)
                camera=copy.deepcopy(ds.cameras[ci:ci+1]).to(model.device);camera.cx-=x0;camera.cy-=y0;camera.width.fill_(x1-x0);camera.height.fill_(y1-y0)
                with torch.inference_mode():out=model.get_outputs_for_camera(camera)
                arrays={k:v.detach().float().cpu().numpy()for k,v in out.items()if torch.is_tensor(v)};pred=arrays['rgb'];gt=ds.get_image_float32(ci).cpu().numpy()[y0:y1,x0:x1,:3];e=pred-gt;h,w=gt.shape[:2]
                luminance=gt@np.array([.2126,.7152,.0722]);gy,gx=np.gradient(luminance);validmask=np.zeros((h,w),bool);px,py=int(round(u))-x0,int(round(v))-y0;validmask[max(0,py-20):min(h,py+21),max(0,px-20):min(w,px+21)]=True
                validmask&=(luminance>.05)&(luminance<.95)&(np.hypot(gx,gy)<.015)
                low=cv2.GaussianBlur(e,(0,0),sigmaX=8,sigmaY=8);high=e-low
                stat=dict(query=q['id'],camera=name,bbox=[x0,y0,x1,y1],world_plane_target=q['plane_xyz'],target_uv=s['uv'],source_GT=str(ds.image_filenames[ci]),source_GT_sha256=file_sha(ds.image_filenames[ci]),valid_center_lowgradient_pixels=int(validmask.sum()),
                    GT_median_RGB=np.median(gt[validmask],axis=0).tolist(),pred_median_RGB=np.median(pred[validmask],axis=0).tolist(),residual_median_RGB=np.median(e[validmask],axis=0).tolist(),
                    center_RMSE=float(np.sqrt(np.mean(e[validmask]**2))),center_lowpass_RMSE=float(np.sqrt(np.mean(low[validmask]**2))),center_highpass_RMSE=float(np.sqrt(np.mean(high[validmask]**2))),
                    lowpass_energy_over_total=float(np.sum(low[validmask]**2)/max(np.sum(e[validmask]**2),1e-12)),median_opacity=float(np.median(arrays['accumulation'][validmask])))
                prefix=f'q{q["id"]:02d}_{name}';np.savez_compressed(a.output/(prefix+'_buffers.npz'),gt=gt,rgb=pred,opacity=arrays['accumulation'],depth=arrays['depth'],residual=e,valid_center_mask=validmask)
                for col,(value,label)in enumerate([(gt,'TRAIN GT'),(pred,'actualguarded8k'),(abs(e)*8,'8xabsRGBdiff')]):
                    image=rgb_im(value);image.save(a.output/(prefix+'_'+str(col)+'.png'));image=image.resize((w*2,h*2),Image.Resampling.NEAREST);contact.paste(image,(col*256,row*290+32));draw.text((col*256+4,row*290+4),f'q{q["id"]:02d} {name} {label}',fill='white')
                draw.text((2,row*290+18),f'GTmed={np.array(stat["GT_median_RGB"]).round(3)} residualmed={np.array(stat["residual_median_RGB"]).round(3)}',fill='yellow');rows.append(stat)
            contact.save(a.output/f'q{q["id"]:02d}_train_GT_render.jpg',quality=95);(a.output/'progress.json').write_text(json.dumps(dict(completed_queries=len({r['query']for r in rows}),total_queries=len(chosen)))+'\n')
        if digest_model(model)!=fixed:raise ValueError('Actualinference changedmodel')
    finally:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(saved)
        if digest_model(model)!=before:raise ValueError('Exactstate restorationfailed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);optimizer_LR={n:[g['lr']for g in z['param_groups']]for n,z in state['optimizers'].items()};del state
    result=dict(rows=rows,checkpoint_sha256=file_sha(checkpoint),config_sha256=file_sha(config),selection_sha256=file_sha(selpath),script_sha256=file_sha(__file__),optimizer_LR_at_selected_checkpoint=optimizer_LR,all_actual_train_cameras=True,no_GT_compositing=True,full_original_state_restored=True,training_updates=0,
        metric_scope='DescriptiveencodedSDRcenterpatchresiduals; lowgradientGTmaskforanalysisONLY, nottraining/rendercompositing',
        caveats=['Projectedapproximateplane identifiescorrespondingneighborhood, notexactdensecorrespondence','Genuinefloorwrinkles/specularcolorchangeswithview; 3viewsdonotidentifyBRDF','Gaussian8pxlowpassdescribesresidualscale, notproofcausalerror','NNsharesgeometry/colorfeatures; lowLRalonecannotguaranteegeometrypreservation'])
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(patches=len(rows),checkpoint_LR=optimizer_LR,output=str(a.output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
