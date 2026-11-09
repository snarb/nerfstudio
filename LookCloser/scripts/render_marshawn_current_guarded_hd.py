"""Provisional full-HD guarded prediction download; same frozen NN/EMA/config.

Compare inference step .00075 to .003 on the same protected mask. Preserve both
variants; select fine only if PSNR/SSIM/LPIPS do not worsen versus guarded coarse.
No training, GT compositing, checkpoint/config rewrite or old ZIP replacement.
"""
from pathlib import Path
import argparse,json,sys,time,shutil
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def main(a):
    for p in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',ROOT/'native_png_linear_distortion_STAGED/code']:
        sys.path.insert(0,str(p))
    import torch
    from PIL import Image,ImageDraw
    from nerfstudio.utils.eval_utils import eval_setup
    from marshawn_release_common import authoritative_grid,native_bounds
    from build_marshawn_release_grid import witness_guard
    from probe_marshawn_occupancy import file_sha,digest_model
    stage=ROOT/'native_png_linear_distortion_STAGED';finished=json.loads((stage/'finished_receipt.json').read_text());checkpoint=Path(finished['checkpoint']);config=checkpoint.parent.parent/'config.yml'
    if file_sha(config)!=finished['configuration_sha256']or file_sha(checkpoint)!=finished['checkpoint_sha256']:raise ValueError('Sealed selectedsource changed')
    torch.set_num_threads(2)
    def configure(cfg):cfg.eval_checkpoint=checkpoint;return cfg
    cfg,pipe,loaded,step=eval_setup(config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);model=pipe.model;pipe.eval();bounds=native_bounds(model)
    before=digest_model(model);saved=model.occupancy_grid.binaries.detach().clone();savedstep=model.config.render_step_size
    if savedstep!=.003 or model.config.background_color!='black':raise ValueError('Expectedexactsource.003/blackbackground')
    stem=f'step-{step:09d}_grid';folder=stage/'grids';core,active,ema,boxes,policy=authoritative_grid(folder/(stem+'.npz'),folder/(stem+'.json'),checkpoint)
    points=np.load(ROOT/'geometry/train_witnesses.npz')['xyz'];_,_,protected,_=witness_guard(core,boxes,points)
    candidate=ROOT/'floor_ray_diagnosis/DISTORTION/witness_union_halo1_candidate';record=json.loads((candidate/'receipt.json').read_text());z=np.load(candidate/'grid.npz')
    if not np.array_equal(saved.cpu().numpy(),active)or not np.array_equal(z['active'],protected)or file_sha(candidate/'grid.npz')!=record['grid_archive_sha256']:raise ValueError('Protectedsourcecandidate changed')
    a.output.mkdir(parents=True,exist_ok=False);rows=[];data=pipe.datamanager.eval_dataset;review=a.output/'review';review.mkdir()
    try:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.as_tensor(protected,device=model.device))
        fixed=digest_model(model)
        for dt in [.00075,.003]:
            model.config.render_step_size=dt;variant=a.output/'variants'/('dt'+str(dt).replace('.','p'));variant.mkdir(parents=True);(variant/'raw').mkdir();local=[]
            for name in ['HC','DC','KA']:
                ci=[Path(p).stem.split('_')[-1]for p in data.image_filenames].index(name);camera=data.cameras[ci:ci+1].to(model.device)
                if int(camera.width.item())!=1920 or int(camera.height.item())!=1012:raise ValueError('RequirednativeHD1920x1012')
                started=time.monotonic()
                with torch.inference_mode():
                    out=model.get_outputs_for_camera(camera);gt=data.get_image_float32(ci).to(model.device);metrics,_=model.get_image_metrics_and_images(out,dict(image=gt))
                arrays={k:v.detach().float().cpu().numpy()for k,v in out.items()if torch.is_tensor(v)};rgb=arrays['rgb'];opacity=arrays['accumulation']
                if rgb.shape!=(1012,1920,3)or not all(np.isfinite(v).all()for v in arrays.values()):raise ValueError('InvalidactualHDoutput')
                Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(variant/(name+'.png'))
                np.savez_compressed(variant/'raw'/(name+'.npz'),rgb=rgb,rgba=np.concatenate([rgb,opacity],-1),opacity=opacity,depth=arrays['depth'],samples=arrays['num_samples_per_ray'])
                # Separate review contact only; downloadable PNG is prediction-only.
                pred=Image.open(variant/(name+'.png'));truth=Image.fromarray(np.rint(np.clip(gt.cpu().numpy(),0,1)*255).astype(np.uint8));pred.thumbnail((640,337));truth.thumbnail((640,337));contact=Image.new('RGB',(1280,362),'black');draw=ImageDraw.Draw(contact);contact.paste(truth,(0,25));contact.paste(pred,(640,25));draw.text((4,5),name+' GT',fill='white');draw.text((644,5),f'{name} guarded8k dt{dt}',fill='white');contact.save(review/(name+'_dt'+str(dt).replace('.','p')+'.jpg'),quality=94)
                row=dict(camera=name,dt=dt,metrics=metrics,seconds=time.monotonic()-started,png_sha256=file_sha(variant/(name+'.png')),raw_sha256=file_sha(variant/'raw'/(name+'.npz')),source_GT=str(data.image_filenames[ci]),source_GT_sha256=file_sha(data.image_filenames[ci]))
                rows.append(row);local.append(row);(a.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=6,last=row),indent=2)+'\n');print(json.dumps(row),flush=True)
            (variant/'metrics.json').write_text(json.dumps(dict(rows=local,mean={k:float(np.mean([r['metrics'][k]for r in local]))for k in ['psnr','ssim','lpips']}),indent=2)+'\n')
        if digest_model(model)!=fixed:raise ValueError('Protectedinference changed NN/EMA/grid')
    finally:
        model.config.render_step_size=savedstep
        with torch.no_grad():model.occupancy_grid.binaries.copy_(saved)
        if digest_model(model)!=before or model.config.render_step_size!=savedstep:raise ValueError('Originalmodel/configrestorationfailed')
    if file_sha(config)!=finished['configuration_sha256']or file_sha(checkpoint)!=finished['checkpoint_sha256']:raise ValueError('Sourcefile changed')
    means={str(dt):{k:float(np.mean([r['metrics'][k]for r in rows if r['dt']==dt]))for k in ['psnr','ssim','lpips']}for dt in [.00075,.003]}
    fine,coarse=means['0.00075'],means['0.003'];fine_pass=fine['psnr']>=coarse['psnr']-1e-6 and fine['ssim']>=coarse['ssim']-1e-6 and fine['lpips']<=coarse['lpips']+1e-6
    selected=.00075 if fine_pass else .003;variant=a.output/'variants'/('dt'+str(selected).replace('.','p'));(a.output/'raw').mkdir()
    for name in ['HC','DC','KA']:
        shutil.copyfile(variant/(name+'.png'),a.output/(name+'.png'));shutil.copyfile(variant/'raw'/(name+'.npz'),a.output/'raw'/(name+'.npz'))
    result=dict(status='PROVISIONAL_CURRENTBEST_PENDING_ROOT_VISUAL_REVIEW',selected_step=8000,selected_inference_dt=selected,
        fine_step_metrics_nonworsening_vs_guarded_coarse=fine_pass,selection_rule='Fine only if all3meanPSNR/SSIM/LPIPS do not worsen vs sameguard.003; else guarded.003',
        checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),config_sha256=file_sha(config),script_sha256=file_sha(__file__),
        protected_policy='ActualexportedGPUcore UNION9022 observedpointcells THEN exactlyONEcubehalo',candidate_grid_sha256=file_sha(candidate/'grid.npz'),candidate_receipt_sha256=file_sha(candidate/'receipt.json'),
        source_default_metrics=finished['metrics'],guarded_metrics=means,rows=rows,shape=[1012,1920,3],live_bounds=bounds,
        source_GT_metric_space='UnmodifiedpreparedfixedscalarSDRPNG, nativeall3HDPSNR/SSIM/LPIPS',
        raw_RGBA_contract='Actualmodel RGB (black-background premultiplied radiance codes) concatenatedwithopacity; notstraight/unassociatedRGBA. No GT inrawNPZ.',
        output_sha256={name+'.png':file_sha(a.output/(name+'.png'))for name in ['HC','DC','KA']},
        full_original_state_and_config_restored=True,optimizer_updates=0,old_ZIP_modified=False,
        limitations=['Provisionalrender; remainingfloor/matpattern/softnessnotdeclaredclean.','Pointcoverageisnotindependentphysicalvolumevalidation.','All3metricsareaggregate; actualimagesrequiremanualreview.'])
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(selected_dt=selected,guarded_metrics=means,output=str(a.output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
