"""Read-only native HD render using checkpoint-bound witness-union ONE-halo."""
from pathlib import Path
from types import SimpleNamespace
import argparse,sys,json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/fsx/tmp/marshawn_001660_hd')
TOOLS=ROOT/'final_pair/candidate_inference_code'
sys.path.insert(0,str(TOOLS))
from verify_marshawn_release import runtime
from compare_marshawn_release_grids import candidate_masks,applied_mask,body128
from probe_marshawn_occupancy import digest_model,file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--step',type=int,default=9000)
    parser.add_argument('--candidate-dir',type=Path);parser.add_argument('--body-reference',type=Path)
    parser.add_argument('--comparison-manifest',type=Path)
    args=parser.parse_args();stage=args.stage;step=args.step
    plan=json.loads((stage/'launch_plan.json').read_text())
    if args.checkpoint:checkpoint=args.checkpoint
    else:checkpoint=Path(json.loads((stage/'run/gate_9k.json').read_text())['checkpoint'])
    source_grid_receipt=stage/'grids'/('step-%09d_grid.json'%step)
    exported=json.loads(source_grid_receipt.read_text());assert exported['checkpoint_step']==step
    assert Path(exported['checkpoint']).resolve()==checkpoint.resolve()
    assert file_sha(checkpoint)==exported['checkpoint_sha256']
    candidate_dir=args.candidate_dir or stage/'witness_union_halo1_candidate_9k'
    body_reference=args.body_reference or stage/'gate9000_body128/audit.json'
    native=SimpleNamespace(config=Path(plan['expected_run_dir'])/'config.yml',checkpoint=checkpoint,
        grid=stage/'grids'/('step-%09d_grid.npz'%step),grid_receipt=source_grid_receipt,
        candidate=candidate_dir/'grid.npz',candidate_receipt=candidate_dir/'receipt.json',
        data=None,chunk_size=512,memory_gib=12,
        code_root=[ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',ROOT/'final_pair/code',stage/'code'])
    import torch
    pipe,core,active,boxes,source=runtime(native);pipe.eval();model=pipe.model
    assert model.config.render_step_size==.003 and not model.training
    assert model.config.background_color in ['black','random']
    # Native random-background training renders uncompensated premultiplied RGB
    # in eval, exactly like black. Verify this contract without changing config.
    probe_rgb=torch.tensor([[[.2,.4,.6]]],device=model.device)
    probe_weight=torch.tensor([[[.25]]],device=model.device)
    random_forward=model.renderer_rgb.combine_rgb(probe_rgb,probe_weight,background_color='random')
    black_forward=model.renderer_rgb.combine_rgb(probe_rgb,probe_weight,background_color='black')
    assert torch.equal(random_forward,black_forward)
    before=digest_model(model);masks,candidate=candidate_masks(native,core,model)
    args.output.mkdir(exist_ok=False);(args.output/'raw').mkdir();(args.output/'review').mkdir()
    rows=[];dataset=pipe.datamanager.eval_dataset
    with applied_mask(model,masks['witness_union_halo1']):
        for name in ['DC','HC','KA']:
            ci=[Path(f).stem.split('_')[-1]for f in dataset.image_filenames].index(name)
            camera=dataset.cameras[ci:ci+1].to(model.device)
            assert int(camera.width.item())==1920 and int(camera.height.item())==1012
            started=time.monotonic()
            with torch.inference_mode():
                output=model.get_outputs_for_camera(camera)
                gt=dataset.get_image_float32(ci).to(model.device)
                metrics,_=model.get_image_metrics_and_images(output,dict(image=gt))
            buffers={k:v.detach().float().cpu().numpy()for k,v in output.items()if torch.is_tensor(v)}
            rgb=buffers['rgb'];opacity=buffers['accumulation'];assert rgb.shape==(1012,1920,3)
            assert all(np.isfinite(v).all()for v in buffers.values())
            png=args.output/(name+'.png');raw=args.output/'raw'/(name+'.npz')
            Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(png)
            np.savez_compressed(raw,rgb=rgb,rgba=np.concatenate([rgb,opacity],-1),opacity=opacity,
                depth=buffers['depth'],samples=buffers['num_samples_per_ray'])
            # A separate diagnostic contact; downloadable PNG stays prediction-only.
            pred=Image.open(png).convert('RGB');truth=Image.fromarray(np.rint(np.clip(gt.cpu().numpy(),0,1)*255).astype(np.uint8))
            pred.thumbnail((640,337));truth.thumbnail((640,337));contact=Image.new('RGB',(1280,362),'black')
            contact.paste(truth,(0,25));contact.paste(pred,(640,25));draw=ImageDraw.Draw(contact)
            draw.text((4,5),name+' GT',fill='white');draw.text((644,5),name+f' step{step} witness-union halo1 dt.003',fill='white')
            contact.save(args.output/'review'/(name+'.jpg'),quality=94)
            row=dict(camera=name,metrics={k:float(metrics[k])for k in ['psnr','ssim','lpips']},seconds=time.monotonic()-started,
                prediction_png_sha256=file_sha(png),raw_sha256=file_sha(raw),source_GT=str(dataset.image_filenames[ci]),source_GT_sha256=file_sha(dataset.image_filenames[ci]))
            rows.append(row);(args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=3,last=row),indent=2)+'\n');print(json.dumps(row),flush=True)
        body_summary=body128(model,pipe.datamanager.train_dataset,body_reference,boxes,args.output)
    assert digest_model(model)==before
    old=args.comparison_manifest or ROOT/'native_png_linear_distortion_STAGED/current_best_guarded_HD/manifest.json'
    baseline_manifest=json.loads(old.read_text());baseline=baseline_manifest['guarded_metrics']
    if '0.003'in baseline:baseline=baseline['0.003']
    mean={k:float(np.mean([r['metrics'][k]for r in rows]))for k in ['psnr','ssim','lpips']}
    manifest=dict(source,status='PROVISIONAL_CURRENTBEST_PENDING_ROOT_VISUAL_REVIEW',selected_step=step,
        checkpoint_sha256=file_sha(checkpoint),config_sha256=file_sha(native.config),script_sha256=file_sha(__file__),
        selected_inference_dt=.003,protected_policy=candidate['policy'],candidate_grid_sha256=file_sha(native.candidate),
        candidate_receipt_sha256=file_sha(native.candidate_receipt),candidate=candidate,guarded_metrics=mean,
        comparison_baseline_metrics=baseline,comparison_manifest=str(old),comparison_manifest_sha256=file_sha(old),
        comparison_baseline_step=baseline_manifest['selected_step'],
        delta_vs_comparison_guarded={k:mean[k]-baseline[k]for k in mean},views=rows,
        body128_summary=body_summary,normalization=source['normalization'],shape=[1012,1920,3],
        display_transform='identity',color_space='Unmodified fixed scalar SDR RGB',
        configured_background=model.config.background_color,native_random_eval_equals_black_forward=True,
        raw_RGBA_contract='Actual model RGB codes concatenated with opacity; RGB already contains black-background premultiplication, not straight RGBA. No GT in raw NPZ.',
        prediction_only=True,no_GT_compositing=True,full_original_model_state_restored=True,original_state_sha256=before,
        NN_EMA_checkpoint_unchanged=True,optimizer_updates=0,old_stage_and_ZIP_modified=False,
        limitations=['Known train-witness cell coverage is not complete physical geometry truth.','Remaining floor mottling and detail must be manually assessed; no clean-scene certification.'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),guarded_metrics=mean,delta=manifest['delta_vs_comparison_guarded'])),flush=True)


if __name__=='__main__':main()
