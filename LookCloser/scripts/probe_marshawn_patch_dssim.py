"""Read-only patch DSSIM gradient probe; never creates or steps an optimizer."""
from __future__ import annotations
import argparse,copy,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
STRICT_PARENT_SCALE=False
for directory in [ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',
                  ROOT/'final_pair/code',ROOT/'final_pair/release_tools_v3',Path(__file__).parent]:
    sys.path.insert(0,str(directory))
import numpy as np
import torch
import yaml
from PIL import Image,ImageDraw
from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.engine.callbacks import TrainingCallbackAttributes,TrainingCallbackLocation
from marshawn_patch_dssim_candidate import patch_dssim,validate_patch_indices,TrainBatchCapture
from marshawn_packed_distortion_candidate import PackedDistortionCapture,packet_terms
from probe_marshawn_occupancy import digest_model


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gradient(term,named,scale):
    grads=torch.autograd.grad(term*scale,[p for _,p in named],retain_graph=True,allow_unused=True)
    total=0.;groups={};finite=True
    for (name,_),g in zip(named,grads):
        if g is None:continue
        ok=bool(torch.isfinite(g).all());finite=finite and ok
        if not ok:continue
        sq=float((g.detach().double()/scale).square().sum());total+=sq
        group='geometry'if 'mlp_base'in name else'color'if'mlp_head'in name else'other'
        groups[group]=groups.get(group,0.)+sq
    return dict(finite=finite,l2=total**.5 if finite else None,
                groups_l2={k:v**.5 for k,v in groups.items()},scale=scale)


def measured_gradient(term,named,scale):
    """Keep real parent-scale failure visible; lower-scale norm is diagnostic only."""
    result=gradient(term,named,scale)
    result['parent_scale_finite']=result['finite']
    result['parent_scale']=scale
    if result['finite'] or STRICT_PARENT_SCALE:return result
    attempts=[dict(scale=scale,finite=False)]
    for _ in range(12):
        scale*=.5
        lower=gradient(term,named,scale);attempts.append(dict(scale=scale,finite=lower['finite']))
        if lower['finite']:
            lower.update(parent_scale_finite=False,parent_scale=result['parent_scale'],
                         probe_only_backoff=attempts,production_scaler_modified=False)
            return lower
    result['probe_only_backoff']=attempts
    return result


def main():
    global STRICT_PARENT_SCALE
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=float,nargs='+',default=[.1,.3])
    parser.add_argument('--strict-parent-scale',action='store_true')
    a=parser.parse_args();a.output.mkdir(exist_ok=False);weights=a.weights;STRICT_PARENT_SCALE=a.strict_parent_scale
    torch.set_num_threads(2);cfg=yaml.load(a.config.read_text(),Loader=yaml.Loader)
    assert cfg.max_num_iterations==21000 and cfg.pipeline.patch_dssim_size==11
    text=yaml.dump(cfg);again=yaml.load(text,Loader=yaml.Loader);assert cfg.pipeline._target is again.pipeline._target
    (a.output/'roundtrip.yml').write_text(text)
    checkpoint=Path(cfg.load_checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['step']==8000
    provenance=ROOT/'native_png_linear_distortion_STAGED/finished_receipt.json'
    parent=json.loads(provenance.read_text());assert sha(checkpoint)==parent['checkpoint_sha256']
    assert parent['conservative_attempt_upper_bound']+(21000-8001)==21609<22000
    metadata=json.loads((cfg.pipeline.datamanager.dataparser.data/'transforms.json').read_text())
    # ignore_mask is only justified by actual all-one source masks, including eval.
    masks=[]
    for frame in metadata['frames']:
        path=cfg.pipeline.datamanager.dataparser.data/frame['mask_path']
        array=np.asarray(Image.open(path));assert bool((array==255).all())
        masks.append(dict(path=str(path),sha256=sha(path)))
    assert len(masks)==66
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    report=dict(config_sha256=sha(a.config),checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
                parent_step=8000,optimizer_objects_created=0,optimizer_updates=0,training_authorized=False,
                patch_sampler_distribution_changed=True,dssim_definition='1-functional_SSIM (no half)',
                actual_runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),python=sys.version),
                all66_masks_are_255=True,mask_files=masks,weights_probed=weights,strict_parent_scale=STRICT_PARENT_SCALE)
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            _set_random_seed(42)
            cold=again.pipeline.setup(device='cuda',test_mode='val');cold.train()
            assert cold.dynamic_num_rays_per_batch==128
            assert cold.datamanager.train_pixel_sampler.num_rays_per_batch==121
            callbacks=cold.get_training_callbacks(TrainingCallbackAttributes(None,None,cold,None))
            assert [c.func.__name__ for c in callbacks]==['update_occupancy_grid','guard_after_native_core']
            for c in callbacks:c.run_callback_at_location(0,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
            before=digest_model(cold.model)
            with TrainBatchCapture(cold.datamanager)as capture:
                with torch.autocast(device_type='cuda',enabled=True):out,losses,metrics=cold.get_train_loss_dict(0)
            validate_patch_indices(capture.batch['indices'])
            assert len(out['rgb'])==121 and set(losses)=={'rgb_loss','observed_sfm_free_space','observed_sfm_surface_band','packed_distortion','patch_dssim'}
            named=[(n,p)for n,p in cold.model.named_parameters()if p.requires_grad]
            statuses=[];scale=cfg.grad_scaler_init_scale
            baseline=sum(v for k,v in losses.items()if k!='patch_dssim')
            cold_structural=losses['patch_dssim']/cfg.pipeline.patch_dssim_weight
            for _ in range(12):
                plain=gradient(baseline,named,scale)
                plus={str(w):gradient(baseline+w*cold_structural,named,scale)for w in weights}
                statuses.append(dict(baseline=plain,with_dssim=plus))
                if plain['finite']and all(g['finite']for g in plus.values()):break
                if STRICT_PARENT_SCALE:break
                scale*=.5
            else:raise ValueError('Cold full backward never finite')
            assert digest_model(cold.model)==before
            report['cold']=dict(initial_dynamic_estimate=128,actual_rays=121,packed_samples=int(metrics['num_samples_per_batch']),
                                full_backward=statuses,next_actual_rays=cold.datamanager.train_pixel_sampler.num_rays_per_batch,
                                zero_native_rays=False,no_sampling_cap_changed=True)
            del cold,named,out,losses,metrics,baseline,cold_structural,plus,plain,capture
            torch.cuda.empty_cache()
            _set_random_seed(42);pipe=again.pipeline.setup(device='cuda',test_mode='val')
            old_keys=set(pipe.state_dict());pipe.load_pipeline(saved['pipeline'],8000);pipe.train();model=pipe.model
            assert set(pipe.state_dict())==old_keys==set(saved['pipeline'])
            before=digest_model(model);ds=pipe.datamanager.train_dataset
            by={Path(f).stem.split('_')[-1]:i for i,f in enumerate(ds.image_filenames)}
            selection=ROOT/'floor_ray_diagnosis/selection.json'
            floors=[q for q in json.loads(selection.read_text())['queries']if q['split']=='train']
            chunks=[];labels=[]
            yy,xx=torch.meshgrid(torch.arange(11),torch.arange(11),indexing='ij')
            for q in floors:
                chunks.append(torch.stack([torch.full_like(xx,by[q['camera']]),yy+q['y']-5,xx+q['x']-5],-1).reshape(-1,3))
                labels.append(dict(kind='vetted_train_floor',camera=q['camera'],x=q['x'],y=q['y']))
            # The exact installed standard native sampler generates random patches.
            random_indices=pipe.datamanager.train_pixel_sampler.sample_method(484,len(ds),int(ds.cameras.height[0]),int(ds.cameras.width[0]),device='cpu')
            for group in random_indices.reshape(-1,121,3):
                chunks.append(group);labels.append(dict(kind='native_random_train_patch',camera=Path(ds.image_filenames[int(group[0,0])]).stem.split('_')[-1],x=int(group[60,2]),y=int(group[60,1])))
            indices=torch.cat(chunks).to(model.device);validate_patch_indices(indices)
            truth=[];proof=[]
            for group,label in zip(chunks,labels):
                ci=int(group[0,0]);image=ds.get_data(ci,image_type='float32')['image']
                target=image[group[:,1],group[:,2]].to(model.device);truth.append(target)
                independent=np.asarray(Image.open(ds.image_filenames[ci]).convert('RGB'))[group[:,1].numpy(),group[:,2].numpy()].astype(np.float32)/255
                error=float(np.max(abs(target.cpu().numpy()-independent)));assert error==0
                proof.append(dict(**label,target_pixel_error=error,source_png_sha256=sha(ds.image_filenames[ci])))
            target=torch.cat(truth)
            ray=ds.cameras.to(model.device).generate_rays(indices[:,:1],coords=indices[:,1:].float()+.5,keep_shape=False)
            with PackedDistortionCapture(model)as packed:
                with torch.autocast(device_type='cuda',enabled=True):
                    outputs=model(ray);rgb_loss=model.get_loss_dict(outputs,{'image':target})['rgb_loss']
                linear=packet_terms(packed.packet,pipe._packed_formula,.01,('linear',))['linear']
            actual_scale=float(saved['scalers']['scale'])
            named=[(n,p)for n,p in model.named_parameters()if p.requires_grad]
            rows=[];canvas=Image.new('RGB',(352,8*66),'black');draw=ImageDraw.Draw(canvas)
            for i,label in enumerate(proof):
                sl=slice(i*121,(i+1)*121);pred=outputs['rgb'][sl];gt=target[sl]
                # Terms divided by eight retain the packet mean's production scale.
                mse=((pred.float()-gt.float())**2).mean()/8
                structural=patch_dssim(pred,gt,indices[sl])/8
                m=measured_gradient(mse,named,actual_scale)
                d={str(w):measured_gradient(w*structural,named,actual_scale)for w in weights}
                for g in d.values():g['norm_ratio_to_RGB']=g['l2']/m['l2']if g['finite']and m['finite']and m['l2']else None
                rows.append(dict(**label,RGB=m,DSSIM=d))
                for j,pixels in enumerate([gt,pred]):
                    im=Image.fromarray(np.rint(pixels.detach().float().cpu().numpy().reshape(11,11,3).clip(0,1)*255).astype(np.uint8)).resize((55,55),Image.Resampling.NEAREST)
                    canvas.paste(im,(j*65,i*66))
                draw.text((132,i*66),f'{label["kind"]}\n{label["camera"]} {label["x"]},{label["y"]}',fill='white')
            free,_=pipe._final_free(model.field,8000);ids=pipe._final_free.sample_batch(8000)['indices'];surface,_=pipe._final_band(model.field,8000,ids)
            whole_ssim=patch_dssim(outputs['rgb'],target,indices)
            full_baseline=measured_gradient(rgb_loss+linear+free+surface,named,actual_scale)
            full={str(w):measured_gradient(rgb_loss+linear+free+surface+w*whole_ssim,named,actual_scale)for w in weights}
            assert digest_model(model)==before
            canvas.save(a.output/'training_patch_contact.png');np.savez_compressed(a.output/'actual_patch_packet.npz',indices=indices.cpu().numpy(),prediction=outputs['rgb'].detach().float().cpu().numpy(),target=target.cpu().numpy())
            report.update(passed=True,rows=rows,full_combined_baseline_without_DSSIM=full_baseline,
                          full_combined_backward=full,parent_scaler_scale=actual_scale,
                          all_combined_finite_at_parent_scale=all(v['parent_scale_finite']for v in full.values()),
                          safe_weights_at_parent_scale=[k for k,v in full.items()if v['parent_scale_finite']],
                          parent_has_RNG=('rng_state'in saved),parent_RNG_keys=sorted(saved.get('rng_state',{})),
                          parent_has_optimizer=bool(saved.get('optimizers')),parent_has_schedulers=bool(saved.get('schedulers')),
                          exact_state_keys=True,field_grid_unchanged=True,selection_sha256=sha(selection),
                          budget_upper_bound=21609,proposal_trainer_stop=21000,patch_count=len(labels))
    finally:
        random.setstate(py);np.random.set_state(npstate)
    state=np.random.get_state();assert random.getstate()==py and state[0]==npstate[0]and np.array_equal(state[1],npstate[1])and state[2:]==npstate[2:]
    assert torch.equal(cpu,torch.get_rng_state())and all(torch.equal(x,y)for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
    report['all_Python_NumPy_Torch_RNG_restored']=True
    (a.output/'probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,output=str(a.output),optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
