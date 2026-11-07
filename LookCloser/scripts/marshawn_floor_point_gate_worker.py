"""Inference-only single-point height selector and streamed crop comparison."""
import argparse,hashlib,json,os,time,types
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def install_point_gate(field,cut):
    original=field._normalize_positions
    def normalized(self,positions):
        flat,selector,shape,world=original(positions)
        return flat,selector & (positions[...,2]>=cut).reshape(selector.shape),shape,world
    field._normalize_positions=types.MethodType(normalized,field)
    return original


def main_as_probe():
    import torch
    from PIL import Image
    from nerfstudio.utils.eval_utils import eval_setup
    import probe_marshawn_occupancy as probe
    p=argparse.ArgumentParser()
    for k in ['config','checkpoint','queries','output','data']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--uniform-samples',type=int);a=p.parse_args()
    plan=json.loads(Path(os.environ['FLOOR_POINT_GATE_PLAN']).read_text())
    def configure(cfg):
        cfg.eval_checkpoint=a.checkpoint.resolve();cfg.pipeline.model.occupancy_eval_dilation_radius=0
        cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data
        return cfg
    cfg,pipe,checkpoint,step=eval_setup(a.config,test_mode='val',update_config_callback=configure)
    model=pipe.model;model.current_train_step=step;model.eval()
    assert step==111316 and model._resolved_ray_sampling_mode()=='adaptive'
    assert model.field.feature_reweighting_strength==.3 and model.config.adaptive_coarse_step_size==.00625
    assert int(model.config.occupancy_dilation_radius)==0
    ready=json.loads((a.data/'ready.json').read_text());live=probe.assert_consistent_bounds(model)
    assert all(np.max(np.abs(np.asarray(box)-ready['aabb']))<1e-6 for box in live.values())
    parser=pipe.datamanager.train_dataparser_outputs
    assert parser.dataparser_scale==1. and np.array_equal(parser.dataparser_transform.cpu().numpy(),np.eye(4)[:3])
    before=probe.digest_model(model);core=model.occupancy_grid.binaries.clone();normalizer=model.field._normalize_positions
    a.output.mkdir(parents=True,exist_ok=False);request=json.loads(a.queries.read_text());train=pipe.datamanager.train_dataset;ev=pipe.datamanager.eval_dataset
    names=[Path(f).stem.split('_')[-1]for f in ev.image_filenames];assert names==['DC','HC','KA']
    assert len(request['pixels'])==109 and request['split']=='train'
    records=[];crop_metrics=[];cut=plan['cut'];original_outputs={};starting=time.monotonic()
    def evaluate(camera,coords):
        with torch.inference_mode():
            ray=camera.generate_rays(0,coords=torch.as_tensor(coords,dtype=torch.float32,device=model.device),keep_shape=False)
            result=model(ray)
            return np.concatenate([result['rgb'].detach().float().cpu().numpy(),result['accumulation'].detach().float().cpu().numpy()],axis=-1)
    def render_crop(cam,box):
        x0,y0,x1,y1=box;yy,xx=np.mgrid[y0:y1,x0:x1];coords=np.stack([yy.ravel()+.5,xx.ravel()+.5],-1)
        camera=ev.cameras[names.index(cam):names.index(cam)+1].to(model.device);result=np.empty((len(coords),4),np.float32)
        for start in range(0,len(coords),plan['chunk_rays']):result[start:start+plan['chunk_rays']]=evaluate(camera,coords[start:start+plan['chunk_rays']])
        return result.reshape(y1-y0,x1-x0,4)
    try:
        # All baseline crops pass parity before any modified-density render.
        for cam,box in plan['regions'].items():
            rgba=render_crop(cam,box);original_outputs[cam]=rgba
            target=np.asarray(Image.open(plan['reference_PNGs'][cam]).convert('RGB'))[box[1]:box[3],box[0]:box[2]]
            error=np.abs(rgba[...,:3].astype(float)*255-target)
            metric=dict(camera=cam,pixels=int(error.size//3),max_float_PNG_byte_error=float(error.max()),mean_float_PNG_byte_error=float(error.mean()))
            crop_metrics.append(metric);(a.output/'baseline_parity.json').write_text(json.dumps(crop_metrics,indent=2)+'\n')
            if metric['max_float_PNG_byte_error']>1.01:raise ValueError('Original native crop/archived PNG parity failed')
            Image.fromarray(np.clip(np.rint(rgba[...,:3]*255),0,255).astype(np.uint8)).save(a.output/(cam+'_original.png'))
            np.savez_compressed(a.output/(cam+'_original.npz'),rgba=rgba)
        for pixel in request['pixels']:
            camera=train.cameras[pixel['camera']:pixel['camera']+1].to(model.device)
            rgba=evaluate(camera,[[pixel['y']+.5,pixel['x']+.5]])[0]
            witness=np.array(pixel['witness_xyz']);assert witness.shape==(3,)
            records.append(dict(pixel=pixel,original_RGBA=rgba.tolist(),measured_point_retained=bool(witness[2]>=cut)))
        if not all(r['measured_point_retained']for r in records):raise ValueError('Measured actor witness rejected by proposed point gate')
        old=install_point_gate(model.field,cut);assert old==normalizer
        for cam,box in plan['regions'].items():
            rgba=render_crop(cam,box);base=original_outputs[cam]
            Image.fromarray(np.clip(np.rint(rgba[...,:3]*255),0,255).astype(np.uint8)).save(a.output/(cam+'_gated.png'));np.savez_compressed(a.output/(cam+'_gated.npz'),rgba=rgba)
            delta=np.abs(rgba-base);crop_metrics[names.index(cam)].update(gated_max_RGB_change=float(delta[...,:3].max()),mean_abs_RGB_change=float(delta[...,:3].mean()),
                RGB_changed_over_one_byte=int((delta[...,:3].max(-1)>1/255).sum()),original_alpha_sum=float(base[...,3].astype(float).sum()),gated_alpha_sum=float(rgba[...,3].astype(float).sum()))
        for record in records:
            pixel=record['pixel'];camera=train.cameras[pixel['camera']:pixel['camera']+1].to(model.device)
            rgba=evaluate(camera,[[pixel['y']+.5,pixel['x']+.5]])[0];oldrgba=np.array(record['original_RGBA'])
            record.update(gated_RGBA=rgba.tolist(),max_abs_RGB_change=float(np.abs(rgba[:3]-oldrgba[:3]).max()),alpha_change=float(rgba[3]-oldrgba[3]))
    finally:model.field._normalize_positions=normalizer
    assert torch.equal(core,model.occupancy_grid.binaries)and probe.digest_model(model)==before
    # Confirm removing the temporary method restores a selected original ray.
    px=request['pixels'][0];camera=train.cameras[px['camera']:px['camera']+1].to(model.device);restored=evaluate(camera,[[px['y']+.5,px['x']+.5]])[0]
    restore_error=float(np.abs(restored-np.array(records[0]['original_RGBA'])).max());assert restore_error<2e-6
    result=dict(passed=True,checkpoint_sha256=probe.file_sha(checkpoint),step=step,feature_reweighting_strength=.3,cut=cut,
        rule='Additional shared density selector retains queried sample positions with floor Z>=−.02. Original coordinates/hash normalization/core/field weights unchanged.',
        empirical_prior_not_emptyspace_certificate=True,source_state_sha256=before,frozen_state_unchanged=True,accepted_bounds=live,
        actual_worker=str(Path(__file__).resolve()),actual_worker_sha256=sha(Path(__file__)),normalizer_restored_RGBA_error=restore_error,
        crop_metrics=crop_metrics,TRAIN_controls=records,measured_witnesses_retained=len(records),elapsed_seconds=time.monotonic()-starting,
        limitation='Gate changes density seen by adaptive refinement and recomputes transmittance. Sample-center gate is not exact interval-plane clipping.109observations include103declaredclothing rays (not103independentpoints) plus6shoe rays;hairpoint5568 retains mixed-opacity interpretation.')
    (a.output/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,crops=crop_metrics,controls=len(records),max_control_RGB_change=max(r['max_abs_RGB_change']for r in records))),flush=True)
