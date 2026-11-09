#!/usr/bin/env python3
"""Same frozen R0 field: ordinary halo1 vs observed-witness union then halo1.

HD mode writes prediction-only native1920x1012 DC/HC/KA PNGs and actual ray
evidence. Path mode uses the exact same candidate mask on a declared24-camera
bundle. Original checkpoint/core/EMA are restored and verified after inference.
"""
from pathlib import Path
import argparse,copy,hashlib,json
from contextlib import contextmanager
import numpy as np
from PIL import Image,ImageDraw
from verify_marshawn_release import runtime,write_json,save_crop,profile_metrics,summarize
from marshawn_release_common import (load_camera_bundle,trace_native,ray_box_interval,point_cells)
from marshawn_occupancy_audit import dilate_one
from probe_marshawn_occupancy import file_sha,digest_model


def frozen_without_binary(model):
    digest=hashlib.sha256()
    import torch
    for name,value in sorted(model.state_dict().items()):
        if name.endswith('occupancy_grid.binaries'):continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def candidate_masks(args,core,model):
    record=json.loads(args.candidate_receipt.read_text())
    if file_sha(args.candidate)!=record['grid_archive_sha256'] or file_sha(args.checkpoint)!=record['checkpoint_sha256']:
        raise ValueError('Candidate/checkpoint hash mismatch')
    if file_sha(args.grid)!=record['source_grid_sha256']:raise ValueError('Candidate authoritative source mismatch')
    if record['source_policy']['radius']!=0:raise ValueError('Candidate must start from R0 core, not a stored guard')
    with np.load(args.candidate,allow_pickle=False) as source:
        arrays={key:source[key].copy() for key in ['core','observed_voxel_seed','protected_core','active','ordinary_halo1','ema']}
    if not np.array_equal(arrays['core'],core):raise ValueError('Candidate rewrote actual native core')
    if not np.array_equal(arrays['ema'].reshape(-1),model.occupancy_grid.occs.detach().cpu().numpy()):raise ValueError('Candidate EMA changed')
    protected=core|arrays['observed_voxel_seed']
    if (not np.array_equal(protected,arrays['protected_core']) or
            not np.array_equal(dilate_one(protected),arrays['active']) or
            not np.array_equal(dilate_one(core),arrays['ordinary_halo1'])):
        raise ValueError('Candidate is not union followed by exactly one halo')
    return {'ordinary_halo1':arrays['ordinary_halo1'],'witness_union_halo1':arrays['active']},record


@contextmanager
def applied_mask(model,mask):
    import torch
    before=digest_model(model);fixed=frozen_without_binary(model);saved=model.occupancy_grid.binaries.detach().clone()
    try:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.as_tensor(mask,device=model.device))
        yield
        if frozen_without_binary(model)!=fixed:raise ValueError('NN/EMA changed during inference mask comparison')
    finally:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(saved)
        if digest_model(model)!=before:raise ValueError('Original frozen model state was not restored')


def body128(model,dataset,reference,boxes,output):
    import torch
    source=json.loads(reference.read_text());by={Path(p).stem.split('_')[-1]:i for i,p in enumerate(dataset.image_filenames)}
    if source['selection_sha256']!='06e131767fb9ecd09b15eb28664a1917b8b1c21d4b976dd288599d5953628cab':raise ValueError('Original matched BODY128 reference required')
    rows=[];buffers={}
    for number,original in enumerate(source['rows']):
        ci=by[original['camera']];coords=original['uv'];camera=dataset.cameras[ci:ci+1].to(model.device)
        ray=camera.generate_rays(0,coords=torch.tensor([[coords[1],coords[0]]],device=model.device),keep_shape=False)
        near,far=ray_box_interval(ray.origins[0].cpu().numpy(),ray.directions[0].cpu().numpy(),boxes[0],model.config.near_plane,model.config.far_plane)
        trace=trace_native(model,ray);values=profile_metrics(trace,near,far,original['witness_depth'])
        values['actual_weight_max_error']=trace['weight_max_error']
        rows.append(dict(observation_idx=original['observation_idx'],point_idx=original['point_idx'],camera=original['camera'],profiles={'native':values}))
        for key,value in trace.items():
            if isinstance(value,np.ndarray):buffers[f'r{number:03d}_{key}']=value
    np.savez_compressed(output/'body128_traces.npz',**buffers)
    result=dict(selection_sha256=source['selection_sha256'],rays=128,summary=summarize(rows),rows=rows)
    write_json(output/'body128.json',result);return result['summary']


def hd(args,pipe,masks,candidate,receipt,boxes):
    import torch
    model=pipe.model;dataset=pipe.datamanager.eval_dataset;train=pipe.datamanager.train_dataset
    obs=np.load(args.observations);names=obs['camera_names'];by={Path(p).stem.split('_')[-1]:i for i,p in enumerate(train.image_filenames)}
    results={}
    for label,mask in masks.items():
        output=args.output/label;output.mkdir();rows=[]
        with applied_mask(model,mask):
            for index,(camera,batch) in enumerate(pipe.datamanager.fixed_indices_eval_dataloader):
                with torch.inference_mode():
                    outputs=model.get_outputs_for_camera(camera)
                    metrics,_=model.get_image_metrics_and_images(outputs,batch)
                rgb=outputs['rgb'].detach().float().cpu().numpy()
                if rgb.shape!=(1012,1920,3) or not np.isfinite(rgb).all():raise ValueError('Expected finite native1920x1012 RGB')
                name=Path(dataset.image_filenames[index]).stem.split('_')[-1]
                Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(output/(name+'.png'))
                np.savez_compressed(output/(name+'_buffers.npz'),rgb=rgb,
                    depth=outputs['depth'].detach().float().cpu().numpy(),opacity=outputs['accumulation'].detach().float().cpu().numpy())
                rows.append(dict(camera=name,metrics={k:float(metrics[k]) for k in ['psnr','ssim','lpips']},
                    prediction_png=name+'.png',prediction_png_sha256=file_sha(output/(name+'.png')),
                    source_gt_sha256=file_sha(dataset.image_filenames[index])))
                write_json(args.output/'progress.json',dict(policy=label,hd_views=len(rows),total=3))
            if len(rows)!=3:raise ValueError('Exactly3 held-out views required')
            summary={key:float(np.mean([row['metrics'][key] for row in rows])) for key in ['psnr','ssim','lpips']}
            body=body128(model,train,args.body_reference,boxes,output)
            crops=[]
            for point_id in candidate['ordinary_halo1_point_misses']:
                observation=int(np.flatnonzero(obs['point_indices']==point_id)[0]);name=str(names[obs['camera_indices'][observation]])
                crop=save_crop(model,train,by[name],obs['uv'][observation],output/f'edge_p{point_id}_{name}.jpg',f'{label} point{point_id}')
                crop.update(point_idx=point_id,camera=name,observation_idx=observation);crops.append(crop)
            record=dict(receipt,policy=label,derived_candidate=candidate,all_eval_metrics=summary,views=rows,body128_summary=body,
                edge_crops=crops,display_transform='identity',color_space='Scalar SDR model-native RGB',
                png_layout='Prediction only1920x1012; no GT panels/compositing',no_gt_compositing=True,
                native_nn_and_ema_unchanged=True)
            write_json(output/'receipt.json',record);results[label]=dict(metrics=summary,body128_summary=body,output=str(output))
    baseline=json.loads(args.baseline_summary.read_text())['trials'];baseline=next(row for row in baseline if row['radius']==0)
    if Path(baseline['best_checkpoint']).resolve()!=args.checkpoint.resolve():raise ValueError('Default baseline checkpoint mismatch')
    write_json(args.output/'comparison.json',dict(default_baseline=baseline,baseline_summary_sha256=file_sha(args.baseline_summary),
        inference_candidates=results,learned_field_and_ema_unchanged=True,original_model_restored=True,
        warning='Known witness coverage does not certify unknown/hidden geometry.'))
    print(json.dumps(results))


def path(args,pipe,mask,candidate,receipt):
    import torch
    from nerfstudio.cameras.cameras import Cameras,CameraType
    model=pipe.model;data=args.data or Path(pipe.config.datamanager.dataparser.data)
    meta=json.loads((data/'transforms.json').read_text());train=set(meta['train_filenames']);frames=[f for f in meta['frames'] if f['file_path'] in train]
    arrays,contract=load_camera_bundle(args.camera_bundle,args.camera_bundle_receipt,frames)
    cameras=Cameras(camera_to_worlds=torch.tensor(arrays['camera_to_worlds'],dtype=torch.float32),
        fx=torch.tensor(arrays['fx'],dtype=torch.float32),fy=torch.tensor(arrays['fy'],dtype=torch.float32),
        cx=torch.tensor(arrays['cx'],dtype=torch.float32),cy=torch.tensor(arrays['cy'],dtype=torch.float32),
        width=torch.tensor(arrays['width']),height=torch.tensor(arrays['height']),camera_type=CameraType.PERSPECTIVE)
    cameras.rescale_output_resolution(args.width/int(arrays['width'][0]));cameras=cameras.to(model.device);rows=[];images=[]
    with applied_mask(model,mask):
        for index in range(24):
            with torch.inference_mode():outputs=model.get_outputs_for_camera(cameras[index:index+1])
            buffers={key:outputs[key].detach().float().cpu().numpy() for key in ['rgb','depth','accumulation']}
            if not all(np.isfinite(value).all() for value in buffers.values()):raise ValueError('Nonfinite inner path output')
            image=Image.fromarray(np.rint(np.clip(buffers['rgb'],0,1)*255).astype(np.uint8));image.save(args.output/f'view_{index:02d}.png');images.append(image)
            np.savez_compressed(args.output/f'view_{index:02d}_buffers.npz',**buffers)
            rows.append(dict(index=index,camera_to_world=cameras.camera_to_worlds[index].cpu().tolist(),
                rendered_intrinsics={key:float(getattr(cameras,key)[index].item()) for key in ['fx','fy','cx','cy']},
                prediction_sha256=file_sha(args.output/f'view_{index:02d}.png')))
            write_json(args.output/'progress.json',dict(rendered=index+1,total=24))
    def sheet(indices,columns,width,filename):
        height=round(images[0].height*width/images[0].width);canvas=Image.new('RGB',(columns*width,int(np.ceil(len(indices)/columns))*(height+24)),'#202020');draw=ImageDraw.Draw(canvas)
        for i,index in enumerate(indices):
            x,y=i%columns*width,i//columns*(height+24);canvas.paste(images[index].resize((width,height)),(x,y+24));draw.text((x+3,y+4),f'inner{index:02d} R0theta8k witness+1',fill='white')
        canvas.save(args.output/filename,quality=95)
    sheet(range(24),6,320,'contact24.jpg')
    for start in range(0,24,8):sheet(range(start,start+8),2,args.width,f'page_{start:02d}_{start+7:02d}.jpg')
    write_json(args.output/'receipt.json',dict(receipt,policy='witness_union_halo1',derived_candidate=candidate,
        camera_bundle_contract=contract,camera_bundle_sha256=file_sha(args.camera_bundle),views=rows,
        display_transform='identity',prediction_only=True,no_gt_compositing=True,
        original_model_restored=True,learned_field_and_ema_unchanged=True))
    print(json.dumps(dict(output=str(args.output),views=24)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['hd','path'])
    for name in ['config','checkpoint','grid','grid-receipt','candidate','candidate-receipt','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--code-root',type=Path,action='append',default=[])
    p.add_argument('--chunk-size',type=int,default=512);p.add_argument('--memory-gib',type=float,default=24)
    p.add_argument('--observations',type=Path);p.add_argument('--body-reference',type=Path);p.add_argument('--baseline-summary',type=Path)
    p.add_argument('--camera-bundle',type=Path);p.add_argument('--camera-bundle-receipt',type=Path);p.add_argument('--width',type=int,default=640)
    a=p.parse_args();pipe,core,active,boxes,receipt=runtime(a);before=digest_model(pipe.model)
    masks,record=candidate_masks(a,core,pipe.model);a.output.mkdir(parents=True,exist_ok=False)
    receipt.update(comparison_script_sha256=file_sha(__file__),candidate_receipt_sha256=file_sha(a.candidate_receipt),
        original_state_sha256=before,normalization='Prepared-world identity; no additional transform or scale')
    if a.mode=='hd':hd(a,pipe,masks,record,receipt,boxes)
    else:path(a,pipe,masks['witness_union_halo1'],record,receipt)
    if digest_model(pipe.model)!=before:raise ValueError('Final original state mismatch')


if __name__=='__main__':main()
