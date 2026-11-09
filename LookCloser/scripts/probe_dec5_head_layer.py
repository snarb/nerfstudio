#!/usr/bin/env python3
"""Probe foreground-conditional LookCloser depth using a train-derived head box.

No target RGB is used. The box uses fitting hair anchors only. Source matte
comparisons are diagnostics, not additional model training or threshold fitting.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np,torch,yaml
from dec5_hair import unproject,packed_depth_quantile,packed_layer_depth_quantile
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha


def head_bounds(anchors,frames):
    a=np.load(anchors);by={f['physical_camera']:f for f in frames};xyz=np.empty((len(a['uv']),3),np.float32)
    for index,name in enumerate(a['camera_names']):
        mask=a['camera']==index;xyz[mask]=unproject(a['uv'][mask],a['camera_z'][mask],by[str(name)])
    fit=xyz[~a['validation'].astype(bool)];bounds=np.stack((fit.min(0),fit.max(0)))
    margin=.1*(bounds[1]-bounds[0]).max();bounds+=np.array([[-margin],[margin]])
    validation=xyz[a['validation'].astype(bool)];coverage=float(((validation>=bounds[0])&(validation<=bounds[1])).all(-1).mean())
    return bounds,coverage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','data','training','checkpoint','anchors','mattes','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(12*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    cfg=yaml.load(next(a.training.rglob('config.yml')).read_text(),Loader=yaml.Loader)
    cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data
    pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();model=pipe.model;model.current_train_step=state['step'];before=fingerprint(model)
    payload=json.loads((a.data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in payload['frames']}
    train=pipe.datamanager.train_dataset;train_frames=[by[Path(p).name] for p in train.image_filenames]
    raw=np.asarray([f['transform_matrix'] for f in train_frames]);normal=train.cameras.camera_to_worlds.cpu().numpy()
    rotation=normal[0,:3,:3]@raw[0,:3,:3].T;scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);translation=normal[0,:3,3]-scale*rotation@raw[0,:3,3]
    if np.linalg.norm(scale*raw[:,:3,3]@rotation.T+translation-normal[:,:3,3],axis=-1).max()>1e-4:raise ValueError('Camera similarity mismatch')
    bounds,coverage=head_bounds(a.anchors,train_frames);tb=torch.as_tensor(bounds,device='cuda',dtype=torch.float32)
    R=torch.as_tensor(rotation,device='cuda',dtype=torch.float32);T=torch.as_tensor(translation,device='cuda',dtype=torch.float32)
    rng=np.random.default_rng(42);rows=[]
    for partition,dataset in [('train',train),('validation',pipe.datamanager.eval_dataset)]:
        for index,path in enumerate(dataset.image_filenames):
            f=by[Path(path).name];name=f['physical_camera']
            if partition=='train' and name not in ['E004_C005_1210YM','F004_C005_121059','G004_C005_121037']:continue
            alpha=None
            if partition=='train':
                sem=np.load(a.root/'semantics'/(name+'.npz'))['probability'][...,0].astype(np.float32)/65535
                data=np.load(a.mattes/(name+'.npz'));x0,y0,x1,y1=data['portrait_crop'].astype(int)
                upright=np.zeros((f['w'],f['h']),np.float32);known=np.zeros_like(upright)
                upright[y0:y1,x0:x1]=data['alpha'].astype(np.float32);known[y0:y1,x0:x1]=1
                alpha=np.ascontiguousarray(np.rot90(upright,-1));known=np.ascontiguousarray(np.rot90(known,-1)).astype(bool)
                roi=(cv2.dilate((sem>.1).astype(np.uint8),np.ones((31,31),np.uint8))>0)&known
                groups=[roi&(alpha>.01)&(alpha<.99),roi&(alpha>=.99),roi&(alpha<=.01)]
                quotas=[2048,1024,1024]
            else:
                parent=np.load(a.root/'c_render_base'/(name+'.npz'));roi=parent['roi'];hair=parent['hair_probability']>.5
                boundary=(cv2.dilate(hair.astype(np.uint8),np.ones((11,11),np.uint8))!=cv2.erode(hair.astype(np.uint8),np.ones((11,11),np.uint8)))
                groups=[roi&boundary,roi&~boundary];quotas=[2048,2048]
            flat=[]
            for group,n in zip(groups,quotas):
                values=np.flatnonzero(group);flat.extend(rng.choice(values,min(n,len(values)),replace=False).tolist())
            flat=np.asarray(flat);yy,xx=np.unravel_index(flat,roi.shape);uv=np.stack((xx+.5,yy+.5),-1)
            camera=dataset.cameras[index:index+1].to('cuda');global_depth=[];head_depth=[];opacities=[];outside=[]
            for begin in range(0,len(uv),256):
                pixels=uv[begin:begin+256];rays=camera.generate_rays(0,coords=torch.as_tensor(pixels[:,::-1].copy(),device='cuda',dtype=torch.float32),keep_shape=False)
                with torch.no_grad():result=model(rays)
                ids=result['packed_ray_indices'].long();starts=result['packed_spacing_starts'].reshape(-1);ends=result['packed_spacing_ends'].reshape(-1)
                distance=rays.nears.reshape(-1)[ids]+((starts+ends)/2)*(rays.fars-rays.nears).reshape(-1)[ids]
                position=(rays.origins[ids]+rays.directions[ids]*distance[:,None]-T)@R/scale
                inside=((position>=tb[0])&(position<=tb[1])).all(-1).cpu().numpy()
                values=[result[k].detach().float().cpu().numpy() for k in ['packed_spacing_starts','packed_spacing_ends','packed_weights']]
                index_np=ids.cpu().numpy();near=rays.nears.cpu().numpy();far=rays.fars.cpu().numpy()
                gd=packed_depth_quantile(*values,index_np,near,far)
                hd,mass=packed_layer_depth_quantile(*values,index_np,near,far,inside,minimum_mass=.01)
                gp=(rays.origins.cpu().numpy()+rays.directions.cpu().numpy()*gd[:,None]-translation)@rotation/scale
                outside.append(~((gp>=bounds[0])&(gp<=bounds[1])).all(-1));global_depth.append(gd);head_depth.append(hd);opacities.append(mass)
            gd,hd,mass,out=[np.concatenate(x) for x in [global_depth,head_depth,opacities,outside]]
            row=dict(camera=name,partition=partition,rays=len(uv),head_opacity_quantiles=np.quantile(mass,[0,.1,.5,.9,1]).tolist(),
                global_median_outside_head_fraction=float(out.mean()),head_valid_fraction=float((hd>0).mean()),
                global_background_despite_head_layer_fraction=float((out&(mass>=.01)).mean()),
                global_background_despite_head_mass_gt_point1_fraction=float((out&(mass>=.1)).mean()))
            if alpha is not None:
                reference=alpha[yy,xx];soft=(reference>.01)&(reference<.99);opaque=reference>=.99;empty=reference<=.01
                row.update(head_opacity_mae=float(np.abs(np.clip(mass,0,1)-reference).mean()),
                    soft_opacity_mae=float(np.abs(np.clip(mass[soft],0,1)-reference[soft]).mean()) if soft.any() else None,
                    opaque_hair_mean_mass=float(mass[opaque].mean()) if opaque.any() else None,
                    empty_mean_mass=float(mass[empty].mean()) if empty.any() else None)
            np.savez_compressed(a.output/(name+'.npz'),uv=uv,global_ray_depth=gd,head_ray_depth=hd,head_opacity=mass,global_outside=out)
            rows.append(row);atomic_json(a.output/'progress.json',dict(rows=rows))
    if fingerprint(model)!=before:raise ValueError('Layer probe mutated learned state')
    atomic_json(a.output/'metrics.json',dict(rows=rows,head_bounds=bounds.tolist(),bounds_rule='fit hair anchors min/max plus 10 percent of largest extent',withheld_anchor_coverage=coverage,
        checkpoint_sha256=sha(a.checkpoint),anchors_sha256=sha(a.anchors),dataset_sha256=sha(a.data/'transforms.json'),
        frozen_state_unchanged=True,uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))

if __name__=='__main__':main()
