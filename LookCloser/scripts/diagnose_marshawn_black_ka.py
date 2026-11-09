"""Read-only archived KA black-ROI rays/crops: core, ONE halo, all cells, dense.

No training, GT compositing, neural/config/checkpoint edits or physical carve.
Dense high-sigma samples excluded by the source mask are explicitly exported.
"""
from pathlib import Path
import argparse,json,sys,copy
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def main(a):
    for p in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',a.stage/'code']:
        sys.path.insert(0,str(p))
    import torch
    from PIL import Image,ImageDraw
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from nerfstudio.cameras.rays import RaySamples,Frustums
    from marshawn_release_common import trace_native,authoritative_grid,point_cells,ray_box_interval
    from marshawn_occupancy_audit import dilate_one
    from probe_marshawn_occupancy import file_sha,digest_model,integrate
    torch.set_num_threads(2);selection=json.loads(a.selection.read_text())
    def configure(cfg):cfg.eval_checkpoint=a.checkpoint;return cfg
    cfg,pipe,loaded,step=eval_setup(a.config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);pipe.eval();model=pipe.model
    if model.config.render_step_size!=.003 or model.config.background_color!='black':raise ValueError('Exact.003blacksource required')
    folder=a.stage/'grids';stem=f'step-{step:09d}_grid';core,active,ema,boxes,policy=authoritative_grid(folder/(stem+'.npz'),folder/(stem+'.json'),a.checkpoint)
    before=digest_model(model);saved=model.occupancy_grid.binaries.detach().clone()
    if policy['radius']!=0 or not np.array_equal(saved.cpu().numpy(),core):raise ValueError('Actualsourcecore0 required')
    masks={'native_core0':core,'one_halo1':dilate_one(core),'all_cells':np.ones_like(core)}
    a.output.mkdir(exist_ok=False);ds=pipe.datamanager.eval_dataset;ci=[Path(p).stem.split('_')[-1]for p in ds.image_filenames].index('KA');camera=ds.cameras[ci:ci+1].to(model.device);gt=ds.get_image_float32(ci).cpu().numpy();rows=[];arrays={}
    def analyze(label,t,rgb,o,v):
        st,en,sigma,w=[t[k]for k in ['starts','ends','sigma','weights']];mid=(st+en)/2;xyz=o+mid[:,None]*v;idx,inside,on=point_cells(xyz,core,boxes);_,_,guard=point_cells(xyz,masks['one_halo1'],boxes);opacity=float(w.sum());color=(w[:,None]*rgb).sum(0);rank=int(np.argmax(w));order=np.cumsum(w);quantiles=[]
        for f in [.1,.5,.9]:
            j=min(int(np.searchsorted(order,opacity*f)),len(mid)-1);quantiles.append(dict(q=f,t=float(mid[j]),xyz=xyz[j].tolist(),sigma=float(sigma[j]),cell_xyz=idx[j].tolist(),core=bool(on[j]),halo1=bool(guard[j])))
        record=dict(RGB=color.tolist(),opacity=opacity,conditional_RGB=(color/max(opacity,1e-12)).tolist(),expected_depth=float((w*mid).sum()/max(opacity,1e-12)),samples=len(st),peak=dict(t=float(mid[rank]),xyz=xyz[rank].tolist(),sigma=float(sigma[rank]),weight=float(w[rank]),cell_xyz=idx[rank].tolist(),core=bool(on[rank]),halo1=bool(guard[rank]),EMA=float(ema.reshape(core.shape)[(0,*idx[rank])])if inside[rank]else None),termination_quantiles=quantiles,mass_at_samples_in_source_skipped_cells=float(w[~on].sum()),mass_at_samples_admitted_by_halo_only=float(w[~on&guard].sum()))
        for key,val in dict(t,rgb_samples=rgb,xyz=xyz,cell_xyz=idx,cell_core=on,cell_halo1=guard).items():
            if isinstance(val,np.ndarray):arrays[label+'_'+key]=val
        return record
    try:
        for k,q in enumerate(selection['queries']):
            x,y=q['x'],q['y'];ray=camera.generate_rays(0,coords=torch.tensor([[y+.5,x+.5]],device=model.device),keep_shape=False);o=ray.origins[0].cpu().numpy();v=ray.directions[0].cpu().numpy();record=dict(pixel=[x,y],origin=o.tolist(),direction=v.tolist(),GT_RGB=gt[y,x].tolist(),profiles={})
            for name,mask in masks.items():
                with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.as_tensor(mask,device=model.device))
                colors=[]
                def hook(_m,_args,out):colors.append(out[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(-1,3).copy())
                handle=model.field.register_forward_hook(hook)
                try:trace=trace_native(model,ray)
                finally:handle.remove()
                record['profiles'][name]=analyze(f'r{k:02d}_{name}',trace,colors[0],o,v)
            lo,hi=ray_box_interval(o,v,model.scene_aabb.cpu().numpy().reshape(-1))
            for n in [16384,32768]:
                with torch.no_grad():
                    edges=torch.linspace(lo,hi,n+1,device=model.device);samples=RaySamples(frustums=Frustums(origins=ray.origins.expand(n,3),directions=ray.directions.expand(n,3),starts=edges[:-1,None],ends=edges[1:,None],pixel_area=torch.ones((n,1),device=model.device)),camera_indices=ray.camera_indices.expand(n,1));out=model.field(samples);sig=out[FieldHeadNames.DENSITY].float().cpu().numpy().reshape(-1);rgb=out[FieldHeadNames.RGB].float().cpu().numpy();st,en=edges[:-1].cpu().numpy(),edges[1:].cpu().numpy();alpha,T,w=integrate(st,en,sig)
                record['profiles']['uniform'+str(n)]=analyze(f'r{k:02d}_uniform{n}',dict(starts=st,ends=en,sigma=sig,alpha=alpha,T=T,weights=w),rgb,o,v)
            if file_sha(a.checkpoint)==selection['checkpoint_sha256']:
                error=float(abs(np.array(record['profiles']['native_core0']['RGB'])-q['raw_RGB']).max());record['raw_saved_RGB_parity_error']=error
                if error>2e-5:raise ValueError('SavedactualRGBmismatch '+str(error))
            rows.append(record);(a.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(selection['queries'])))+'\n')
        # Actual native crop at each mask. Render whole small context, not a GT mask.
        bbox=[100,630,370,880];x0,y0,x1,y1=bbox;cam=copy.deepcopy(camera);cam.cx-=x0;cam.cy-=y0;cam.width.fill_(x1-x0);cam.height.fill_(y1-y0);images=[('GT',gt[y0:y1,x0:x1])]
        for name,mask in masks.items():
            with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.as_tensor(mask,device=model.device))
            with torch.inference_mode():out=model.get_outputs_for_camera(cam)
            values={k:v.detach().float().cpu().numpy()for k,v in out.items()if torch.is_tensor(v)};np.savez_compressed(a.output/(name+'_crop_buffers.npz'),**values);images.append((name,values['rgb']))
            Image.fromarray(np.rint(np.clip(values['rgb'],0,1)*255).astype(np.uint8)).save(a.output/(name+'_crop.png'))
        panel=Image.new('RGB',(len(images)*(x1-x0),y1-y0+25),'black');draw=ImageDraw.Draw(panel)
        for j,(name,value)in enumerate(images):panel.paste(Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8)),(j*(x1-x0),25));draw.text((j*(x1-x0)+3,5),name,fill='white')
        panel.save(a.output/'GT_native_halo_allcells.png')
    finally:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(saved)
        if digest_model(model)!=before:raise ValueError('Originalfullstate notrestored')
    np.savez_compressed(a.output/'traces.npz',**arrays);result=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=file_sha(a.checkpoint),config_sha256=file_sha(a.config),selection_sha256=file_sha(a.selection),script_sha256=file_sha(__file__),state_unchanged=True,NN_EMA_checkpoint_unchanged=True,render_step=.003,background='black',grid_source=policy,rows=rows,mask_counts={k:int(v.sum())for k,v in masks.items()},crop_bbox=bbox,no_GT_compositing=True,optimizer_updates=0)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(a.output),rays=len(rows),state_unchanged=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['stage','config','checkpoint','selection','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
